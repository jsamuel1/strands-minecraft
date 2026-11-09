"""Model factory and the CLM SageMaker adapter's wire mapping."""

from __future__ import annotations

import io
import json
from typing import Any

import pytest
from strands.experimental.decisions import Choice, Score, YesNo

from mcbrain.models import SageMakerSystemOneModel, answer_from_wire, describe, question_to_wire, resolve_brain_spec


def test_default_brain_is_sonnet_5_on_bedrock() -> None:
    s = resolve_brain_spec({})
    assert (s.provider, s.model_id, s.is_decision) == ("bedrock", "global.anthropic.claude-sonnet-5", False)


def test_every_decision_provider_resolves() -> None:
    assert resolve_brain_spec({"STRANDS_MODEL_PROVIDER": "typesafe"}).model_id == "jev-latest"
    assert resolve_brain_spec({"STRANDS_MODEL_PROVIDER": "kev"}).base_url == "http://127.0.0.1:8009"
    clm = resolve_brain_spec({"STRANDS_MODEL_PROVIDER": "clm-sagemaker"})
    assert clm.label == "clm-v0.1-8b@sagemaker:clm-v01-8b" and clm.region == "us-west-2"
    assert resolve_brain_spec({"STRANDS_MODEL_PROVIDER": "mantle"}).model_id == "openai.gpt-5.6-terra"
    hob = resolve_brain_spec({"STRANDS_MODEL_PROVIDER": "hobson-sagemaker"})
    assert hob.is_decision and hob.endpoint == "hobson-2b-v19" and hob.label == "hobson-2b-v19@sagemaker:hobson-2b-v19"
    kev = resolve_brain_spec({"STRANDS_MODEL_PROVIDER": "kev-sagemaker"})
    assert kev.is_decision and kev.endpoint == "kev-4b" and kev.label == "kev-4b@sagemaker:kev-4b"


def test_unknown_provider_names_the_valid_ones() -> None:
    with pytest.raises(ValueError, match="bedrock \\| mantle"):
        resolve_brain_spec({"STRANDS_MODEL_PROVIDER": "gemini"})


def test_describe_says_its_number() -> None:
    assert "decision policy" in describe(resolve_brain_spec({"STRANDS_MODEL_PROVIDER": "kev"}))


def test_questions_to_typesafe_wire() -> None:
    assert question_to_wire(Choice("pick", {"a": "A", "b": None})) == {
        "type": "choice",
        "instructions": "pick",
        "criteria": {"a": "A", "b": None},
    }
    assert question_to_wire(Score("rate", ["lo", "hi"]))["criteria"] == ["lo", "hi"]
    assert question_to_wire(YesNo("ok?")) == {"type": "noul", "instructions": "ok?"}


def test_answers_from_wire_and_unknown_types_refused() -> None:
    c = answer_from_wire({"type": "choice", "choice": "b", "confidence": 0.9, "probabilities": {"a": 0.1, "b": 0.9}})
    assert c.choice == "b"  # type: ignore[union-attr]
    n = answer_from_wire({"type": "noul", "noul": 0.8})
    assert n.probability == 0.8 and round(n.confidence or 0, 2) == 0.6  # type: ignore[union-attr]
    s = answer_from_wire({"type": "score", "score": 1.2, "confidence": 0.7, "probabilities": {"0": 0.1, "1": 0.6, "2": 0.3}})
    assert s.probabilities[1] == 0.6  # type: ignore[union-attr]
    with pytest.raises(ValueError):
        answer_from_wire({"type": "extract"})


class FakeRuntime:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.sent: list[dict[str, Any]] = []

    def invoke_endpoint(self, **kw: Any) -> dict[str, Any]:
        self.sent.append(kw)
        return {"Body": io.BytesIO(json.dumps(self.payload).encode())}


async def test_sagemaker_adapter_sends_typesafe_wire_and_keeps_server_latency() -> None:
    rt = FakeRuntime(
        {
            "model": "clm-latest",
            "answers": {"q": {"type": "choice", "choice": "x", "confidence": 0.8, "probabilities": {"x": 0.9, "y": 0.1}}},
            "usage": {"input_tokens": 42, "output_tokens": 0},
            "server_latency_ms": 12.5,
        }
    )
    m = SageMakerSystemOneModel(endpoint_name="clm-v01-8b", client=rt)
    r = await m.ask({"goal": "g"}, {"q": Choice("pick", {"x": None, "y": None})})
    assert r.answers["q"].choice == "x" and r.model_id == "clm-latest"  # type: ignore[union-attr]
    assert r.usage["inputTokens"] == 42
    assert m.last_server_latency_ms == 12.5 and m.last_client_latency_ms is not None
    body = json.loads(rt.sent[0]["Body"])
    assert rt.sent[0]["EndpointName"] == "clm-v01-8b"
    assert body["questions"]["q"]["type"] == "choice" and body["state"] == {"goal": "g"}
