"""The model factory — one place that turns env into a brain, for the primary and every worker.

Two kinds of brain, because the benchmark compares both:

- An **LLM** (``strands.models.Model``) drives the bot the way the TS brain did: free-form
  tool calling over the sidecar's 54 tools. Providers: ``bedrock`` (Claude Sonnet 5) and
  ``mantle`` (OpenAI GPT models on Amazon Bedrock's OpenAI-compatible endpoint).
- A **System One decision model** (``DecisionModel``) cannot call tools. It becomes the
  *policy* (``policy.py``): code builds the legal macro-actions, the model picks one with a
  calibrated ``Choice``. Providers: ``typesafe`` (hosted Jev), ``kev`` (a self-hosted Kev
  server on the same ``/v1/systemone`` wire), ``clm`` (a clm-serve HTTP endpoint, same
  wire) and ``clm-sagemaker`` (CLM behind SageMaker ``InvokeEndpoint`` with SigV4).

An LLM can also be the policy (``LLMDecisionModel``) — the benchmark's control arm that
separates "decision model vs LLM" from "picker vs free tool calling".

``resolve_brain_spec`` is the pure half (env → spec), ``create_*`` the half that builds.
``describe`` prints the choice at boot: a brain nobody can name is a brain nobody can trust.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from strands.experimental.decisions import (
    Answer,
    Choice,
    ChoiceAnswer,
    DecisionModel,
    DecisionResponse,
    DecisionState,
    LLMDecisionModel,
    Question,
    Score,
    ScoreAnswer,
    YesNoAnswer,
    yes_no_confidence,
)
from strands.models import Model
from strands.types.event_loop import Usage
from strands.types.exceptions import ModelThrottledException

from .config import cfg
from .policy import ContextProfile

LLM_PROVIDERS = ("bedrock", "mantle")
#: SageMaker-hosted System One models: provider -> (default endpoint, label template, env prefix).
#: Every container fronts TypeSafe's /v1/systemone body on /invocations (infra/*-sagemaker).
SAGEMAKER_PROVIDERS: dict[str, tuple[str, str, str]] = {
    "clm-sagemaker": ("clm-v01-8b", "clm-v0.1-8b@sagemaker:{endpoint}", "CLM"),
    "kev-sagemaker": ("kev-4b", "kev-4b@sagemaker:{endpoint}", "KEV"),
    # strands-decider: github.com/strands-labs/strands-decider, v19 checkpoint (infra/hobson-sagemaker).
    "hobson-sagemaker": ("hobson-2b-v19", "hobson-2b-v19@sagemaker:{endpoint}", "HOBSON"),
}
DECISION_PROVIDERS = ("typesafe", "kev", "clm", *SAGEMAKER_PROVIDERS)
PROVIDERS = LLM_PROVIDERS + DECISION_PROVIDERS
Provider = Literal["bedrock", "mantle", "typesafe", "kev", "kev-sagemaker", "clm", "clm-sagemaker", "hobson-sagemaker"]

#: Default model ids per provider, overridden by STRANDS_MODEL_ID.
DEFAULT_MODEL_ID: dict[str, str] = {
    "bedrock": "global.anthropic.claude-sonnet-5",
    # "GPT-6 Terra" does not exist on Bedrock (2026-09-26, us-west-2/us-east-2: the gpt-6 line is
    # astra/sol/luna and only astra is invocable on-demand). Terra is gpt-5.6-terra. See docs/bench.
    "mantle": "openai.gpt-5.6-terra",
    "typesafe": "jev-latest",
    "kev": "kev-latest",
    "kev-sagemaker": "kev-latest",
    "clm": "clm-latest",
    "clm-sagemaker": "clm-latest",
    "hobson-sagemaker": "hobson-latest",
}

#: Kev and CLM serve one state row plus one question in a fixed window; checked before send.
KEV_ROW_TOKENS = 8_192
CLM_ROW_TOKENS = 2_048


@dataclass(frozen=True)
class BrainSpec:
    provider: Provider
    model_id: str
    region: str | None = None
    base_url: str | None = None
    endpoint: str | None = None
    profile: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_decision(self) -> bool:
        return self.provider in DECISION_PROVIDERS

    @property
    def label(self) -> str:
        """The name the benchmark records: provider-qualified, and hosting-qualified for CLM."""
        if self.provider in SAGEMAKER_PROVIDERS:
            return SAGEMAKER_PROVIDERS[self.provider][1].format(endpoint=self.endpoint)
        return f"{self.provider}:{self.model_id}"


def resolve_brain_spec(env: Mapping[str, str] | None = None, *, model_id: str | None = None) -> BrainSpec:
    """Env → spec. Raises on an unknown provider, naming the valid ones."""
    env = os.environ if env is None else env
    raw = (env.get("STRANDS_MODEL_PROVIDER") or "bedrock").strip().lower()
    if raw not in PROVIDERS:
        raise ValueError(f"STRANDS_MODEL_PROVIDER={raw!r} is not a provider this brain knows — use one of {' | '.join(PROVIDERS)}")
    provider: Provider = raw  # type: ignore[assignment]
    mid = model_id or (env.get("STRANDS_MODEL_ID") or "").strip() or DEFAULT_MODEL_ID[provider]
    region = (env.get("AWS_REGION") or env.get("AWS_DEFAULT_REGION") or "us-west-2").strip()
    profile = (env.get("AWS_PROFILE") or "").strip() or None
    if provider == "bedrock":
        return BrainSpec(provider, mid, region=region, profile=profile)
    if provider == "mantle":
        return BrainSpec(provider, mid, region=(env.get("MANTLE_REGION") or region).strip(), profile=profile)
    if provider == "typesafe":
        return BrainSpec(provider, mid, base_url=(env.get("TYPESAFE_BASE_URL") or "").strip() or None)
    if provider == "kev":
        return BrainSpec(provider, mid, base_url=(env.get("KEV_BASE_URL") or "http://127.0.0.1:8009").strip())
    if provider == "clm":
        return BrainSpec(provider, mid, base_url=(env.get("CLM_BASE_URL") or "http://127.0.0.1:8700").strip())
    default_endpoint, _label, pre = SAGEMAKER_PROVIDERS[provider]
    return BrainSpec(
        provider,
        mid,
        region=(env.get(f"{pre}_REGION") or "us-west-2").strip(),
        endpoint=(env.get(f"{pre}_ENDPOINT") or default_endpoint).strip(),
        profile=(env.get(f"{pre}_AWS_PROFILE") or env.get("CLM_AWS_PROFILE") or profile or "").strip() or None,
    )


def context_profile(spec: BrainSpec) -> ContextProfile:
    """This model's decision-state operating point (operator, 2026-09-28: CLM lower, the others more)."""
    p = cfg.policy
    if spec.provider in ("clm", "clm-sagemaker"):
        return ContextProfile(p.state_chars, p.history_steps_clm, p.event_chars_clm, p.max_events, False, p.outcome_chars_clm)
    if spec.provider in ("kev", "kev-sagemaker"):
        return ContextProfile(p.state_chars_kev, p.history_steps_kev, p.event_chars, p.max_events, True, p.outcome_chars)
    if spec.provider == "hobson-sagemaker":
        return ContextProfile(p.state_chars_hobson, p.history_steps_hobson, p.event_chars, p.max_events, True, p.outcome_chars)
    return ContextProfile(p.state_chars_jev, p.history_steps_jev, p.event_chars, p.max_events, True, p.outcome_chars)


def describe(spec: BrainSpec) -> str:
    """One boot line (config doctrine: say your number)."""
    where = spec.region or spec.base_url or ""
    kind = "decision policy" if spec.is_decision else "tool-calling LLM"
    return f"🧠 brain {spec.label} · {kind}{f' · {where}' if where else ''}"


def _boto_session(spec: BrainSpec) -> Any:
    import boto3

    return boto3.Session(profile_name=spec.profile, region_name=spec.region) if spec.profile else boto3.Session(region_name=spec.region)


def create_llm(spec: BrainSpec) -> Model:
    """A tool-calling Model for ``bedrock`` or ``mantle``."""
    if spec.provider == "bedrock":
        from strands.models import BedrockModel

        return BedrockModel(model_id=spec.model_id, boto_session=_boto_session(spec))
    if spec.provider == "mantle":
        # The Responses API, not Chat Completions: Mantle refuses function tools with reasoning on
        # /v1/chat/completions for gpt-5.6-terra ("use /v1/responses", observed 2026-09-26).
        from strands.models.openai_responses import OpenAIResponsesModel

        config: dict[str, Any] = {"region": spec.region, "boto_session": _boto_session(spec)}
        return OpenAIResponsesModel(model_id=spec.model_id, bedrock_mantle_config=config)  # type: ignore[arg-type]
    raise ValueError(f"{spec.provider} is a decision model, not a tool-calling LLM — use create_decision_model")


def create_decision_model(spec: BrainSpec) -> DecisionModel:
    """A DecisionModel for any provider; LLM providers are wrapped in ``LLMDecisionModel`` (uncalibrated)."""
    if spec.provider in LLM_PROVIDERS:
        return LLMDecisionModel(create_llm(spec))
    if spec.provider in SAGEMAKER_PROVIDERS:
        return SageMakerSystemOneModel(
            endpoint_name=spec.endpoint or SAGEMAKER_PROVIDERS[spec.provider][0],
            region=spec.region,
            profile=spec.profile,
            model_id=spec.model_id,
        )
    from strands.models.typesafe import KEV_MAX_STATE_PLUS_QUESTION_TOKENS, TypeSafeDecisionModel

    if spec.provider == "typesafe":
        return TypeSafeDecisionModel(model_id=spec.model_id, base_url=spec.base_url)
    if spec.provider == "kev":
        return TypeSafeDecisionModel(
            base_url=spec.base_url,
            api_key=os.environ.get("KEV_API_KEY") or None,
            model_id=spec.model_id,
            max_state_plus_question_tokens=KEV_MAX_STATE_PLUS_QUESTION_TOKENS,
            max_request_tokens=None,
        )
    # clm-serve speaks the same wire; its encoder window is 2,048 tokens per row.
    return TypeSafeDecisionModel(
        base_url=spec.base_url,
        api_key=os.environ.get("CLM_API_KEY") or None,
        model_id=spec.model_id,
        max_state_plus_question_tokens=CLM_ROW_TOKENS,
        max_request_tokens=None,
    )


# ── CLM on SageMaker ──────────────────────────────────────────────────────────


def question_to_wire(question: Question) -> dict[str, Any]:
    """A Choice/Score/YesNo as TypeSafe's ``/v1/systemone`` JSON (the same shape typesafe.py sends)."""
    if isinstance(question, Choice):
        return {"type": "choice", "instructions": question.instructions, "criteria": dict(question.options)}
    if isinstance(question, Score):
        return {"type": "score", "instructions": question.instructions, "criteria": list(question.levels)}
    criteria = {k: v for k, v in (("true", question.true), ("false", question.false)) if v is not None}
    return {"type": "noul", "instructions": question.instructions, **({"criteria": criteria} if criteria else {})}


def answer_from_wire(raw: Mapping[str, Any]) -> Answer:
    """One wire answer → the strands answer type. Unknown types are refused, never guessed."""
    kind = raw.get("type")
    if kind == "noul":
        p = float(raw["noul"])
        return YesNoAnswer(probability=p, confidence=yes_no_confidence(p))
    if kind == "choice":
        return ChoiceAnswer(
            choice=str(raw["choice"]),
            probabilities={str(k): float(v) for k, v in raw["probabilities"].items()},
            confidence=raw.get("confidence"),
        )
    if kind == "score":
        return ScoreAnswer(
            score=float(raw["score"]),
            probabilities={int(k): float(v) for k, v in raw["probabilities"].items()},
            confidence=raw.get("confidence"),
        )
    raise ValueError(f"unsupported System One answer type {kind!r}")


class SageMakerSystemOneModel(DecisionModel):
    """System One decisions from a SageMaker real-time endpoint that fronts clm-serve.

    The container (``infra/clm-sagemaker/gateway.py``) maps ``/invocations`` to clm-serve's
    ``/v1/systemone``, so the body is exactly TypeSafe's wire; ``InvokeEndpoint`` carries it
    under SigV4 instead of a bearer key. The gateway echoes clm-serve's ``X-CLM-Latency-Ms``
    as ``server_latency_ms`` (InvokeEndpoint drops custom headers); it is kept on
    ``last_server_latency_ms`` so the benchmark can separate server time from the round trip.

    Calibrated: CLM's contrastive head outputs a fitted distribution over the supplied options.
    """

    def __init__(
        self,
        *,
        endpoint_name: str,
        region: str | None = None,
        profile: str | None = None,
        model_id: str = "clm-latest",
        client: Any | None = None,
    ) -> None:
        self.config: dict[str, Any] = {"model_id": model_id, "endpoint_name": endpoint_name, "region": region}
        if client is None:
            import boto3
            from botocore.config import Config as BotoConfig

            session = boto3.Session(profile_name=profile, region_name=region) if profile else boto3.Session(region_name=region)
            client = session.client(
                "sagemaker-runtime", config=BotoConfig(retries={"max_attempts": 2, "mode": "standard"}, read_timeout=60)
            )
        self._client = client
        self.last_server_latency_ms: float | None = None
        self.last_client_latency_ms: float | None = None

    @property
    def calibrated(self) -> bool:
        return True

    def get_config(self) -> dict[str, Any]:
        return self.config

    def update_config(self, **config: Any) -> None:
        self.config.update(config)

    async def _ask(self, state: DecisionState, questions: Mapping[str, Question], **kwargs: Any) -> DecisionResponse:
        body = json.dumps(
            {
                "state": state,
                "model": self.config["model_id"],
                "questions": {k: question_to_wire(q) for k, q in questions.items()},
            },
            ensure_ascii=False,
            default=str,
        )
        started = time.perf_counter()
        try:
            resp = await asyncio.to_thread(
                self._client.invoke_endpoint,
                EndpointName=self.config["endpoint_name"],
                ContentType="application/json",
                Accept="application/json",
                Body=body,
            )
        except Exception as error:
            code = getattr(error, "response", {}).get("Error", {}).get("Code", "")
            if code in {"ThrottlingException", "ServiceUnavailable", "ModelNotReadyException"}:
                raise ModelThrottledException(str(error)) from error
            raise
        self.last_client_latency_ms = (time.perf_counter() - started) * 1000
        payload = json.loads(resp["Body"].read())
        if "answers" not in payload:
            raise ValueError(f"CLM endpoint returned no answers: {str(payload)[:300]}")
        self.last_server_latency_ms = payload.get("server_latency_ms")
        usage = payload.get("usage") or {}
        tokens_in = int(usage.get("input_tokens") or 0)
        tokens_out = int(usage.get("output_tokens") or 0)
        return DecisionResponse(
            answers={k: answer_from_wire(v) for k, v in payload["answers"].items()},
            model_id=str(payload.get("model") or self.config["model_id"]),
            usage=Usage(inputTokens=tokens_in, outputTokens=tokens_out, totalTokens=tokens_in + tokens_out),
        )
