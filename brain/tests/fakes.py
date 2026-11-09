"""Test doubles: a scripted Model (tool-calling), a scripted DecisionModel, and a fake sidecar Body."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator, Mapping, Sequence
from typing import Any

from strands.experimental.decisions import ChoiceAnswer, DecisionModel, DecisionResponse, Question, YesNoAnswer
from strands.models import Model
from strands.types.event_loop import Usage

from mcbrain.sidecar import SidecarEvent, ToolOutcome


class ScriptedModel(Model):
    """Streams pre-defined assistant messages in order; can raise, or wait on an asyncio.Event first."""

    def __init__(self, responses: Sequence[Mapping[str, Any] | BaseException], gates: Mapping[int, asyncio.Event] | None = None) -> None:
        self.responses = list(responses)
        self.gates = dict(gates or {})
        self.index = 0
        self.seen: list[list[dict[str, Any]]] = []

    def update_config(self, **model_config: Any) -> None:
        pass

    def get_config(self) -> Any:
        return {"model_id": "scripted"}

    async def structured_output(
        self, output_model: Any, prompt: Any, system_prompt: str | None = None, **kwargs: Any
    ) -> AsyncGenerator[Any, None]:
        raise NotImplementedError
        yield  # pragma: no cover

    async def stream(
        self, messages: Any, tool_specs: Any = None, system_prompt: str | None = None, **kwargs: Any
    ) -> AsyncGenerator[Any, None]:
        i = self.index
        self.index += 1
        self.seen.append(json.loads(json.dumps(messages, default=str)))
        if i in self.gates:
            await self.gates[i].wait()
        r = self.responses[i]
        if isinstance(r, BaseException):
            raise r
        yield {"messageStart": {"role": "assistant"}}
        stop = "end_turn"
        for block in r["content"]:
            if "text" in block:
                yield {"contentBlockStart": {"start": {}}}
                yield {"contentBlockDelta": {"delta": {"text": block["text"]}}}
                yield {"contentBlockStop": {}}
            if "toolUse" in block:
                stop = "tool_use"
                tu = block["toolUse"]
                yield {"contentBlockStart": {"start": {"toolUse": {"name": tu["name"], "toolUseId": tu["toolUseId"]}}}}
                yield {"contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps(tu["input"])}}}}
                yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": stop}}
        yield {
            "metadata": {
                "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
                "metrics": {"latencyMs": 1},
            }
        }


def text(t: str) -> dict[str, Any]:
    return {"role": "assistant", "content": [{"text": t}]}


def tool_use(name: str, tid: str, **inp: Any) -> dict[str, Any]:
    return {"role": "assistant", "content": [{"toolUse": {"name": name, "toolUseId": tid, "input": inp}}]}


class ScriptedDecision(DecisionModel):
    """Picks a scripted option id (or the first option) and records every request."""

    def __init__(self, picks: Sequence[str | BaseException] = (), *, calibrated: bool = True) -> None:
        self.picks = list(picks)
        self._calibrated = calibrated
        self.requests: list[tuple[Any, dict[str, Question]]] = []
        self.last_server_latency_ms = 1.5

    @property
    def calibrated(self) -> bool:
        return self._calibrated

    def get_config(self) -> dict[str, Any]:
        return {"model_id": "scripted-s1"}

    def update_config(self, **config: Any) -> None:
        pass

    async def _ask(self, state: Any, questions: Mapping[str, Question], **kwargs: Any) -> DecisionResponse:
        self.requests.append((state, dict(questions)))
        q = questions["next_action"]
        options = list(q.options)  # type: ignore[union-attr]
        pick = self.picks.pop(0) if self.picks else options[0]
        if isinstance(pick, BaseException):
            raise pick
        probs = {o: (0.7 if o == pick else 0.3 / max(1, len(options) - 1)) for o in options}
        answers: dict[str, Any] = {"next_action": ChoiceAnswer(choice=pick, probabilities=probs, confidence=0.6)}
        if "last_step_succeeded" in questions:
            answers["last_step_succeeded"] = YesNoAnswer(probability=0.2, confidence=0.6)
        return DecisionResponse(
            answers=answers,
            model_id="scripted-s1",
            usage=Usage(inputTokens=100, outputTokens=0, totalTokens=100),
        )


class FakeBody:
    """The sidecar surface the brain uses, in memory."""

    def __init__(self, state: dict[str, Any] | None = None, tools: Mapping[str, Any] | None = None) -> None:
        self.name = "StrandsBot"
        self._state = state or {
            "username": "StrandsBot",
            "pos": {"x": 0, "y": 64, "z": 0},
            "health": 20,
            "food": 20,
            "inv": {},
            "foodPortions": 0,
            "isDay": True,
            "digest": "HP 20/20",
            "threats": [],
        }
        self.tools = dict(tools or {})
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.busy: list[bool] = []
        self.queue: list[SidecarEvent] = []

    def state(self) -> dict[str, Any]:
        return json.loads(json.dumps(self._state))

    def call(self, tool: str, **args: Any) -> ToolOutcome:
        self.calls.append((tool, args))
        fn = self.tools.get(tool)
        value = fn(self, **args) if callable(fn) else fn
        if isinstance(value, ToolOutcome):
            return value
        txt = value if isinstance(value, str) else json.dumps(value)
        return ToolOutcome(ok=True, text=txt, data=value if value is not None else txt)

    def blocks(self, positions: list[dict[str, int]]) -> list[dict[str, Any]]:
        solid = self._state.get("_solid", set())
        return [
            {**p, "name": "stone" if (p["x"], p["y"], p["z"]) in solid else "air", "solid": (p["x"], p["y"], p["z"]) in solid}
            for p in positions
        ]

    def set_busy(self, busy: bool) -> None:
        self.busy.append(busy)

    def events(self, wait_ms: int = 0) -> list[SidecarEvent]:
        out, self.queue = self.queue, []
        return out
