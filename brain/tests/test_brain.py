"""The LLM brain's rails against a fake sidecar: notes ride in front, reflexes are gated, peers ignored."""

from __future__ import annotations

import asyncio
from typing import Any

from mcbrain.brain import Brain
from mcbrain.sidecar import SidecarEvent

from .fakes import FakeBody, ScriptedModel, text


def make_brain(model: ScriptedModel, monkeypatch: Any, tmp_path: Any) -> tuple[Brain, FakeBody]:
    monkeypatch.setenv("MEMORY_DIR", str(tmp_path))
    body = FakeBody()
    brain = Brain(model, body, username="StrandsBot", thinker=False, body_tools=[])  # type: ignore[arg-type]
    return brain, body


def ev(kind: str, text_: str, name: str | None = None, seq: int = 1) -> SidecarEvent:
    return SidecarEvent(seq=seq, at=0, kind=kind, text=text_, name=name)


async def test_notes_ride_in_front_of_the_next_turn(monkeypatch: Any, tmp_path: Any) -> None:
    model = ScriptedModel([text("ok")])
    brain, _ = make_brain(model, monkeypatch, tmp_path)
    brain.handle(ev("note", "(system) You DIED and respawned."))
    await brain.run("what now?")
    first_user = model.seen[0][0]["content"][0]["text"]
    assert first_user.startswith("(system) You DIED") and first_user.endswith("what now?")


async def test_reflexes_share_one_gate(monkeypatch: Any, tmp_path: Any) -> None:
    gate = asyncio.Event()
    model = ScriptedModel([text("handled")], gates={0: gate})
    brain, _ = make_brain(model, monkeypatch, tmp_path)
    brain.handle(ev("reflex", "(reflex) You just TOOK DAMAGE", "hurt"))
    brain.handle(ev("reflex", "(reflex) You just TOOK DAMAGE again", "hurt"))
    await asyncio.sleep(0.05)
    gate.set()
    await asyncio.sleep(0.05)
    assert model.index == 1, "a burst of danger buys ONE turn"


async def test_peer_bot_chat_never_buys_a_turn_but_players_do(monkeypatch: Any, tmp_path: Any) -> None:
    model = ScriptedModel([text("hi Cagatay")])
    brain, _ = make_brain(model, monkeypatch, tmp_path)
    brain.handle(ev("peer", "hello friend", "Digger"))
    await asyncio.sleep(0.02)
    assert model.index == 0
    brain.handle(ev("chat", "Cagatay says: come here", "Cagatay"))
    await asyncio.sleep(0.05)
    assert model.index == 1


async def test_gave_up_stops_the_pump(monkeypatch: Any, tmp_path: Any) -> None:
    brain, body = make_brain(ScriptedModel([]), monkeypatch, tmp_path)
    body.queue.append(ev("gave_up", "(system) body gave up"))
    await asyncio.wait_for(brain.pump(), 2)


def test_the_agent_gets_its_own_mcp_client_not_the_harness_one() -> None:
    """v3 run 1: sharing the harness's started client made every agent episode fail to start."""
    from mcbrain.sidecar import Body

    body = Body("StrandsBot", "http://127.0.0.1:9")
    assert body.agent_mcp_client() is not body.mcp_client()
    assert body.agent_mcp_client() is not body.agent_mcp_client(), "a fresh client per agent"
