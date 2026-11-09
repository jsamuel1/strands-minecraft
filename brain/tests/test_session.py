"""Session fork/fold — the invariants that were each once a live bug (session.ts)."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from strands import Agent, tool
from strands.agent.conversation_manager import NullConversationManager
from strands.types.exceptions import ModelThrottledException

from mcbrain.doctor import audit_history
from mcbrain.session import TRIM_ANCHOR, Session, choose_cut

from .fakes import ScriptedModel, text, tool_use


@tool
def dig(x: int) -> str:
    """Dig a block.

    Args:
        x: where.
    """
    return f"dug {x}"


def make(model: ScriptedModel, **kw: Any) -> Session:
    def agent(seed: list[dict[str, Any]] | None = None, manager: Any = None) -> Agent:
        return Agent(
            model=model,
            tools=[dig],
            messages=seed,
            callback_handler=None,
            conversation_manager=manager or NullConversationManager(),
        )

    return Session(agent(), lambda seed, mgr: agent(seed, mgr), **kw)


async def test_sequential_asks_share_one_history() -> None:
    s = make(ScriptedModel([text("one"), text("two")]))
    assert await s.ask("a") == "one\n"
    assert await s.ask("b") == "two\n"
    assert [m["role"] for m in s.messages] == ["user", "assistant", "user", "assistant"]
    assert s.forks == 0


async def test_concurrent_ask_forks_and_folds_after_the_root_seam_closes() -> None:
    gate = asyncio.Event()
    # root: tool call (gated) then text; fork: answers at once.
    model = ScriptedModel([tool_use("dig", "t1", x=1), text("fork answer"), text("root done")], gates={0: gate})
    s = make(model)
    root = asyncio.create_task(s.ask("root prompt"))
    await asyncio.sleep(0.05)
    assert s.busy == 1
    fork_answer = await s.ask("fork prompt")
    assert fork_answer.strip() == "fork answer"
    # The fork completed while the root was mid-turn: its fold is PARKED, not spliced in.
    assert not any(b.get("text") == "fork prompt" for m in s.messages for b in m["content"])
    gate.set()
    await root
    a = audit_history(s.messages)
    assert a.ok, a.problems
    texts = [b.get("text") for m in s.messages for b in m["content"] if "text" in b]
    assert texts.index("root prompt") < texts.index("fork prompt"), "parked fold lands after the root turn"
    assert s.forks == 1


async def test_fork_seeds_from_the_root_turn_start_so_it_cannot_see_a_pending_prompt() -> None:
    """Issue #25: a fork seeded mid-turn answered the ROOT's question to the wrong caller."""
    gate = asyncio.Event()
    model = ScriptedModel([text("earlier"), tool_use("dig", "t9", x=2), text("fork"), text("root")], gates={1: gate})
    s = make(model)
    await s.ask("earlier prompt")
    root = asyncio.create_task(s.ask("PENDING root prompt"))
    await asyncio.sleep(0.05)
    await s.ask("fork prompt")
    fork_view = model.seen[2]
    seen_text = [b.get("text") for m in fork_view for b in m["content"] if "text" in b]
    assert "PENDING root prompt" not in seen_text
    assert "earlier prompt" in seen_text
    gate.set()
    await root


async def test_throttle_buys_one_retry_that_resumes_instead_of_re_asking() -> None:
    model = ScriptedModel([ModelThrottledException("slow down"), text("after retry")])
    s = make(model, throttle_retry_ms=1)
    assert (await s.ask("hello")).strip() == "after retry"
    users = [m for m in s.messages if m["role"] == "user"]
    assert len(users) == 1, "the retry must not append a second user message"


async def test_non_retryable_error_is_loud() -> None:
    s = make(ScriptedModel([ValueError("ValidationException: bad toolResult")]), throttle_retry_ms=1)
    with pytest.raises(ValueError, match="ValidationException"):
        await s.ask("x")


async def test_empty_prompt_is_refused_before_it_poisons_history() -> None:
    s = make(ScriptedModel([]))
    with pytest.raises(ValueError, match="#39"):
        await s.ask("   ")


def _pair(i: int) -> list[dict[str, Any]]:
    return [
        {"role": "assistant", "content": [{"toolUse": {"toolUseId": f"u{i}", "name": "dig", "input": {}}}]},
        {
            "role": "user",
            "content": [{"toolResult": {"toolUseId": f"u{i}", "status": "success", "content": [{"text": "ok"}]}}],
        },
    ]


def test_choose_cut_prefers_the_next_clean_user_message() -> None:
    msgs = [{"role": "user", "content": [{"text": "a"}]}, {"role": "assistant", "content": [{"text": "b"}]}] * 6
    cut, anchor = choose_cut(msgs, 5)
    assert msgs[cut]["role"] == "user" and not anchor


def test_choose_cut_is_total_and_pair_safe_without_any_clean_boundary() -> None:
    msgs = [{"role": "user", "content": [{"text": "start"}]}]
    for i in range(10):
        msgs += _pair(i)
    cut, anchor = choose_cut(msgs, 6)
    assert cut > 0 and anchor, "always reduces; synthesizes an anchor"
    assert "toolResult" not in msgs[cut]["content"][0], "never cut between a toolUse and its toolResult"


async def test_trim_keeps_history_provider_valid() -> None:
    responses = [text(f"r{i}") for i in range(12)]
    s = make(ScriptedModel(responses), window=8)
    for i in range(12):
        await s.ask(f"q{i}")
    assert len(s.messages) <= 9
    assert s.messages[0]["role"] == "user"
    assert audit_history(s.messages).ok


def test_trim_anchor_constant_is_plain_text() -> None:
    assert TRIM_ANCHOR.strip()
