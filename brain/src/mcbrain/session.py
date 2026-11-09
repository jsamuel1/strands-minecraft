"""Concurrent conversations over ONE shared history — fork and fold. Port of ``src/session.ts``.

One Agent must not run two turns at once: interleaved toolUse/toolResult pairs are a corrupt
transcript. So a second request does not queue, it FORKS: a fresh Agent seeded with a snapshot
of the history runs concurrently and its finished exchange is folded back in completion order.

The invariants that were each once a live bug, kept:

- Fork boundaries are IDENTITY-based (``tracking_id`` sets), never positional.
- A fork does not manage its own window (``NullConversationManager``); the session trims once,
  after folding, so a long fork cannot eat its own output on the way home.
- A fork seeds from the last point the root history was COMPLETE (the root turn's start), so it
  can never see — and answer — another rail's pending prompt (issue #25).
- A fold that completes while the root turn runs is parked and lands after the root's seam
  closes, never between its toolUse and toolResult.
- A throttle buys exactly one jittered retry; if our text already landed, the retry resumes from
  history instead of re-asking (two consecutive user messages is its own ValidationException).
- The empty-text brick (#39) is healed in place and the same turn retried.
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Callable
from typing import Any

from strands import Agent
from strands.agent.conversation_manager import NullConversationManager
from strands.types.content import Message, Messages

from .config import cfg
from .doctor import audit_history, classify_provider_error, diagnose, heal_empty_text

logger = logging.getLogger(__name__)

TRIM_ANCHOR = "[earlier conversation trimmed]"

ForkFactory = Callable[[Messages, NullConversationManager], Agent]


def _has_tool_result(m: Message) -> bool:
    return any("toolResult" in b for b in m.get("content") or [])


def _is_clean_user(m: Message) -> bool:
    return m.get("role") == "user" and not _has_tool_result(m)


def choose_cut(msgs: Messages, window: int) -> tuple[int, bool]:
    """A TOTAL sliding-window cut: always reduces an over-window history (see session.ts).

    Forward to the next clean user message; else backward; else a pair-safe cut at the ideal
    point with a synthesized user anchor. Returns ``(cut, needs_anchor)``.
    """
    if len(msgs) <= window:
        return 0, False
    ideal = len(msgs) - window
    for i in range(ideal, len(msgs)):
        if _is_clean_user(msgs[i]):
            return i, False
    for i in range(ideal - 1, 0, -1):
        if _is_clean_user(msgs[i]):
            return i, False
    cut = ideal
    while cut < len(msgs) and _has_tool_result(msgs[cut]):
        cut += 1
    if cut >= len(msgs):
        return 0, False
    return cut, True


def _ids(msgs: Messages) -> set[str]:
    return {str(m["tracking_id"]) for m in msgs if m.get("tracking_id")}


class Session:
    def __init__(
        self,
        agent: Agent,
        fork_factory: ForkFactory,
        *,
        window: int | None = None,
        throttle_retry_ms: int | None = None,
    ) -> None:
        self.agent = agent
        self.fork_factory = fork_factory
        self.window = window if window is not None else cfg.session.window
        self.throttle_retry_ms = throttle_retry_ms if throttle_retry_ms is not None else cfg.session.throttle_retry_ms
        self._active = 0
        self._root_busy = False
        self._root_turn_start: int | None = None
        self._pending_folds: list[Messages] = []
        self.on_diagnosis: Callable[[str], None] = lambda report: logger.warning("%s", report)
        #: Every turn's accumulated usage/latency, appended by ``ask`` — the benchmark reads it.
        self.turns: list[dict[str, Any]] = []
        self.forks = 0

    @property
    def messages(self) -> Messages:
        return self.agent.messages

    @property
    def busy(self) -> int:
        return self._active

    async def _invoke_watched(self, agent: Agent, text: str) -> str:
        before = _ids(agent.messages)

        def landed() -> bool:
            return any(m.get("tracking_id") not in before for m in agent.messages)

        async def resume_or_ask() -> Any:
            return await agent.invoke_async(None if landed() else text)

        try:
            result = await agent.invoke_async(text)
        except Exception as err:  # noqa: BLE001 - classified below, re-raised when not ours to heal
            did_land = landed()
            p = classify_provider_error(err)
            self.on_diagnosis(diagnose(err, agent.messages))
            if p.kind == "history-empty-text":
                stripped, removed = heal_empty_text(agent.messages)
                if stripped or removed:
                    self.on_diagnosis(
                        f"[session] healed empty text: stripped {stripped or 'none'}, dropped {removed or 'none'} — retrying the turn"
                    )
                    result = await resume_or_ask()
                    self._record(agent, result)
                    return str(result)
                self.on_diagnosis("[session] empty-text refusal but no empty block in this history — upstream (fork seed) or trimmed")
            if not p.retryable:
                raise
            jittered = round(self.throttle_retry_ms * (0.5 + random.random()))
            await asyncio.sleep(jittered / 1000)
            self.on_diagnosis(
                f"[session] retrying once after {p.kind} ({p.name}){' — resuming from history, not re-asking' if did_land else ''}"
            )
            result = await agent.invoke_async(None if did_land else text)
        self._record(agent, result)
        return str(result)

    def _record(self, agent: Agent, result: Any) -> None:
        metrics = getattr(result, "metrics", None)
        # THIS invocation's usage: accumulated_usage is the agent's lifetime total, so on the
        # long-lived root it would count every earlier turn again.
        invocations = getattr(metrics, "agent_invocations", None) or []
        usage = dict(invocations[-1].usage) if invocations else dict(getattr(metrics, "accumulated_usage", {}) or {})
        latency = (getattr(metrics, "accumulated_metrics", {}) or {}).get("latencyMs")
        cycles = getattr(metrics, "cycle_count", None)
        tools = {name: m.call_count for name, m in (getattr(metrics, "tool_metrics", {}) or {}).items()}
        self.turns.append(
            {
                "usage": usage,
                "latencyMs": latency,
                "cycles": cycles,
                "tools": tools,
                "stop": str(getattr(result, "stop_reason", "")),
            }
        )

    def _audit(self, where: str) -> None:
        stripped, removed = heal_empty_text(self.agent.messages)
        if stripped or removed:
            self.on_diagnosis(
                f"[session] HEALED empty text after {where}: stripped {stripped or 'none'}, dropped {removed or 'none'} (issue #39)"
            )
        if not cfg.session.audit:
            return
        a = audit_history(self.agent.messages)
        if not a.ok:
            self.on_diagnosis(f"[session] HISTORY AUDIT FAILED after {where}: {'; '.join(a.problems)}\n[session] shape: {a.shape}")

    async def ask(self, text: str) -> str:
        """Always safe to call concurrently: the first caller runs the root, later callers fork."""
        if not (text or "").strip():
            raise ValueError("session.ask: refusing an empty prompt — an empty text block poisons the whole history (issue #39)")
        if self._active == 0:
            self._active += 1
            self._root_busy = True
            self._root_turn_start = len(self.agent.messages)
            try:
                return await self._invoke_watched(self.agent, text)
            finally:
                self._root_busy = False
                self._root_turn_start = None
                self._flush_folds()
                # The root runs with NullConversationManager too (the TS root kept the SDK's
                # sliding window; one trimmer is simpler and pair-safe): trim here, once.
                self._trim()
                self._audit("root turn")
                self._active -= 1

        self._active += 1
        self.forks += 1
        seed = list(self.agent.messages) if self._root_turn_start is None else list(self.agent.messages[: self._root_turn_start])
        seed_ids = _ids(seed)
        fork = self.fork_factory(seed, NullConversationManager())
        try:
            answer = await self._invoke_watched(fork, text)
            add = [m for m in fork.messages if m.get("tracking_id") not in seed_ids]
            if self._root_busy:
                self._pending_folds.append(add)
            else:
                self.agent.messages.extend(add)
                self._trim()
                self._audit("fold")
            return answer
        finally:
            self._active -= 1

    def _flush_folds(self) -> None:
        if not self._pending_folds:
            return
        folds, self._pending_folds = self._pending_folds, []
        for add in folds:
            self.agent.messages.extend(add)
        self._trim()
        self._audit(f"flush of {len(folds)} parked fold(s)")

    def _trim(self) -> None:
        msgs = self.agent.messages
        if len(msgs) <= self.window:
            return
        cut, needs_anchor = choose_cut(msgs, self.window)
        if cut <= 0:
            logger.warning("[session] window %d exceeded (%d) but no safe cut exists — keeping all", self.window, len(msgs))
            return
        del msgs[:cut]
        if needs_anchor or msgs[0].get("role") != "user":
            msgs.insert(0, Message(role="user", content=[{"text": TRIM_ANCHOR}]))
