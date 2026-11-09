"""The idle mind / journey supervisor. Port of ``src/thinker.ts``.

IDLE (no journey): a small rotating focus plus the digest; SUPERVISOR (journey running): judge
the journal with a COMPUTED progress verdict, never move the legs. A starving bot gets the
survival override instead of the rotation. Undelivered notes make a cycle fire immediately.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Awaitable, Callable
from typing import Any

from .config import cfg
from .journeys import Journey, JourneyRunner, progress_verdict

IDLE_FOCI: tuple[str, ...] = (
    "Survival basics: nearby drops, exposure for the hour it is, food if a meal is in reach — whatever cheap thing keeps you alive.",
    "Security patrol: is where you are settled lit and safe against spawns? check_darkness maps the dark spots and drafts a torch plan; what to do about it — and whether now — is your call.",
    "Logistics: take stock — short on food, torches, tools, or blocks? Fix what is cheap now, journey what is not. Tend waypoints: name the spot you keep returning to if it has no name yet.",
    "Progression: read your status digest and the goal ledgers above. What is the next tech-tree step your bag actually supports? Propose ONE novel task VERIFIABLE from your own state and start a journey for it. Never re-propose a too-hard goal; never re-do a completed one.",
)

SURVIVAL_OVERRIDE = (
    "SURVIVAL OVERRIDE — you are badly hurt and cannot heal: health only regenerates at food >= 18, and nothing else heals you. "
    "Not waiting, not dawn, not a bed (a bed skips night, it does not feed you), not standing in a sealed box. "
    "Do NOT idle waiting for a player to bring food — ask once, then feed yourself. "
    "THIS cycle: get food by whatever route your own state supports; if it is more than a few steps away, start_journey for exactly that. "
    "If hostiles make that suicidal at this health, get safe first — but the goal stays food."
)


def pick_focus(cycle: int, vitals: dict[str, Any] | None = None) -> str:
    if (
        vitals
        and not vitals.get("foodPortions")
        and ((isinstance(vitals.get("health"), (int, float)) and vitals["health"] < 10) or vitals.get("food") == 0)
    ):
        return SURVIVAL_OVERRIDE
    return IDLE_FOCI[cycle % len(IDLE_FOCI)]


class Thinker:
    def __init__(self, session: Any, journeys: JourneyRunner) -> None:
        self.session = session
        self.journeys = journeys
        self.enabled = not cfg.thinker.disabled
        self.interval_ms = cfg.thinker.interval_ms
        self.cycles = 0
        self._last_activity = time.monotonic()
        self._cycling = False
        self._seen: dict[str, int] = {}
        self._task: asyncio.Task[None] | None = None
        self.digest: Callable[[], Awaitable[str]] | None = None
        self.vitals: Callable[[], Awaitable[dict[str, Any]]] | None = None
        self.has_notes: Callable[[], bool] | None = None
        self.take_notes: Callable[[], str] | None = None
        self.on_thought: Callable[[str], None] | None = None

    def touch(self) -> None:
        self._last_activity = time.monotonic()

    def start(self) -> None:
        if self._task or not self.enabled:
            return
        self._task = asyncio.get_running_loop().create_task(self._loop())

    def stop(self) -> None:
        if self._task:
            self._task.cancel()
            self._task = None

    async def _loop(self) -> None:
        tick = max(15.0, self.interval_ms / 3000)
        while True:
            await asyncio.sleep(tick)
            await self.maybe_cycle()

    async def maybe_cycle(self) -> None:
        if self._cycling:
            return
        urgent = self.has_notes() if self.has_notes else False
        if not urgent and (time.monotonic() - self._last_activity) * 1000 < self.interval_ms:
            return
        if self.session.busy > 0:
            return
        self._cycling = True
        self.cycles += 1
        try:
            notes = self.take_notes() if self.take_notes else ""
            j = self.journeys.running
            base = await (self.supervisor_prompt(j) if j else self.idle_prompt())
            thought = await self.session.ask(f"{notes}\n\n{base}" if notes else base)
            if self.on_thought:
                self.on_thought(str(thought)[:300])
        except Exception:  # noqa: BLE001 - an idle cycle failing is not an event
            pass
        finally:
            self._cycling = False
            self._last_activity = time.monotonic()

    async def idle_prompt(self) -> str:
        digest = await self.digest() if self.digest else ""
        led = self.journeys.ledger()
        ledger_lines = (
            f"\nToo-hard ledger (goals that FAILED — never re-propose as-is): {', '.join(repr(g[:60]) for g in led['too_hard'][-5:])}."
            if led["too_hard"]
            else ""
        ) + (f"\nCompleted goals: {', '.join(repr(g[:60]) for g in led['completed'][-5:])}." if led["completed"] else "")
        vitals = await self.vitals() if self.vitals else None
        return (
            f"[idle thinker, cycle {self.cycles} — no player prompt for a while] "
            + (
                f"Your status digest (no need to perceive first — act on it):\n{digest}\n"
                if digest
                else "Check on yourself first: get_status. "
            )
            + "Vitals beat everything below: eat if hungry, flee if in danger."
            + ledger_lines
            + f"\nThis cycle's focus — {pick_focus(self.cycles, vitals)} "
            + "If something deserves a longer errand, start a journey for it — one NOVEL task, verifiable from your own state; never a too-hard goal. "
            + "If a player is nearby and something is worth saying, say ONE short line in chat — otherwise stay silent. "
            + "Reply with one sentence about what you observed/did."
        )

    async def supervisor_prompt(self, j: Journey) -> str:
        grew = len(j.journal) - self._seen.get(j.id, 0)
        self._seen[j.id] = len(j.journal)
        recent = "\n".join(f"  {i + 1}. {ln}" for i, ln in enumerate(j.journal[-4:]))
        digest = await self.digest() if self.digest else ""
        verdict, reason = progress_verdict(j.journal)
        return (
            f'[journey supervisor, cycle {self.cycles}] Journey {j.id} is running ("{j.goal}", step {j.iterations}, {grew} step(s) since your last look). Recent journal:\n{recent}\n'
            + (f"Status digest:\n{digest}\n" if digest else "")
            + "The journey OWNS the body right now — you MUST NOT move, dig, place or pathfind in this turn. Your job is judgement, not labor. "
            + "The [Δ …] tail on each journal line is measured truth — trust it over the narration. A [waiting: …] tail means the step DECLARED deliberate patience.\n"
            + f"Measured progress signal: {verdict.upper()} — {reason}. Start from this, not from the prose.\n"
            + "PROGRESSING → do nothing. PATIENT → LEAVE IT ALONE. STALLED → stop_journey and start_journey with a SHARPER goal that routes around the blocker. "
            + "DOOMED → stop_journey with doomed:true. Reply with one sentence: your verdict and what you did."
        )


async def wait_idle(thinker: Thinker) -> None:
    """Test helper: let a pending cycle settle."""
    with contextlib.suppress(asyncio.TimeoutError):
        await asyncio.wait_for(asyncio.sleep(0), 0.1)
