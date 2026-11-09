"""Journeys — long goals that outlive a turn. Port of ``src/journeys.ts``.

One step per model turn through the shared Session, until the model ends with
``[JOURNEY_DONE]``, someone stops it, or it hits its caps. Each step's journal line carries the
Δ critic (what the body and bag MEASURABLY did) and, when declared, a ``[waiting: …]`` tail so
the supervisor can tell chosen patience from a stall (issue #12).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from strands import tool

from .config import cfg
from .doctor import classify_provider_error

DONE = "[JOURNEY_DONE]"
WAIT = re.compile(r"\[WAITING:\s*([^\]]{1,160})\]", re.I)
JOURNAL_MEMORY_KEEP = 60

EndedBy = Literal["goal", "cap", "wall", "stopped", "error", "crash", "process-death", "stale"]
Status = Literal["running", "done", "stopped", "error", "interrupted", "abandoned"]


@dataclass
class Journey:
    id: str
    goal: str
    status: Status
    iterations: int
    started_at: float
    journal: list[str] = field(default_factory=list)
    result: str | None = None
    ended_by: EndedBy | None = None


@dataclass
class BodySnapshot:
    pos: dict[str, float]
    health: float
    food: float
    inv: dict[str, int]

    @staticmethod
    def from_state(state: dict[str, Any]) -> BodySnapshot | None:
        if not state.get("pos"):
            return None
        return BodySnapshot(
            pos=state["pos"],
            health=float(state.get("health") or 0),
            food=float(state.get("food") or 0),
            inv=dict(state.get("inv") or {}),
        )


def failure_owner(err: BaseException) -> tuple[EndedBy, str]:
    """WHOSE fault ended the step — only a world that refuses is evidence about the goal (#29)."""
    kind = classify_provider_error(err).kind
    return ("error" if kind == "unknown" else "crash"), kind


def ledger_verdict(ended_by: EndedBy | None, doomed: bool = False) -> Literal["completed", "too_hard", "none"]:
    if ended_by == "goal":
        return "completed"
    if ended_by == "error":
        return "too_hard"
    if ended_by == "stopped":
        return "too_hard" if doomed else "none"
    return "none"


def step_delta(before: BodySnapshot, after: BodySnapshot) -> str:
    """``Δ +12 cobblestone, -2 hp, moved 34m`` — measured, not narrated."""
    gains: list[str] = []
    losses: list[str] = []
    for n in dict.fromkeys([*before.inv, *after.inv]):  # insertion order, like the TS Set
        d = after.inv.get(n, 0) - before.inv.get(n, 0)
        if d > 0:
            gains.append(f"+{d} {n}")
        elif d < 0:
            losses.append(f"{d} {n}")
    parts = gains[:4] + losses[:3]
    hp = round(after.health - before.health)
    if hp:
        parts.append(f"{'+' if hp > 0 else ''}{hp} hp")
    food = after.food - before.food
    if food:
        parts.append(f"{'+' if food > 0 else ''}{food:g} food")
    moved = sum((after.pos[k] - before.pos[k]) ** 2 for k in ("x", "y", "z")) ** 0.5
    if moved >= 2:
        parts.append(f"moved {round(moved)}m")
    return f"Δ {', '.join(parts)}" if parts else "Δ nothing"


def recent_steps_block(journal: list[str], keep: int = 2, clip: int = 220) -> str:
    lines = [ln for ln in journal if not ln.startswith("error:")][-keep:]
    if not lines:
        return ""
    body = "\n".join(
        f"  {'last' if len(lines) - i == 1 else f'{len(lines) - i} ago'}: {ln[:clip] + '…' if len(ln) > clip else ln}"
        for i, ln in enumerate(lines)
    )
    return (
        f"Your own last step(s), each ending in what the world MEASURED:\n{body}\n"
        "The [Δ …] bracket is measured from your body and inventory, not from your account of it — where the two disagree, the bracket is right and your memory is wrong.\n"
    )


def declared_wait(answer: str) -> str | None:
    """A wait only counts if it names something to wait FOR ('none' is declining to wait)."""
    m = WAIT.search(answer)
    raw = m.group(1).strip() if m else ""
    if not raw:
        return None
    bare = re.sub(r"[.!,;\s]+$", "", re.sub(r"^[\s:'\"-]+", "", raw.lower()))
    if len(bare) < 3 or re.match(r"^(none|nothing|nope|no|not waiting|n/?a|na|null|undefined|-+)\b", bare):
        return None
    return raw


def trailing_waits(journal: list[str]) -> int:
    n = 0
    for line in reversed(journal):
        if "[waiting:" not in line:
            break
        n += 1
    return n


def progress_verdict(journal: list[str], max_waits: int | None = None) -> tuple[str, str]:
    """``(progressing|patient|stalled, reason)`` — computed, not left to the model's reading."""
    max_waits = cfg.journey.max_consecutive_waits if max_waits is None else max_waits
    blanks = 0
    for line in reversed(journal):
        if "Δ nothing" in line or line.startswith("error:"):
            blanks += 1
        else:
            break
    waits = trailing_waits(journal)
    if waits > max_waits:
        return (
            "stalled",
            f"{waits} waiting steps in a row — patience exhausted, the wait's own condition should have ended by now",
        )
    if blanks == 0:
        return "progressing", "the last step changed something measurable"
    undeclared = blanks - waits
    if undeclared >= 3:
        return "stalled", f"{undeclared} consecutive steps with no measured change and no declared wait"
    if waits > 0:
        return (
            "patient",
            f"{waits} consecutive step(s) declared as deliberate waiting — nothing changing is the CORRECT outcome of waiting",
        )
    return "progressing", f"only {blanks} no-change step(s) — too early to call a stall"


def triage_interrupted(
    items: list[Journey],
    now: float,
    stale_ms: float = 45 * 60_000,
    empty_ms: float = 10 * 60_000,
    max_announce: int = 2,
) -> tuple[list[Journey], list[Journey]]:
    retire: list[Journey] = []
    live: list[Journey] = []
    for j in items:
        age = now - j.started_at
        (retire if age > stale_ms or (j.iterations == 0 and age > empty_ms) else live).append(j)
    live.sort(key=lambda j: -j.started_at)
    return live[:max_announce], retire + live[max_announce:]


class Asker(Protocol):
    """What a journey needs from the session: ``ask`` and ``busy``."""

    @property
    def busy(self) -> int: ...

    async def ask(self, text: str) -> str: ...


class JourneyRunner:
    def __init__(self, store: Path | None = None) -> None:
        self.store = store or Path(os.environ.get("MEMORY_DIR") or Path.home() / ".strands-minecraft") / "journeys.py.json"
        self.session: Asker | None = None
        self.journeys: dict[str, Journey] = {}
        self._stop: set[str] = set()
        self._doomed: set[str] = set()
        self.completed: list[str] = []
        self.too_hard: list[str] = []
        self.on_progress: Callable[[Journey, str], None] | None = None
        self.snapshot: Callable[[], Awaitable[BodySnapshot | None]] | None = None
        self.digest: Callable[[], Awaitable[str]] | None = None
        self.take_notes: Callable[[], str] | None = None
        self._tasks: set[asyncio.Task[None]] = set()
        self.cooldown_ms = cfg.journey.cooldown_ms
        self.wait_cooldown_ms = cfg.journey.wait_cooldown_ms
        self._load()

    def _load(self) -> None:
        try:
            data = json.loads(self.store.read_text())
        except (OSError, ValueError):
            return
        self.completed = list(data.get("completed", []))
        self.too_hard = list(data.get("too_hard", []))
        for raw in data.get("journeys", []):
            j = Journey(**raw)
            if j.status == "running":
                j.status, j.ended_by = "interrupted", "process-death"
                j.result = j.result or f"Process died mid-journey after {j.iterations} step(s)."
            self.journeys[j.id] = j

    def _persist(self) -> None:
        finished = sorted((j for j in self.journeys.values() if j.status != "running"), key=lambda j: -j.started_at)
        for j in finished[cfg.journey.keep :]:
            self.journeys.pop(j.id, None)
        for j in self.journeys.values():
            j.journal = j.journal[-JOURNAL_MEMORY_KEEP:]
        try:
            self.store.parent.mkdir(parents=True, exist_ok=True)
            rows = [
                dict(asdict(j), journal=j.journal[-20:])
                for j in sorted(self.journeys.values(), key=lambda j: -j.started_at)[: cfg.journey.keep]
            ]
            tmp = self.store.with_suffix(".tmp")
            tmp.write_text(json.dumps({"journeys": rows, "completed": self.completed, "too_hard": self.too_hard}, indent=2))
            tmp.replace(self.store)
        except OSError:
            pass  # a full disk must not kill the journey itself

    @property
    def interrupted(self) -> list[Journey]:
        announce, retire = triage_interrupted([j for j in self.journeys.values() if j.status == "interrupted"], time.time() * 1000)
        for j in retire:
            j.status, j.ended_by = "abandoned", "stale"
        if retire:
            self._persist()
        return announce

    def ledger(self) -> dict[str, list[str]]:
        return {"completed": list(self.completed), "too_hard": list(self.too_hard)}

    @property
    def running(self) -> Journey | None:
        return next((j for j in self.journeys.values() if j.status == "running"), None)

    def bind(self, session: Asker) -> None:
        self.session = session

    def stop(self, jid: str, doomed: bool = False) -> bool:
        j = self.journeys.get(jid)
        if not j or j.status != "running":
            return False
        self._stop.add(jid)
        if doomed:
            self._doomed.add(jid)
        return True

    def start(self, goal: str) -> Journey:
        if self.session is None:
            raise RuntimeError("Journey runner not bound to a session yet.")
        if self.running:
            raise RuntimeError(f'Already on a journey ({self.running.id}: "{self.running.goal}") — stop it first or wait.')
        jid = f"j{int(time.time() * 1000):x}"
        j = Journey(id=jid, goal=goal, status="running", iterations=0, started_at=time.time() * 1000)
        self.journeys[jid] = j
        self._persist()
        task = asyncio.get_running_loop().create_task(self.run(j))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return j

    def _ledgerize(self, j: Journey) -> None:
        v = ledger_verdict(j.ended_by, j.id in self._doomed)
        if v == "completed":
            self.completed.append(j.goal)
            self.too_hard = [g for g in self.too_hard if g != j.goal]
        elif v == "too_hard":
            self.too_hard.append(j.goal)
        self._doomed.discard(j.id)
        self.completed, self.too_hard = self.completed[-20:], self.too_hard[-20:]

    def _prompt(self, j: Journey, news: str, digest: str) -> str:
        waits = trailing_waits(j.journal)
        return (
            (
                "News from your own senses since your last step — each line carries when it was true; "
                f"if it changes what the next step should be, ACT on it and say so, otherwise carry on:\n{news}\n\n"
                if news
                else ""
            )
            + f'[journey {j.id}, step {j.iterations}/{cfg.journey.max_iterations}] Long-term goal: "{j.goal}".'
            + (f"\nYour status right now:\n{digest}\n" if digest else " ")
            + recent_steps_block(j.journal)
            + "Take the NEXT concrete step toward it now. Report in 1-2 sentences what you did. "
            + "If this step is deliberate WAITING on something outside your control (a furnace smelting, crops growing, "
            + "a player fetching something), say so with a marker: [WAITING: what you await and what ends the wait]. "
            + "Never use it to pad a step you don't know how to take. "
            + (
                f"You have waited {waits} step(s) in a row; past {cfg.journey.max_consecutive_waits} consecutive waits the supervisor reads this journey as STALLED. "
                if waits
                else ""
            )
            + f"If the goal is fully achieved, end your reply with {DONE}. If it is impossible, explain why and end with {DONE}."
        )

    async def run(self, j: Journey) -> None:
        assert self.session is not None
        while j.status == "running":
            if j.id in self._stop:
                self._stop.discard(j.id)
                j.status, j.ended_by = "stopped", "stopped"
                break
            if j.iterations >= cfg.journey.max_iterations:
                j.status, j.ended_by, j.result = "done", "cap", f"Hit the {cfg.journey.max_iterations}-iteration cap."
                break
            if time.time() * 1000 - j.started_at > cfg.journey.max_wall_ms:
                j.status, j.ended_by, j.result = "done", "wall", "Hit the wall-clock cap."
                break
            yield_start = time.monotonic()
            while self.session.busy > 0 and (time.monotonic() - yield_start) * 1000 < cfg.journey.yield_max_ms and j.id not in self._stop:
                await asyncio.sleep(0.5)
            if j.id in self._stop:
                continue
            j.iterations += 1
            before = await self.snapshot() if self.snapshot else None
            try:
                digest = await self.digest() if self.digest else ""
                news = (self.take_notes() if self.take_notes else "").strip()
                answer = await self.session.ask(self._prompt(j, news, digest))
                after = await self.snapshot() if self.snapshot else None
                delta = f" [{step_delta(before, after)}]" if before and after else ""
                waited = declared_wait(answer)
                waits_before = trailing_waits(j.journal)
                tail = (
                    f" [waiting: {waited[:120]}{' ⚠ patience exhausted' if waits_before + 1 > cfg.journey.max_consecutive_waits else ''}]"
                    if waited
                    else ""
                )
                line = (WAIT.sub("", answer.replace(DONE, "")).strip()[:300] + delta + tail).strip()
                j.journal.append(line)
                self._persist()
                if self.on_progress:
                    self.on_progress(j, line)
                if DONE in answer:
                    j.status, j.ended_by, j.result = "done", "goal", line
                    break
            except Exception as err:  # noqa: BLE001 - one bad step is survivable, two are not
                j.journal.append(f"error: {err}")
                if len(j.journal) >= 2 and all(ln.startswith("error:") for ln in j.journal[-2:]):
                    ended_by, kind = failure_owner(err)
                    j.status, j.ended_by = "error", ended_by
                    j.result = (
                        f"Stopped by a fault in OUR stack ({kind}), not by the world: {err}. The goal is untested — retry it."
                        if ended_by == "crash"
                        else str(err)
                    )
                    break
            waiting = "[waiting:" in (j.journal[-1] if j.journal else "")
            await asyncio.sleep((self.wait_cooldown_ms if waiting else self.cooldown_ms) / 1000)
        j.result = j.result or (j.journal[-1] if j.journal else "(no steps)")
        self._ledgerize(j)
        self._persist()
        if self.on_progress:
            self.on_progress(j, f"journey {j.status}: {j.result}")


def journey_tools(runner: JourneyRunner) -> list[Any]:
    @tool
    def start_journey(goal: str) -> str:
        """Begin a long-term background goal that outlives this conversation turn — you will automatically take a step toward it every few seconds until it is done ("collect 64 iron", "build a cobblestone tower"). Use for anything that takes many steps over minutes. Returns the journey id immediately.

        Args:
            goal: The long-term goal, concrete and checkable.
        """
        j = runner.start(goal)
        return f'Journey {j.id} started: "{goal}". Progress will be reported as steps complete.'

    @tool
    def journey_status(id: str | None = None) -> dict[str, Any]:  # noqa: A002 - the TS tool's name
        """Check your journeys — current one and past ones, with their step-by-step journal.

        Args:
            id: A journey id; omit for all.
        """
        items = [runner.journeys[id]] if id and id in runner.journeys else list(runner.journeys.values())
        led = runner.ledger()
        if not items:
            return {"journeys": "none yet", **led}
        return {
            "journeys": [
                {
                    "id": j.id,
                    "goal": j.goal,
                    "status": j.status,
                    "iterations": j.iterations,
                    "recentSteps": j.journal[-5:],
                    "result": j.result,
                }
                for j in items
            ],
            "completedGoals": led["completed"][-5:],
            "tooHardGoals": led["too_hard"][-5:],
        }

    @tool
    def stop_journey(id: str, doomed: bool = False) -> str:  # noqa: A002
        """Stop the running journey after its current step lands. Mark doomed:true when the goal is impossible/misconceived — doomed goals land in the too-hard ledger.

        Args:
            id: The journey id.
            doomed: true = impossible as stated, remember never to retry it as-is.
        """
        return (
            f"Journey {id} will stop after the current step{' (ledgered as too hard)' if doomed else ''}."
            if runner.stop(id, doomed)
            else f"No running journey {id}."
        )

    return [start_journey, journey_status, stop_journey]
