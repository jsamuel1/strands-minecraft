"""Journey sentinels, the Δ critic and the ledgers — journeys.ts (#12, #29)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from strands.types.exceptions import ModelThrottledException

from mcbrain.journeys import (
    DONE,
    BodySnapshot,
    JourneyRunner,
    declared_wait,
    failure_owner,
    ledger_verdict,
    progress_verdict,
    step_delta,
)


def snap(**inv: int) -> BodySnapshot:
    return BodySnapshot(pos={"x": 0, "y": 64, "z": 0}, health=20, food=20, inv=inv)


def test_step_delta_measures_what_changed() -> None:
    after = BodySnapshot(pos={"x": 30, "y": 64, "z": 16}, health=18, food=19, inv={"cobblestone": 12, "torch": 1})
    assert step_delta(snap(torch=3), after) == "Δ +12 cobblestone, -2 torch, -2 hp, -1 food, moved 34m"
    assert step_delta(snap(), snap()) == "Δ nothing"


def test_a_sham_wait_is_not_a_wait() -> None:
    assert declared_wait("smelting [WAITING: 8 iron in the furnace, ~90s]") == "8 iron in the furnace, ~90s"
    assert declared_wait("mining on [waiting: none, continuing next vein]") is None
    assert declared_wait("no marker") is None


def test_progress_verdict_separates_patience_from_stall() -> None:
    assert progress_verdict(["a [Δ +1 log]"])[0] == "progressing"
    assert progress_verdict(["x [Δ nothing] [waiting: furnace]"] * 2)[0] == "patient"
    assert progress_verdict(["x [Δ nothing]"] * 3)[0] == "stalled"
    assert progress_verdict(["x [Δ nothing] [waiting: furnace]"] * 7, max_waits=6)[0] == "stalled"


def test_ledger_blames_the_right_party() -> None:
    assert failure_owner(ModelThrottledException("x"))[0] == "crash", "a throttle says nothing about the goal (#29)"
    assert failure_owner(RuntimeError("the world said no"))[0] == "error"
    assert ledger_verdict("goal") == "completed"
    assert ledger_verdict("crash") == "none"
    assert ledger_verdict("stopped", doomed=True) == "too_hard"


class FakeSession:
    def __init__(self, answers: list[str | BaseException]) -> None:
        self.answers = answers
        self.prompts: list[str] = []
        self.busy = 0

    async def ask(self, text: str) -> str:
        self.prompts.append(text)
        a = self.answers.pop(0)
        if isinstance(a, BaseException):
            raise a
        return a


@pytest.fixture
def runner(tmp_path: Path) -> JourneyRunner:
    r = JourneyRunner(store=tmp_path / "journeys.json")
    r.cooldown_ms = r.wait_cooldown_ms = 1
    return r


async def test_journey_runs_until_done_sentinel_with_delta_and_news(runner: JourneyRunner) -> None:
    session = FakeSession(["chopped a tree", f"crafted the pickaxe {DONE}"])
    inv = iter([snap(), snap(oak_log=5), snap(oak_log=5), snap(wooden_pickaxe=1)])

    async def snapshot() -> BodySnapshot:
        return next(inv)

    runner.bind(session)
    runner.snapshot = snapshot
    notes = iter(["(system) You DIED and respawned.", ""])
    runner.take_notes = lambda: next(notes)
    j = runner.start("get a wooden pickaxe")
    while j.status == "running":
        await asyncio.sleep(0.01)
    assert j.status == "done" and j.ended_by == "goal"
    assert j.journal[0] == "chopped a tree [Δ +5 oak_log]"
    assert DONE not in j.journal[1]
    assert session.prompts[0].startswith("News from your own senses")
    assert "last: chopped a tree [Δ +5 oak_log]" in session.prompts[1], "the Δ critic is replayed to the model"
    assert runner.ledger()["completed"] == ["get a wooden pickaxe"]


async def test_two_failed_steps_end_it_and_our_fault_is_not_ledgered(runner: JourneyRunner) -> None:
    runner.bind(FakeSession([ModelThrottledException("a"), ModelThrottledException("b")]))
    j = runner.start("mine iron")
    while j.status == "running":
        await asyncio.sleep(0.01)
    assert j.status == "error" and j.ended_by == "crash"
    assert "OUR stack" in (j.result or "")
    assert runner.ledger()["too_hard"] == []


async def test_journeys_persist_and_come_back_interrupted(tmp_path: Path) -> None:
    store = tmp_path / "j.json"
    r = JourneyRunner(store=store)
    gate = asyncio.Event()

    class Slow(FakeSession):
        async def ask(self, text: str) -> str:
            await gate.wait()
            return "x"

    r.bind(Slow([]))
    r.start("build a tower")
    await asyncio.sleep(0.02)
    again = JourneyRunner(store=store)
    assert [j.goal for j in again.interrupted] == ["build a tower"]
    gate.set()


async def test_only_one_journey_at_a_time(runner: JourneyRunner) -> None:
    gate = asyncio.Event()

    class Slow(FakeSession):
        async def ask(self, text: str) -> str:
            await gate.wait()
            return DONE

    runner.bind(Slow([]))
    runner.start("a")
    with pytest.raises(RuntimeError, match="Already on a journey"):
        runner.start("b")
    gate.set()
