"""Benchmark scoring: goals are verified from body state; the table reads only the JSONL."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

from mcbench.cli import ARMS, spec_for
from mcbench.pricing import token_usd
from mcbench.runner import Episode, _pct, append_jsonl
from mcbench.summary import load, table
from mcbench.tasks import BY_ID


def st(**inv: int) -> dict:
    return {"inv": inv, "counters": {"deaths": 0}}


def test_tasks_are_checked_from_inventory_not_narration() -> None:
    assert BY_ID["logs16"].check(st(oak_log=10, birch_log=6)) == (True, 1.0)
    assert BY_ID["logs16"].check(st(oak_log=8)) == (False, 0.5)
    assert BY_ID["wood_tools"].check(st(crafting_table=1, wooden_pickaxe=1))[0]
    assert not BY_ID["wood_tools"].check(st(crafting_table=1))[0]
    placed = {**st(wooden_pickaxe=1), "_placedTable": True}
    assert BY_ID["wood_tools"].check(placed)[0], "a PLACED table counts; v1 demanded a second one in the bag"
    assert BY_ID["cobble8"].check(st(cobblestone=5, cobbled_deepslate=3))[0]
    assert BY_ID["iron1"].check(st(raw_iron=1))[0]


def test_night_needs_a_real_dawn_and_no_deaths() -> None:
    night = BY_ID["night"]
    assert not night.check({"inv": {}, "counters": {"deaths": 0}, "timeOfDay": 18000, "_passedNight": False})[0]
    assert night.check({"inv": {}, "counters": {"deaths": 0}, "timeOfDay": 300, "_passedNight": True})[0]
    assert not night.check({"inv": {}, "counters": {"deaths": 1}, "timeOfDay": 300, "_passedNight": True})[0]


def test_unknown_price_is_nan_never_zero() -> None:
    assert math.isnan(token_usd("kev-latest", 1000, 0))
    assert token_usd("jev-latest", 1_000_000, 0) == 0.042
    assert token_usd("global.anthropic.claude-sonnet-5", 1_000_000, 100_000) == 3.0


def test_every_arm_resolves_and_decision_arms_are_policies() -> None:
    for arm, (kind, _) in ARMS.items():
        k, spec = spec_for(arm)
        assert k == kind
        if spec.is_decision:
            assert kind == "policy"
    assert spec_for("clm")[1].label.startswith("clm-v0.1-8b@sagemaker")


def test_percentiles() -> None:
    assert _pct([], 50) is None
    assert _pct([5.0], 95) == 5.0
    assert _pct([float(i) for i in range(1, 101)], 95) == 95.0


def test_summary_counts_infra_failures_separately(tmp_path: Path) -> None:
    f = tmp_path / "e.jsonl"
    good = Episode(
        "r",
        "jev",
        "policy",
        "typesafe:jev-latest",
        "logs16",
        1,
        "1",
        "1.21.11",
        "t",
        success=True,
        progress=1.0,
        time_to_goal_s=30.0,
        steps=3,
        tokens_in=900,
        cost_usd=0.00004,
        latency_p50_ms=300,
        latency_p95_ms=400,
        answering_models=["jev-1.13.0"],
    )
    bad = Episode("r", "jev", "policy", "typesafe:jev-latest", "logs16", 2, "1", "1.21.11", "t", error="RuntimeError: never healthy")
    append_jsonl(f, good)
    append_jsonl(f, bad)
    assert json.loads(f.read_text().splitlines()[0]) == asdict(good)
    t = table(load(f))
    assert "| jev | jev-1.13.0 | logs16 | 1 | 1/1 |" in t
    assert t.rstrip().endswith("| 1 |"), "one infra failure, not counted as a model failure"


def test_an_uncapped_task_shows_the_model_a_null_step_cap() -> None:
    from mcbench.tasks import BY_ID, UNCAPPED_STEPS

    assert BY_ID["night"].max_steps == UNCAPPED_STEPS, "night is wall-clock bound"
    assert all(t.max_steps < UNCAPPED_STEPS for t in BY_ID.values() if t.id != "night")
