"""The fixed task suite. Every goal is VERIFIED FROM BODY STATE by code, never from the model's say-so.

Each task names its budget in decisions (policy arms) / turns (LLM arms) and in wall-clock
seconds, and a ``check(state) -> (done, progress)`` over the sidecar's ``/state``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

State = dict[str, Any]


def _inv(s: State) -> dict[str, int]:
    return dict(s.get("inv") or {})


def _sum(s: State, pred: Callable[[str], bool]) -> int:
    return sum(c for n, c in _inv(s).items() if pred(n))


def logs(s: State) -> int:
    return _sum(s, lambda n: n.endswith("_log"))


def cobble(s: State) -> int:
    return _sum(s, lambda n: n in ("cobblestone", "cobbled_deepslate"))


def iron(s: State) -> int:
    return _sum(s, lambda n: n in ("raw_iron", "iron_ore", "deepslate_iron_ore", "iron_ingot"))


@dataclass(frozen=True)
class Task:
    id: str
    goal: str
    check: Callable[[State], tuple[bool, float]]
    max_steps: int
    max_wall_s: int
    #: RCON commands that shape the world before the bot acts (e.g. set night). Never gives items.
    setup: tuple[str, ...] = ()


#: A step cap this large means "no step limit": the wall-clock cap binds. Shown to the model as null.
UNCAPPED_STEPS = 10_000


def _ratio(n: float, of: float) -> tuple[bool, float]:
    return n >= of, round(min(1.0, n / of), 3)


def _wood_tools(s: State) -> tuple[bool, float]:
    inv = _inv(s)
    # A PLACED table counts: crafting the pickaxe needs the table in the world, not in the bag.
    # v1 of this check read the bag only, so success waited for a pointless second table
    # (2026-09-26 run; those rows are kept aside as episodes.wood_tools-v1.jsonl).
    table = inv.get("crafting_table", 0) > 0 or bool(s.get("_placedTable"))
    pick = inv.get("wooden_pickaxe", 0) > 0 or inv.get("stone_pickaxe", 0) > 0
    return table and pick, round((0.5 if table else min(0.4, logs(s) / 10)) + (0.5 if pick else 0), 3)


def _survive(s: State) -> tuple[bool, float]:
    deaths = int((s.get("counters") or {}).get("deaths", 0))
    t = s.get("timeOfDay") or 0
    # survived when it is morning again (MC day ticks 0..~1000 after 13000+ night) with no deaths
    done = deaths == 0 and bool(s.get("_passedNight"))
    return done, 1.0 if done else (0.0 if deaths else round(min(1.0, max(0, t - 13000) / 10000), 3))


SUITE: tuple[Task, ...] = (
    Task("logs16", "Collect 16 logs (any wood type).", lambda s: _ratio(logs(s), 16), max_steps=20, max_wall_s=420),
    Task("wood_tools", "Craft a crafting table and a wooden pickaxe.", _wood_tools, max_steps=24, max_wall_s=480),
    Task("cobble8", "Get a pickaxe and mine 8 cobblestone.", lambda s: _ratio(cobble(s), 8), max_steps=32, max_wall_s=600),
    Task("iron1", "Obtain at least one iron ore (needs a stone pickaxe).", lambda s: _ratio(iron(s), 1), max_steps=48, max_wall_s=900),
    Task(
        "night",
        "Survive the night without dying.",
        _survive,
        # Wall-clock binds, never steps: night is 11,500 ticks (575 s) from 12500. v1's 40 steps
        # ran out at ~70% of the night for every waiting policy; v2's 150 still bound Jev twice
        # (a 2 s/step shelter loop spent 150 steps by ~330 s, alive, before dawn). v3: no step cap.
        max_steps=UNCAPPED_STEPS,
        max_wall_s=660,
        setup=("time set 12500", "difficulty easy"),
    ),
)

BY_ID = {t.id: t for t in SUITE}
