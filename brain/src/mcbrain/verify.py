"""Did the action actually happen? A post-condition per macro-action, read from the WORLD.

A tool returning without an exception only means the request was sent and answered. The body's
tools report partial or no effect as ordinary text ("placed 0/25 … FAILED 24", "COULD NOT REACH",
"gave up after 20s"), and v1/v2 labelled all of that ``ok``. Jev's night episodes then looped
130× on a shelter that placed nothing while ``recent_actions`` told it ``build_shelter: ok``.

So the verdict is computed from world state before and after the step, never from the tool's
text or transport status:

- ``done``: the post-condition held (the bag gained the item, the table is in reach, …).
- ``in_progress``: a multi-step effect moved forward but is not complete (a shelter with some
  positions still open); the check says how far and, when it can be timed, how long to finish.
- ``no_effect``: the world shows no change toward the post-condition, or a tool call raised.
- ``unverified``: this action has no checkable post-condition (none today; kept for honesty).
- ``waited``: ``wait`` did nothing, by design.

``reason`` is WHY, in the tool's own words when it gave one (the first refusal line, e.g. "is
occupied by stone"), otherwise what the world check found. ``check`` is WHAT was measured. The
verdict is information for the model (``recent_actions``) and for the table. It never removes an
option: the model decides what to do with it.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Collection
from dataclasses import dataclass
from typing import Any

from .sidecar import Body

VERDICTS = ("done", "in_progress", "no_effect", "unverified", "waited")

#: Food items that count as a hunt's success (raw or cooked drops of FOOD_MOBS).
FOOD_ITEMS = ("beef", "porkchop", "mutton", "chicken", "rabbit", "cooked_beef", "cooked_porkchop", "cooked_mutton", "cooked_chicken")


@dataclass(frozen=True)
class Verdict:
    verdict: str
    check: str
    reason: str = ""


@dataclass(frozen=True)
class Snapshot:
    """The world facts a post-condition needs, taken before and after the action."""

    inv: dict[str, int]
    pos: dict[str, float]
    food: float
    tables_near: int
    hostiles: dict[str, float]  # name -> nearest distance
    #: Plan positions that are solid (any full block): the shelter's world truth.
    solid: frozenset[tuple[int, int, int]] = frozenset()
    t: float = 0.0

    @staticmethod
    def take(body: Body, hostile: tuple[str, ...], plan: list[dict[str, int]] | None = None, t: float = 0.0) -> Snapshot:
        s = body.state()
        tables = body.call("find_blocks", name="crafting_table", maxDistance=4, count=4)
        around = body.call("look_around", radius=24)
        ents = around.data.get("entities", []) if isinstance(around.data, dict) else []
        near: dict[str, float] = {}
        for e in ents:
            if isinstance(e, dict) and e.get("name") in hostile:
                d = float(e.get("distance", 99))
                near[str(e["name"])] = min(d, near.get(str(e["name"]), 99.0))
        solid = frozenset((b["x"], b["y"], b["z"]) for b in body.blocks(plan) if b.get("solid")) if plan else frozenset()
        return Snapshot(
            inv=dict(s.get("inv") or {}),
            pos=dict(s.get("pos") or {"x": 0, "y": 0, "z": 0}),
            food=float(s.get("food") or 0),
            tables_near=len(tables.data) if isinstance(tables.data, list) else 0,
            hostiles=near,
            solid=solid,
            t=t,
        )


#: The first thing a tool says went wrong, e.g. "(20, 57, 92) is occupied by stone.", "COULD NOT REACH …".
_REFUSAL = re.compile(
    r"(is occupied by [\w ]+|COULD NOT REACH[^.|]*|gave up[^.|]*|out of time[^.|]*|Missing materials[^|]*|No '[\w' /]+' found[^.|]*"
    r"|Cannot [^.|]*|can't [^.|]*|not enough [^.|]*|was NOT placed[^.|]*|No solid neighbor[^.|]*)",
    re.IGNORECASE,
)


def tool_reason(outcome: str) -> str:
    """The tool's own first refusal, coordinates stripped, or '' when it gave none."""
    m = _REFUSAL.search(outcome or "")
    if not m:
        return ""
    text = re.sub(r"\s*\(-?\d+(?:\.\d+)?, -?\d+(?:\.\d+)?, -?\d+(?:\.\d+)?\)", "", m.group(1))
    return re.sub(r"\s+([:,;])", r"\1", re.sub(r"\s+", " ", text)).strip(" :")[:160]


def _gained(a: Snapshot, b: Snapshot, pred: Callable[[str], bool] = lambda _n: True) -> dict[str, int]:
    return {n: c - a.inv.get(n, 0) for n, c in b.inv.items() if c > a.inv.get(n, 0) and pred(n)}


def _moved(a: Snapshot, b: Snapshot) -> float:
    return math.dist((a.pos["x"], a.pos["z"]), (b.pos["x"], b.pos["z"]))


def _fmt(g: dict[str, int]) -> str:
    return ", ".join(f"+{c} {n}" for n, c in g.items()) or "nothing"


def _yes(ok: bool) -> str:
    return "done" if ok else "no_effect"


def shelter(meta: dict[str, Any], a: Snapshot, b: Snapshot, outcome: str, s_per_block: float | None) -> Verdict:
    """A shelter is its plan's positions being SOLID — whatever block fills them (stone seals too)."""
    plan = [tuple(p) for p in meta.get("plan", [])]
    total = len(plan)
    before, after = sum(p in a.solid for p in plan), sum(p in b.solid for p in plan)
    open_ = total - after
    check = f"{after}/{total} shelter positions solid (was {before})"
    if total and open_ == 0:
        return Verdict("done", check + " — sealed")
    eta = f", ~{open_ * s_per_block:.0f}s to finish at {s_per_block:.1f}s/block" if s_per_block else ""
    if after > before:
        return Verdict("in_progress", f"{check}, {open_} open{eta}", tool_reason(outcome))
    return Verdict("no_effect", f"{check}, {open_} open{eta}", tool_reason(outcome) or "no plan position became solid")


def check(action: str, meta: dict[str, Any], a: Snapshot, b: Snapshot, outcome: str = "", s_per_block: float | None = None) -> Verdict:
    """The verdict for one executed macro-action. Pure: facts in, verdict out."""
    why = tool_reason(outcome)
    if action == "wait":
        return Verdict("waited", "wait does nothing by design")
    if action == "build_shelter":
        return shelter(meta, a, b, outcome, s_per_block)
    if action.startswith("craft_"):
        item = action.removeprefix("craft_")
        n = b.inv.get(item, 0) - a.inv.get(item, 0)
        return Verdict(_yes(n > 0), f"bag {item} {a.inv.get(item, 0)}→{b.inv.get(item, 0)}", "" if n > 0 else why or f"no {item} appeared")
    if action == "place_crafting_table":
        ok = b.tables_near > a.tables_near
        return Verdict(
            _yes(ok), f"crafting tables within 4 blocks {a.tables_near}→{b.tables_near}", "" if ok else why or "no new table in reach"
        )
    if action == "walk_to_table":
        ok = b.tables_near > 0
        return Verdict(_yes(ok), f"crafting tables within 4 blocks after: {b.tables_near}", "" if ok else why or "still out of reach")
    if action.startswith("explore_"):
        d, need = _moved(a, b), float(meta.get("min_move", 1.0))
        return Verdict(_yes(d >= need), f"moved {d:.1f} blocks, need ≥{need:.0f}", "" if d >= need else why or "barely moved")
    if action == "flee_hostile":
        mob = str(meta.get("mob"))
        da, db = a.hostiles.get(mob, 99.0), b.hostiles.get(mob, 99.0)
        return Verdict(_yes(db > da), f"{mob} distance {da:.1f}→{db:.1f}", "" if db > da else why or f"the {mob} is no farther")
    if action == "fight_hostile":
        mob = str(meta.get("mob"))
        fa, fb = a.hostiles.get(mob), b.hostiles.get(mob)
        gone = fb is None or (fa is not None and fb > fa + 4)
        return Verdict(_yes(gone), f"{mob} nearest {fa}→{fb} (gone = done)", "" if gone else why or f"the {mob} is still there")
    if action == "eat":
        ok = b.food > a.food
        return Verdict(_yes(ok), f"food {a.food:.0f}→{b.food:.0f}", "" if ok else why or "food did not rise")
    if action == "hunt_animal":
        g = _gained(a, b, lambda n: n in FOOD_ITEMS)
        return Verdict(_yes(bool(g)), f"food items gained: {_fmt(g)}", "" if g else why or "no food dropped into the bag")
    if action.startswith(("fell_tree", "mine_", "dig_down", "collect_drops")):
        want: Collection[str] | None = meta.get("gain")
        g = _gained(a, b, (lambda n: n in want) if want else (lambda _n: True))
        counting = f" (counting {', '.join(want)})" if want else ""
        return Verdict(_yes(bool(g)), f"gained {_fmt(g)}{counting}", "" if g else why or "nothing reached the bag")
    return Verdict("unverified", "no post-condition defined for this action")
