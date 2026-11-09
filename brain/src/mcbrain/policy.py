"""The System One brain: a decision model picks the next macro-action; code executes it.

A System One model (Jev, Kev, CLM) returns typed judgments, not tool calls. So the harness
splits the work the way the TypeSafe guidance says to:

- **Facts are code.** ``observe()`` reads the body (state + a few perception tools) and
  ``candidates()`` builds only the macro-actions that are LEGAL and GROUNDED right now:
  "fell the oak_log tree at (12,70,-3)" exists only when find_blocks saw one; "craft a
  wooden_pickaxe" only when the bag can pay for it (or a table is in reach for 3×3 recipes).
  Illegal options never dilute the Choice's distribution.
- **Judgment is the model.** One ``Choice`` over those candidate ids (plus ``wait``) asks
  which action best advances the goal from this state. Each option carries a description.
- **Execution is code.** A macro-action is one to three sidecar tool calls (the same 54 tools
  the LLM brain calls, the same composite ones the turn-economy doctrine recommends).

Every decision is recorded (state size, options, answer distribution, confidence, model id,
latency, usage) so the benchmark can compare the System One models to an LLM *on the same
alphabet* (``LLMDecisionModel``), separating "which model" from "picker vs free tool calling".
"""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from strands.experimental.decisions import Choice, DecisionModel, YesNo

from .config import cfg
from .sidecar import Body, ToolOutcome
from .verify import Snapshot, check

ToolStep = tuple[str, dict[str, Any]]

LOGS = ("oak_log", "birch_log", "spruce_log", "jungle_log", "acacia_log", "dark_oak_log", "mangrove_log", "cherry_log")
STONE = ("stone", "cobblestone", "deepslate", "andesite", "diorite", "granite")
COAL = ("coal_ore", "deepslate_coal_ore")
IRON = ("iron_ore", "deepslate_iron_ore")
FOOD_MOBS = ("cow", "pig", "sheep", "chicken", "rabbit")
HOSTILE = (
    "zombie",
    "skeleton",
    "spider",
    "creeper",
    "husk",
    "drowned",
    "stray",
    "witch",
    "cave_spider",
    "slime",
    "phantom",
    "enderman",
)
PICKS = ("netherite_pickaxe", "diamond_pickaxe", "iron_pickaxe", "stone_pickaxe", "wooden_pickaxe")
SWORDS = ("netherite_sword", "diamond_sword", "iron_sword", "stone_sword", "wooden_sword")
AXES = ("netherite_axe", "diamond_axe", "iron_axe", "stone_axe", "wooden_axe")
PLANKS_SUFFIX = "_planks"


@dataclass
class Observation:
    state: dict[str, Any]
    trees: list[dict[str, Any]] = field(default_factory=list)
    stone: list[dict[str, Any]] = field(default_factory=list)
    coal: list[dict[str, Any]] = field(default_factory=list)
    iron: list[dict[str, Any]] = field(default_factory=list)
    tables: list[dict[str, Any]] = field(default_factory=list)
    furnaces: list[dict[str, Any]] = field(default_factory=list)
    entities: list[dict[str, Any]] = field(default_factory=list)
    drops: int = 0
    obs_ms: float = 0.0
    #: The block under the body (look_around.standingOn), e.g. stone, water, oak_leaves.
    standing_on: str = ""
    #: Body events (sentinel notes, reflex actions) since the previous observation.
    events: list[str] = field(default_factory=list)

    @property
    def inv(self) -> dict[str, int]:
        return dict(self.state.get("inv") or {})

    @property
    def pos(self) -> dict[str, float]:
        return self.state.get("pos") or {"x": 0, "y": 0, "z": 0}


def _clusters(out: ToolOutcome) -> list[dict[str, Any]]:
    return [c for c in out.data if isinstance(c, dict) and "position" in c] if isinstance(out.data, list) else []


def observe(body: Body, profile: ContextProfile | None = None) -> Observation:
    """Facts, from code. Four find_blocks + one look_around; each is cheap and read-only."""
    t0 = time.perf_counter()
    state = body.state()
    r = cfg.policy.search_radius
    ob = Observation(state=state)
    ob.trees = _clusters(body.call("find_blocks", name=",".join(LOGS), maxDistance=r, count=3))
    ob.stone = _clusters(body.call("find_blocks", name=",".join(STONE), maxDistance=min(r, 24), count=3))
    ores = _clusters(body.call("find_blocks", name=",".join(COAL + IRON), maxDistance=min(r, 32), count=6))
    ob.coal = [c for c in ores if c.get("block") in COAL][:2]
    ob.iron = [c for c in ores if c.get("block") in IRON][:2]
    stations = _clusters(body.call("find_blocks", name="crafting_table,furnace", maxDistance=24, count=4))
    ob.tables = [c for c in stations if c.get("block") == "crafting_table"]
    ob.furnaces = [c for c in stations if c.get("block") == "furnace"]
    around = body.call("look_around", radius=16)
    ents = around.data.get("entities", []) if isinstance(around.data, dict) else []
    ob.entities = [e for e in ents if isinstance(e, dict)]
    ob.standing_on = str(around.data.get("standingOn", "")) if isinstance(around.data, dict) else ""
    ob.events = body_events(body, *((profile.event_chars, profile.max_events) if profile else (None, None)))
    # Only drops collect_ground_items can reach (radius 8): v1 counted look_around's 16 and
    # offered pickups that could only fail (CLM iron1, 2026-09-26: collect_drops! loops).
    ob.drops = sum(1 for e in ob.entities if e.get("name") == "item" and float(e.get("distance", 99)) <= 8)
    ob.obs_ms = (time.perf_counter() - t0) * 1000
    return ob


#: Event kinds the body raises about ITSELF (sentinel notes, reflexes, reconnects, death). Chat and
#: peer lines are excluded: the benchmark has no players, and peers are never answered anyway.
BODY_EVENT_KINDS = ("note", "reflex", "revive", "death", "gave_up", "system")
_COORDS = re.compile(r"\s*\(-?\d+(?:\.\d+)?, -?\d+(?:\.\d+)?, -?\d+(?:\.\d+)?\)")


_ADVANCEMENT = re.compile(r"has made the advancement \[([^\]]+)\]")


def advancement(text: str) -> str | None:
    """The advancement name in a server message, e.g. 'Stone Age' — Minecraft's own progress marker."""
    m = _ADVANCEMENT.search(text)
    return m.group(1) if m else None


@dataclass(frozen=True)
class ContextProfile:
    """How much decision state one model is shown: its operating point, recorded per episode.

    CLM reads 2,048 tokens per row through a contrastive encoder. In the v3 smoke (2026-09-28) the
    full context cost it 3× server latency and buried the ARMED/hostile events in 5 kB of state;
    it waited under attack and died 11 times. So CLM gets less text of each kind, not less truth.
    """

    state_chars: int
    history_steps: int
    event_chars: int
    max_events: int
    previous_states: bool
    outcome_chars: int


def body_events(body: Body, event_chars: int | None = None, max_events: int | None = None) -> list[str]:
    """Body events since the last read, deduplicated (with ×N) and capped, oldest first.

    Advancements ("has made the advancement [Stone Age]") are kept and shortened to
    ``advancement: Stone Age`` — the game's own statement of progress.
    """
    try:
        raw = body.events(0)
    except Exception:  # noqa: BLE001 - a missed event read is a gap in the state, not a crash
        return []
    counts: dict[str, int] = {}
    for e in raw:
        if e.kind not in BODY_EVENT_KINDS:
            continue
        adv = advancement(e.text)
        text = (
            f"advancement: {adv}" if adv else _COORDS.sub("", re.sub(r"\s+", " ", e.text)).strip()[: event_chars or cfg.policy.event_chars]
        )
        key = f"{e.name}: {text}" if e.kind == "reflex" and e.name else text
        counts[key] = counts.get(key, 0) + 1
    lines = [k if n == 1 else f"{k} ×{n}" for k, n in counts.items()]
    return lines[-(max_events or cfg.policy.max_events) :]


def _has(inv: dict[str, int], names: tuple[str, ...]) -> str | None:
    return next((n for n in names if inv.get(n, 0) > 0), None)


def _count(inv: dict[str, int], pred: Callable[[str], bool]) -> int:
    return sum(c for n, c in inv.items() if pred(n))


def _fmt(p: dict[str, Any]) -> str:
    return f"({int(p['x'])}, {int(p['y'])}, {int(p['z'])})"


def _near_table(ob: Observation) -> dict[str, Any] | None:
    return next((t for t in ob.tables if float(t.get("distance", 99)) <= 4.0), None)


def _craftable(ob: Observation, item: str) -> bool:
    """A conservative, code-owned affordability check (craft_item itself is the final word)."""
    inv = ob.inv
    logs = _count(inv, lambda n: n.endswith("_log"))
    planks = _count(inv, lambda n: n.endswith(PLANKS_SUFFIX)) + 4 * logs
    sticks = inv.get("stick", 0) + 2 * planks  # 2 planks → 4 sticks
    cobble = inv.get("cobblestone", 0) + inv.get("cobbled_deepslate", 0)
    table = _near_table(ob) is not None
    need = {
        "crafting_table": planks >= 4,
        "stick": planks >= 2,
        "wooden_pickaxe": table and planks >= 5,
        "wooden_sword": table and planks >= 3,
        "wooden_axe": table and planks >= 5,
        "stone_pickaxe": table and cobble >= 3 and sticks >= 2,
        "stone_sword": table and cobble >= 2 and sticks >= 1,
        "furnace": table and cobble >= 8,
        "torch": inv.get("coal", 0) + inv.get("charcoal", 0) >= 1 and sticks >= 1,
    }
    return need.get(item, False)


@dataclass(frozen=True, eq=False)
class MacroAction:
    """One legal next move: an id the model picks, a description it reads, the tool steps code runs."""

    id: str
    description: str
    steps: tuple[ToolStep, ...] = ()
    #: Placement that searches a ring of spots around the body instead of one fixed offset.
    #: The 2026-09-26 run placed at +1x only and failed 24× in a row wherever a trunk or leaf
    #: stood there — a harness defect every model then "chose"; see ``place_near``.
    place_ring: bool = False
    #: Keep going past a failed step (a shaft meets air pockets); the bag decides success.
    tolerant: bool = False
    #: Seconds of game time this action spends after its steps (only `wait`).
    wait_s: float = 0.0
    #: What ``verify.check`` needs to judge this action from the world (item, mob, thresholds).
    meta: dict[str, Any] = field(default_factory=dict)

    def run(self, body: Body) -> list[ToolOutcome]:
        """Run the steps in order; stop at the first failure (its advice is the outcome)."""
        if self.place_ring:
            return [place_near(body, "crafting_table")]
        outs: list[ToolOutcome] = []
        for tool, args in self.steps:
            out = body.call(tool, **args)
            outs.append(out if out.ok or not self.tolerant else ToolOutcome(ok=True, text=f"(skipped: {out.short(80)})", data=None))
            if not out.ok and not self.tolerant:
                break
        if self.wait_s:
            time.sleep(self.wait_s)
        return outs


#: Placement ring: the 8 neighbours at foot level, then one block up, nearest first.
RING: tuple[tuple[int, int, int], ...] = tuple(
    (dx, dy, dz) for dy in (0, 1) for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1))
)


def place_near(body: Body, item: str) -> ToolOutcome:
    """Place ``item`` at the first free spot in RING; the last refusal is the outcome."""
    p = body.state().get("pos") or {"x": 0, "y": 0, "z": 0}
    # floor, not int(): int(-3.5) is -3, one block east of a body standing at x=-3.5.
    x0, y0, z0 = math.floor(p["x"]), math.floor(p["y"]), math.floor(p["z"])
    last = ToolOutcome(ok=False, text="no spot tried", data=None)
    for dx, dy, dz in RING:
        last = body.call("place_block", item=item, x=x0 + dx, y=y0 + dy, z=z0 + dz)
        if last.ok:
            return last
    return last


def _dig_column(pos: dict[str, Any], depth: int) -> tuple[ToolStep, ...]:
    """dig_block steps for a 1×1 shaft offset one block east (never dig the block you stand on)."""
    x, y, z = math.floor(pos["x"]) + 1, math.floor(pos["y"]), math.floor(pos["z"])
    return tuple(("dig_block", {"x": x, "y": y - d, "z": z}) for d in range(0, depth + 1))


def _xyz(p: dict[str, Any]) -> dict[str, Any]:
    return {"x": p["x"], "y": p["y"], "z": p["z"]}


COLLECT: ToolStep = ("collect_ground_items", {"radius": 8})
#: Macro-actions whose success is measured by the bag, not by the tool's return.
STONE_DROPS = ("cobblestone", "cobbled_deepslate", "andesite", "diorite", "granite")
COAL_DROPS = ("coal",)
IRON_DROPS = ("raw_iron", "iron_ore", "deepslate_iron_ore")
#: POLICY_WAIT_S lives in config; mirrored here for the MacroAction default.
WAIT_S = float(cfg.policy.wait_s)
CRAFT_ORDER = ("crafting_table", "wooden_pickaxe", "wooden_sword", "stone_pickaxe", "stone_sword", "furnace", "torch", "wooden_axe")
SHELTER_BLOCKS = ("cobblestone", "dirt", "cobbled_deepslate")


def _tree_actions(ob: Observation, axe: str | None) -> list[MacroAction]:
    equip: tuple[ToolStep, ...] = (("equip_item", {"item": axe}),) if axe else ()
    return [
        MacroAction(
            f"fell_tree_{i}",
            f"Fell the {t.get('block')} tree ({t.get('veinSize')} logs) at {_fmt(t['position'])}, {t.get('distance')} blocks away, and pick up the logs.",
            (*equip, ("dig_vein", {**_xyz(t["position"]), "maxBlocks": 24}), COLLECT),
            meta={"gain": LOGS},
        )
        for i, t in enumerate(ob.trees[:2])
    ]


def _craft_actions(ob: Observation) -> list[MacroAction]:
    out: list[MacroAction] = []
    for item in CRAFT_ORDER:
        if ob.inv.get(item, 0) and item != "torch":
            continue  # one of each tool/station is enough for this suite
        if item == "crafting_table" and _near_table(ob):
            continue  # one in reach already: a second is a wasted step (all three models took it, 2026-09-26 ablation)
        if not _craftable(ob, item):
            continue
        count = 4 if item == "torch" else 1
        out.append(
            MacroAction(
                f"craft_{item}",
                f"Craft {count} {item} (intermediates such as planks and sticks are chained automatically).",
                (("craft_item", {"item": item, "count": count}),),
            )
        )
    return out


def _table_actions(ob: Observation) -> list[MacroAction]:
    if _near_table(ob):
        return []
    out: list[MacroAction] = []
    if ob.inv.get("crafting_table", 0):
        out.append(
            MacroAction(
                "place_crafting_table",
                "Place a crafting table next to you so 3×3 recipes (pickaxes, swords, furnace) become craftable here.",
                (),
                place_ring=True,
            )
        )
    if ob.tables:
        t = ob.tables[0]
        out.append(
            MacroAction(
                "walk_to_table",
                f"Walk to the crafting table at {_fmt(t['position'])} ({t.get('distance')} blocks) to use 3×3 recipes.",
                (("go_to", {**_xyz(t["position"]), "range": 2}),),
            )
        )
    return out


def _mining_actions(ob: Observation, pick: str | None) -> list[MacroAction]:
    if not pick:
        return []  # no pickaxe, no mining: stone drops nothing by hand
    equip: ToolStep = ("equip_item", {"item": pick})
    out: list[MacroAction] = []
    exposed = [s for s in ob.stone if s.get("exposed")]
    if exposed:
        s = exposed[0]
        out.append(
            MacroAction(
                "mine_stone",
                f"Mine the exposed {s.get('block')} patch ({s.get('veinSize')} blocks) at {_fmt(s['position'])} with the {pick} and pick up the cobblestone.",
                (equip, ("dig_vein", {**_xyz(s["position"]), "also": "cobblestone,andesite,diorite,granite", "maxBlocks": 12}), COLLECT),
                meta={"gain": STONE_DROPS},
            )
        )
    elif ob.stone:
        # Buried stone: dig_vein from the surface cannot reach it (it digs nothing and says
        # "is the position right?" — 2026-09-26, 27× in one episode). Dig the column down to
        # it instead: dig_block walks into range per block, and each dug stone drops cobble.
        out.append(
            MacroAction(
                "dig_down_to_stone",
                f"Dig straight down through the ground under you with the {pick} to reach the buried stone ~{int(ob.pos['y'] - ob.stone[0]['position']['y'])} blocks below, collecting what drops.",
                (equip, *_dig_column(ob.pos, depth=6), COLLECT),
                tolerant=True,
            )
        )
    for kind, veins in (("coal", ob.coal), ("iron", ob.iron)):
        if kind == "iron" and pick == "wooden_pickaxe":
            continue  # a wooden pick drops nothing from iron ore — a fact, not a preference
        out += [
            MacroAction(
                f"mine_{kind}",
                f"Mine the {v.get('block')} vein ({v.get('veinSize')} blocks, exposed={v.get('exposed')}) at {_fmt(v['position'])}, {v.get('distance')} blocks away, with the {pick}.",
                (equip, ("dig_vein", {**_xyz(v["position"]), "also": v.get("block"), "maxBlocks": 16}), COLLECT),
                meta={"gain": COAL_DROPS if kind == "coal" else IRON_DROPS},
            )
            # Buried ore is unreachable to dig_vein from the surface (same zero-dig as stone);
            # only exposed veins are offered. A shaft or exploration gets to the rest.
            for v in [v for v in veins if v.get("exposed")][:1]
        ]
    return out


def _nearest(ob: Observation, names: tuple[str, ...]) -> dict[str, Any] | None:
    hits = sorted((e for e in ob.entities if e.get("name") in names), key=lambda e: float(e.get("distance", 99)))
    return hits[0] if hits else None


def _danger_actions(ob: Observation, sword: str | None) -> list[MacroAction]:
    h = _nearest(ob, HOSTILE)
    if h is None:
        return []
    out: list[MacroAction] = []
    if h.get("name") != "creeper":  # punching a primed creeper is the death creeper_flee exists to avoid
        out.append(
            MacroAction(
                "fight_hostile",
                f"Fight the {h.get('name')} {h.get('distance')} blocks away until it dies (auto-equips your best weapon{'' if sword else ' — you have no sword'}).",
                (("attack_entity", {"entity": str(h.get("name")), "until": "dead", "minHealth": 8, "timeoutSec": 20}),),
                meta={"mob": h.get("name")},
            )
        )
    me = ob.pos
    hp = h.get("position") or me
    dx, dz = me["x"] - hp["x"], me["z"] - hp["z"]
    norm = max(1e-6, (dx * dx + dz * dz) ** 0.5)
    away = {"x": round(me["x"] + 14 * dx / norm), "y": round(me["y"]), "z": round(me["z"] + 14 * dz / norm), "range": 3}
    out.append(
        MacroAction(
            "flee_hostile", f"Run ~14 blocks directly away from the {h.get('name')}.", (("go_to", away),), meta={"mob": h.get("name")}
        )
    )
    return out


def _food_actions(ob: Observation) -> list[MacroAction]:
    out: list[MacroAction] = []
    food = float(ob.state.get("food") or 20)
    has_meal = bool(ob.state.get("foodPortions"))
    a = _nearest(ob, FOOD_MOBS)
    if a is not None and (food < 16 or not has_meal):
        out.append(
            MacroAction(
                "hunt_animal",
                f"Kill the {a.get('name')} {a.get('distance')} blocks away for food and pick up the drop.",
                (("attack_entity", {"entity": str(a.get("name")), "until": "dead", "timeoutSec": 20}), COLLECT),
            )
        )
    if has_meal and food <= 16:
        out.append(MacroAction("eat", "Eat the best food in your bag.", (("eat", {}),)))
    return out


def box_plan(anchor: dict[str, int], w: int = 3, d: int = 3, h: int = 3) -> list[tuple[int, int, int]]:
    """The positions ``build_structure(shape='box', door='none')`` fills: walls plus a flat roof, no floor.

    Mirrors src/tools/blueprints.ts ``draftStructure`` (pinned by test_policy); used only to ask the
    world whether those positions are solid, never to place anything.
    """
    out: list[tuple[int, int, int]] = []
    for x in range(w):
        for z in range(d):
            for y in range(h):
                shell = x in (0, w - 1) or z in (0, d - 1)
                if shell or y == h - 1:
                    out.append((anchor["x"] + x, anchor["y"] + y, anchor["z"] + z))
    return out


def _shelter_action(ob: Observation) -> list[MacroAction]:
    inv = ob.inv
    usable = {n: c for n, c in inv.items() if n in SHELTER_BLOCKS or n.endswith(PLANKS_SUFFIX)}
    if sum(usable.values()) < 28 or ob.state.get("isDay") is not False:
        return []
    item = max(usable, key=lambda n: usable[n])
    me = ob.pos
    anchor = {"x": math.floor(me["x"]) - 1, "y": math.floor(me["y"]), "z": math.floor(me["z"]) - 1}
    return [
        MacroAction(
            "build_shelter",
            f"Build a 3×3×3 {item} box shelter around yourself for the night (no door).",
            (("build_structure", {"shape": "box", "item": item, "anchor": anchor, "width": 3, "depth": 3, "height": 3, "door": "none"}),),
            # Sealed = every plan position solid, whatever fills it (dug-into stone seals too).
            meta={"item": item, "plan": box_plan(anchor)},
        )
    ]


def _movement_actions(ob: Observation) -> list[MacroAction]:
    me = ob.pos
    out = [
        MacroAction(
            f"explore_{label}",
            f"Walk ~24 blocks {label} to find new resources.",
            (("go_to", {"x": round(me["x"] + dx), "y": round(me["y"]), "z": round(me["z"] + dz), "range": 4}),),
            meta={"min_move": cfg.policy.explore_min_move},
        )
        for label, dx, dz in (("north", 0, -24), ("east", 24, 0))
    ]
    # `wait` passes game time (≈5 s): a no-op step is instantaneous, so the night task's
    # decision budget ran out long before dawn for any arm that waited correctly (2026-09-26 run).
    out.append(
        MacroAction(
            "wait",
            "Do nothing this step (only when every other option is worse, e.g. safe inside a shelter at night).",
            (("stop_moving", {}),),
            wait_s=WAIT_S,
        )
    )
    return out


def candidates(ob: Observation) -> list[MacroAction]:
    """Only the actions that are legal and grounded in THIS observation, most urgent first."""
    inv = ob.inv
    pick, sword, axe = _has(inv, PICKS), _has(inv, SWORDS), _has(inv, AXES)
    drops: list[MacroAction] = (
        [MacroAction("collect_drops", f"Pick up the {ob.drops} dropped item(s) lying within 8 blocks.", (COLLECT,))] if ob.drops else []
    )
    ordered = [
        *_danger_actions(ob, sword),
        *_food_actions(ob),
        *_tree_actions(ob, axe),
        *_craft_actions(ob),
        *_table_actions(ob),
        *_mining_actions(ob, pick),
        *drops,
        *_shelter_action(ob),
    ]
    tail = _movement_actions(ob)  # explore + wait always survive the cap
    return ordered[: max(0, cfg.policy.max_candidates - len(tail))] + tail


#: Tool boilerplate that carries no result and would eat an outcome's character budget.
_BOILERPLATE = re.compile(r"Equipped [\w ]+ to hand, confirmed in the slot\.\s*\|?\s*|\s*\|\s*No dropped items nearby\.")


def _res(t: dict[str, Any]) -> str:
    """A resource cluster as the model reads it: what, how big, where, how far, reachable or buried."""
    pos = f" at {_fmt(t['position'])}" if t.get("position") else ""
    return f"{t.get('block')} x{t.get('veinSize')}{pos} {t.get('distance')}m {'exposed' if t.get('exposed') else 'buried'}"


def _mob(e: dict[str, Any]) -> str:
    p = e.get("position")
    return f"{e.get('name')} {e.get('distance')}m" + (f" at {_fmt(p)}" if isinstance(p, dict) and "x" in p else "")


#: Tool-authored next-step hints ("vein continues — call again to keep going", "Fix: re-run …").
_HINT = re.compile(r"\((?:[^()]*?)(?:call again|re-run|run again)[^()]*\)|(?:Fix|Next|Try): [^.|]+\.?", re.IGNORECASE)


#: A tool line that reports no result ("Inventory change: nothing", "No dropped items nearby").
_EMPTY = re.compile(r"^(?:No dropped items nearby\.?|\(skipped: .*\)|.*Inventory change: nothing\..*)$", re.IGNORECASE)


def step_outcome(texts: list[str]) -> str:
    """What the model reads for a step: each tool call's text cleaned, empty ones dropped, NEWEST FIRST.

    A shaft is several dig_block calls whose first answers are usually 'Inventory change: nothing'
    (leaf litter, grass). v3's first smoke spent the whole 240-char budget on those while the
    step's net gain was the cobblestone at the bottom. The net effect is in ``change.bag``; the
    outcome should carry the latest words, and the tool's own next-step hint first.
    """
    parts = [clean_outcome(t) for t in texts]
    useful = [t for t in parts if t and not _EMPTY.match(t)] or [t for t in parts if t][-1:]
    return " | ".join(reversed(useful))


def clean_outcome(text: str) -> str:
    """The outcome with equip/pickup boilerplate removed, whitespace collapsed, and the tool's own
    next-step hint moved to the front so it survives truncation."""
    t = re.sub(r"\s+", " ", _BOILERPLATE.sub("", text)).strip(" |")
    # The tool's own next-step hint must survive truncation: lead with it.
    hints = _HINT.findall(t)
    return (" ".join(dict.fromkeys(h.strip() for h in hints)) + " | " + t) if hints else t


def snapshot(ob: Observation) -> dict[str, Any]:
    """The body facts a step started from: shown beside the step, so cause and effect line up."""
    s = ob.state
    return {
        "health": s.get("health"),
        "food": s.get("food"),
        "day": s.get("isDay"),
        "timeOfDay": s.get("timeOfDay"),
        "position": {k: round(float(v), 1) for k, v in ob.pos.items()},
        "standing_on": ob.standing_on,
        "in_water": ob.standing_on in ("water", "bubble_column") or any("drowning" in e for e in ob.events),
        "inventory": ob.inv,
    }


def delta(a: dict[str, Any], b: dict[str, Any], seconds: float) -> dict[str, Any]:
    """What one step changed in the body: health, food, distance moved, bag, and how long it took."""
    pa, pb = a.get("position") or {}, b.get("position") or {}
    moved = math.dist((pa.get("x", 0), pa.get("z", 0)), (pb.get("x", 0), pb.get("z", 0)))
    ia, ib = a.get("inventory") or {}, b.get("inventory") or {}
    bag = {n: ib.get(n, 0) - ia.get(n, 0) for n in set(ia) | set(ib) if ib.get(n, 0) != ia.get(n, 0)}
    out: dict[str, Any] = {"seconds": round(seconds, 1)}
    for k in ("health", "food"):
        if isinstance(a.get(k), (int, float)) and isinstance(b.get(k), (int, float)) and a[k] != b[k]:
            out[k] = round(float(b[k]) - float(a[k]), 1)
    if moved >= 0.5:
        out["moved"] = round(moved, 1)
    if pb.get("y") is not None and pa.get("y") is not None and abs(pb["y"] - pa["y"]) >= 1:
        out["dy"] = round(pb["y"] - pa["y"], 1)
    if bag:
        out["bag"] = {n: (f"+{d}" if d > 0 else str(d)) for n, d in sorted(bag.items())}
    return out


@dataclass
class Step:
    """One executed step, as the model is shown it next time."""

    step: int
    action: str
    verdict: str
    check: str
    outcome: str
    before: dict[str, Any]
    reason: str = ""
    change: dict[str, Any] = field(default_factory=dict)
    events: list[str] = field(default_factory=list)

    def line(self, chars: int, event_chars: int | None = None) -> dict[str, Any]:
        out: dict[str, Any] = {"step": self.step, "action": self.action, "verdict": self.verdict, "check": self.check}
        if self.reason:
            out["reason"] = self.reason
        if self.change:
            out["change"] = self.change
        if self.events:
            out["events"] = [e[:event_chars] for e in self.events] if event_chars else self.events
        return {**out, "outcome": self.outcome[:chars]}


def tally(steps: list[Step]) -> dict[str, str]:
    """Per action over the whole episode: attempts / done / failed, so a loop is visible at any window."""
    out: dict[str, list[int]] = {}
    for st in steps:
        t = out.setdefault(st.action, [0, 0, 0])
        t[0] += 1
        t[1] += st.verdict == "done"
        t[2] += st.verdict == "no_effect"
    return {k: f"{a} tried, {d} done, {f} no effect" for k, (a, d, f) in out.items()}


def decision_state(
    goal: str,
    ob: Observation,
    steps: list[Step],
    budget: dict[str, Any] | None = None,
    *,
    max_chars: int | None = None,
    max_steps: int | None = None,
    profile: ContextProfile | None = None,
) -> dict[str, Any]:
    """The observable projection the model judges from, bounded by ``max_chars``.

    ``recent_actions[i]`` and ``previous_states[i]`` are the same step: the body facts it started
    from, then what was done and what the world says happened. The current facts are the top-level
    fields. Trimmed oldest-first when over budget; ``action_tally`` and ``budget`` always stay.
    """
    cap = max_chars or (profile.state_chars if profile else cfg.policy.state_chars)
    window = steps[-(max_steps or (profile.history_steps if profile else cfg.policy.history_steps)) :]
    chars = profile.outcome_chars if profile else cfg.policy.outcome_chars
    ev_chars = profile.event_chars if profile else cfg.policy.event_chars
    state: dict[str, Any] = {
        "goal": goal,
        **snapshot(ob),
        "nearby": {
            "trees": [_res(t) for t in ob.trees],
            "stone": [_res(t) for t in ob.stone],
            "ores": [_res(t) for t in ob.coal + ob.iron],
            "crafting_table": [f"at {_fmt(t['position'])} {t.get('distance')}m" for t in ob.tables],
            "furnace": [f"at {_fmt(t['position'])} {t.get('distance')}m" for t in ob.furnaces],
            "mobs": [_mob(e) for e in ob.entities if e.get("name") not in ("item", "player")][:6],
            "dropped_items_within_8": ob.drops,
        },
        "threats": ob.state.get("threats") or [],
        "events_since_last_step": ob.events,
        "budget": {**(budget or {}), "deaths": int((ob.state.get("counters") or {}).get("deaths", 0))},
        "action_tally": tally(steps),
        # Never trimmed: the episode's progress markers, in the order they were earned.
        "advancements": [a.removeprefix("advancement: ") for st in steps for a in st.events if a.startswith("advancement: ")],
        "recent_actions": [st.line(chars, ev_chars) for st in window],
    }
    if profile is None or profile.previous_states:
        state["previous_states"] = [st.before for st in window]
    # No empty containers. v3 run 2 (c7e45cb): with {} / [] for action_tally, advancements, recent_actions,
    # events and threats on step 1, CLM picked `wait` (p 0.45–0.78) in all 9 wood_tools/cobble8/iron1
    # episodes; replaying those exact requests without the empties it picked an action 9/9, and Jev and Kev
    # kept their pick 23/24 (.agents/scratchpad/empties_ablation.log). An absent key says the same thing.
    state = _drop_empty(state)
    while len(json.dumps(state, ensure_ascii=False)) > cap and state.get("recent_actions"):
        state["recent_actions"] = state["recent_actions"][1:]
        if "previous_states" in state:
            state["previous_states"] = state["previous_states"][1:]
    return _drop_empty(state)


def _drop_empty(d: dict[str, Any]) -> dict[str, Any]:
    """Remove empty lists/dicts, one level into ``nearby`` too (an empty ``furnace: []`` is noise)."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, dict) and k == "nearby":
            v = {nk: nv for nk, nv in v.items() if not (isinstance(nv, (list, dict)) and not nv)}
        if isinstance(v, (list, dict)) and not v:
            continue
        out[k] = v
    return out


#: The instruction every policy arm gets — ONE string for all models, so the comparison is fair.
#: Ablation 2026-09-26 (4 fixed scenes, same request per model; .agents/scratchpad/instr-ablation.log):
#: long → Jev 3/4, Kev 3/4, CLM 1/4 (CLM collapsed to `wait` at 0.8); short → 3/4, 3/4, 4/4.
#: A contrastive encoder is sensitive to instruction text that shares words with an option
#: ("Stay alive first…" vs the `wait` description); Jev and Kev were not.
INSTRUCTION_VARIANTS: dict[str, str] = {
    "short": "Which action best advances the `goal`?",
    "long": (
        "You control a Minecraft survival bot. Given its `goal`, `vitals`, `inventory`, `nearby` resources and `recent_actions`, "
        "choose the ONE next action that most directly advances the goal. Stay alive first: handle close threats and hunger when they matter. "
        "Follow the tech tree (logs → crafting table → wooden pickaxe → cobblestone → stone tools → coal/iron). Avoid repeating an action that just failed."
    ),
}
INSTRUCTIONS = INSTRUCTION_VARIANTS.get(cfg.policy.instructions, INSTRUCTION_VARIANTS["short"])


#: The second question, asked in the same call: the model's own read of the last step's outcome.
#: Recorded beside the world-checked verdict; it drives nothing (the model decides what to do).
LAST_STEP_QUESTION = "Did the most recent action in `recent_actions` achieve what it set out to do?"


@dataclass
class Decision:
    step: int
    action: str
    confidence: float | None
    top: list[tuple[str, float]]
    model_id: str | None
    latency_ms: float
    server_latency_ms: float | None
    usage: dict[str, int]
    options: int
    state_chars: int
    outcome: str = ""
    ok: bool = True
    error: str | None = None
    #: done | in_progress | no_effect | unverified | waited, from the world (``verify.check``).
    verdict: str = "no_effect"
    check: str = ""
    #: Why, in the tool's words when it gave any (e.g. "is occupied by stone"), else the check's finding.
    reason: str = ""
    #: The model's answer to LAST_STEP_QUESTION about the PREVIOUS step (None on step 1 / on error).
    last_step_p: float | None = None
    #: The exact request: the state the model saw and the option ids → descriptions.
    request: dict[str, Any] = field(default_factory=dict)


class PolicyBrain:
    """Observe → candidates → one ask (next action + last-step judgment) → execute → verify."""

    def __init__(
        self,
        model: DecisionModel,
        body: Body,
        goal: str,
        *,
        max_state_chars: int | None = None,
        max_history: int | None = None,
        profile: ContextProfile | None = None,
    ) -> None:
        self.model = model
        self.body = body
        self.goal = goal
        self.max_state_chars = max_state_chars
        self.max_history = max_history
        self.profile = profile
        self.steps: list[Step] = []
        self.decisions: list[Decision] = []
        #: Set by the harness each step: steps/seconds used against the task's budgets.
        self.budget: dict[str, Any] = {}
        #: Measured seconds per shelter position made solid, for the in-progress ETA.
        self._block_s: list[float] = []

    async def step(self) -> Decision:
        t_start = time.monotonic()
        # observe() drains body events raised since the previous step ENDED (the ones raised during
        # it were drained at its end and sit on its recent_actions line), so nothing shows twice.
        ob = observe(self.body, self.profile)
        acts = candidates(ob)
        state = decision_state(
            self.goal, ob, self.steps, self.budget, max_chars=self.max_state_chars, max_steps=self.max_history, profile=self.profile
        )
        q: dict[str, Any] = {"next_action": Choice(instructions=INSTRUCTIONS, options={a.id: a.description for a in acts})}
        if self.steps:
            q["last_step_succeeded"] = YesNo(LAST_STEP_QUESTION)
        request = {"state": state, "options": {a.id: a.description for a in acts}}
        n = len(self.decisions) + 1
        t0 = time.perf_counter()
        try:
            resp = await self.model.ask(state, q)
        except Exception as err:  # noqa: BLE001 - a failed decision is data, not a crash
            d = Decision(n, "error", None, [], None, (time.perf_counter() - t0) * 1000, None, {}, len(acts), len(json.dumps(state)))
            d.ok, d.error, d.verdict, d.request = False, f"{type(err).__name__}: {err}"[:300], "no_effect", request
            d.check, d.reason = "the decision call raised; no action ran", d.error
            self.decisions.append(d)
            return d
        latency = (time.perf_counter() - t0) * 1000
        ans = resp.answers["next_action"]
        top = sorted(ans.probabilities.items(), key=lambda kv: -kv[1])[:3]  # type: ignore[union-attr]
        usage = dict(resp.usage) if resp.usage else {}
        last = resp.answers.get("last_step_succeeded")
        d = Decision(
            step=n,
            action=ans.choice,  # type: ignore[union-attr]
            confidence=ans.confidence,
            top=[(k, round(v, 4)) for k, v in top],
            model_id=resp.model_id,
            latency_ms=round(latency, 1),
            server_latency_ms=getattr(self.model, "last_server_latency_ms", None),
            usage={k: int(v) for k, v in usage.items() if isinstance(v, (int, float))},
            options=len(acts),
            state_chars=len(json.dumps(state)),
            last_step_p=round(float(last.probability), 4) if last is not None else None,  # type: ignore[union-attr]
            request=request,
        )
        act = next((a for a in acts if a.id == d.action), None)
        model_outcome = ""
        if act is None:
            d.ok, d.error = False, f"model chose {d.action!r}, not one of the offered ids"
            d.verdict, d.check, d.reason = "no_effect", "no such option; nothing ran", d.error
        else:
            plan = [{"x": x, "y": y, "z": z} for x, y, z in act.meta.get("plan", [])] or None
            before = Snapshot.take(self.body, HOSTILE, plan, time.monotonic())
            self.body.set_busy(True)
            try:
                outs = act.run(self.body)
            finally:
                self.body.set_busy(False)
            after = Snapshot.take(self.body, HOSTILE, plan, time.monotonic())
            d.ok = all(o.ok for o in outs)  # transport only: kept for the record, never shown as "ok"
            full = " | ".join(o.text for o in outs)
            d.outcome = " | ".join(o.short(160) for o in outs)[:400]
            model_outcome = step_outcome([o.text for o in outs])
            if plan:
                gained = len(after.solid) - len(before.solid)
                if gained > 0:
                    self._block_s.append((after.t - before.t) / gained)
            spb = sum(self._block_s) / len(self._block_s) if self._block_s else None
            v = check(act.id, act.meta, before, after, full, spb)
            d.verdict, d.check, d.reason = v.verdict, v.check, v.reason
            if not d.ok and d.verdict == "done":
                d.check += " (a tool call raised, but the world shows the effect)"
        before_facts = snapshot(ob)
        t_end = time.monotonic()
        after_ob = Observation(state=self.body.state(), standing_on=ob.standing_on)
        change = delta(before_facts, snapshot(after_ob), t_end - t_start)
        # Events that fired DURING this action belong to this step (read now, so the next observe
        # starts clean); the pre-step events were already shown in this step's state.
        during = body_events(self.body, *((self.profile.event_chars, self.profile.max_events) if self.profile else (None, None)))
        self.steps.append(
            Step(n, d.action, d.verdict, d.check, model_outcome or clean_outcome(d.error or ""), before_facts, d.reason, change, during)
        )
        self.decisions.append(d)
        return d
