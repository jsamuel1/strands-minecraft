"""World-checked verdicts: a post-condition per action, never the tool's own words, plus the reason."""

from __future__ import annotations

from mcbrain.policy import box_plan
from mcbrain.verify import Snapshot, check, tool_reason


def snap(inv: dict[str, int] | None = None, **kw: object) -> Snapshot:
    base: dict[str, object] = {"inv": inv or {}, "pos": {"x": 0.0, "y": 64.0, "z": 0.0}, "food": 20.0, "tables_near": 0, "hostiles": {}}
    base.update(kw)
    return Snapshot(**base)  # type: ignore[arg-type]


PLAN = box_plan({"x": 0, "y": 60, "z": 0})
OCCUPIED = (
    "Blueprint: placed 0/25, 18 already in place. FAILED 7:\ncobblestone at (1, 60, 0) (offset 1,0,0): (1, 60, 0) is occupied by stone."
)


def test_box_plan_matches_the_tools_box_shape() -> None:
    """draftStructure box 3×3×3, door none, roof on, no floor: 8 walls × 3 layers + 1 roof centre = 25."""
    assert len(PLAN) == 25 and len(set(PLAN)) == 25
    assert (1, 60, 1) not in PLAN and (1, 61, 1) not in PLAN, "the inside stays open for the body"
    assert (1, 62, 1) in PLAN, "the roof closes it"


def test_a_shelter_sealed_by_stone_is_done_whatever_the_tool_counted() -> None:
    """v3 smoke 2026-09-28: 415 'occupied by stone' refusals from a shelter the stone had already sealed."""
    v = check("build_shelter", {"plan": PLAN}, snap(solid=frozenset(PLAN[:18])), snap(solid=frozenset(PLAN)), OCCUPIED)
    assert v.verdict == "done" and "25/25" in v.check


def test_a_shelter_that_grew_is_in_progress_with_an_eta() -> None:
    v = check(
        "build_shelter", {"plan": PLAN}, snap(solid=frozenset(PLAN[:10])), snap(solid=frozenset(PLAN[:20])), OCCUPIED, s_per_block=1.5
    )
    assert v.verdict == "in_progress"
    assert "20/25" in v.check and "5 open" in v.check and "~8s to finish" in v.check
    assert v.reason == "is occupied by stone", "the tool's own words, coordinates stripped"


def test_a_shelter_that_did_not_change_is_no_effect_with_the_reason() -> None:
    v = check("build_shelter", {"plan": PLAN}, snap(solid=frozenset(PLAN[:18])), snap(solid=frozenset(PLAN[:18])), OCCUPIED)
    assert v.verdict == "no_effect" and v.reason == "is occupied by stone"


def test_crafting_is_done_only_when_the_item_count_rises() -> None:
    assert check("craft_wooden_pickaxe", {}, snap(), snap({"wooden_pickaxe": 1})).verdict == "done"
    v = check("craft_wooden_pickaxe", {}, snap(), snap(), "Missing materials — gather these first")
    assert v.verdict == "no_effect" and v.reason.startswith("Missing materials")


def test_placing_a_table_is_judged_by_the_world_not_the_bag() -> None:
    assert check("place_crafting_table", {}, snap(tables_near=0), snap(tables_near=1)).verdict == "done"
    v = check("place_crafting_table", {}, snap(tables_near=1), snap(tables_near=1))
    assert v.verdict == "no_effect" and v.reason == "no new table in reach"


def test_a_gather_counts_only_the_items_it_is_for() -> None:
    meta = {"gain": ("cobblestone",)}
    assert check("mine_stone", meta, snap(), snap({"dirt": 3})).verdict == "no_effect", "dirt is not cobblestone"
    assert check("mine_stone", meta, snap(), snap({"cobblestone": 3})).verdict == "done"


def test_explore_needs_real_movement_and_wait_is_neither() -> None:
    far = snap(pos={"x": 20.0, "y": 64.0, "z": 0.0})
    assert check("explore_east", {"min_move": 8}, snap(), far).verdict == "done"
    v = check("explore_east", {"min_move": 8}, snap(), snap(), "COULD NOT REACH (1, 2, 3): out of time after 31s")
    assert v.verdict == "no_effect" and v.reason.startswith("COULD NOT REACH")
    assert check("wait", {}, snap(), snap()).verdict == "waited"


def test_fight_flee_eat_hunt() -> None:
    assert check("fight_hostile", {"mob": "zombie"}, snap(hostiles={"zombie": 3.0}), snap()).verdict == "done"
    assert check("fight_hostile", {"mob": "zombie"}, snap(hostiles={"zombie": 3.0}), snap(hostiles={"zombie": 2.0})).verdict == "no_effect"
    assert check("flee_hostile", {"mob": "zombie"}, snap(hostiles={"zombie": 3.0}), snap(hostiles={"zombie": 12.0})).verdict == "done"
    assert check("eat", {}, snap(food=10.0), snap(food=16.0)).verdict == "done"
    assert check("hunt_animal", {}, snap(), snap({"leather": 1})).verdict == "no_effect", "leather is not food"
    assert check("hunt_animal", {}, snap(), snap({"beef": 1})).verdict == "done"


def test_an_unknown_action_is_unverified_not_ok() -> None:
    assert check("do_a_dance", {}, snap(), snap()).verdict == "unverified"


def test_tool_reason_takes_the_first_refusal_and_strips_coordinates() -> None:
    assert (
        tool_reason("Fought cow for 0 swing(s): gave up: COULD NOT REACH (21, 64, 91): out of time")
        == "gave up: COULD NOT REACH: out of time"
    )
    assert tool_reason("Dug 5 block(s) of oak_log. Inventory change: +5 oak_log.") == ""
    assert tool_reason("Dug 0 block(s). | No dropped items nearby.") == "", "pickup boilerplate is not a reason"
    assert tool_reason("No 'oak_log' / 'birch_log' found within 48 blocks.") == "No 'oak_log' / 'birch_log' found within 48 blocks"
