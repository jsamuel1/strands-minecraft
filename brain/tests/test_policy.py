"""The System One policy: legal-only candidates, one Choice, code executes."""

from __future__ import annotations

from mcbrain.policy import Observation, PolicyBrain, Step, candidates, clean_outcome, decision_state

from .fakes import FakeBody, ScriptedDecision

TREE = [{"position": {"x": 5, "y": 64, "z": 3}, "distance": 6.1, "veinSize": 5, "exposed": True, "block": "oak_log"}]
STONE = [{"position": {"x": 2, "y": 62, "z": 1}, "distance": 3.0, "veinSize": 30, "exposed": True, "block": "stone"}]
IRON = [{"position": {"x": 9, "y": 40, "z": 9}, "distance": 25.0, "veinSize": 4, "exposed": True, "block": "iron_ore"}]


def ob(inv: dict[str, int] | None = None, **kw: object) -> Observation:
    state = {
        "pos": {"x": 0, "y": 64, "z": 0},
        "health": 20,
        "food": 20,
        "inv": inv or {},
        "foodPortions": 0,
        "isDay": True,
    }
    return Observation(state=state, **kw)  # type: ignore[arg-type]


def ids(o: Observation) -> list[str]:
    return [a.id for a in candidates(o)]


def test_fresh_spawn_offers_trees_and_exploration_but_no_tools() -> None:
    got = ids(ob(trees=TREE))
    assert "fell_tree_0" in got and "explore_north" in got and "wait" in got
    assert not any(i.startswith("craft_") for i in got), "nothing is affordable with an empty bag"
    assert "mine_stone" not in got, "no pickaxe, no mining"


def test_crafting_is_only_offered_when_affordable_and_3x3_needs_a_table() -> None:
    got = ids(ob({"oak_log": 3}))
    assert "craft_crafting_table" in got and "craft_wooden_pickaxe" not in got
    with_table = ob(
        {"oak_log": 3, "crafting_table": 1},
        tables=[{"position": {"x": 1, "y": 64, "z": 0}, "distance": 1.5, "block": "crafting_table"}],
    )
    got = ids(with_table)
    assert "craft_wooden_pickaxe" in got and "place_crafting_table" not in got


def test_carrying_a_table_without_one_nearby_offers_to_place_it() -> None:
    assert "place_crafting_table" in ids(ob({"crafting_table": 1}))


def test_buried_stone_is_reached_by_a_shaft_not_an_unreachable_vein() -> None:
    """2026-09-26: dig_vein at buried stone dug 0 blocks, 27 times in one episode."""
    buried = [{**STONE[0], "exposed": False}]
    got = ids(ob({"wooden_pickaxe": 1}, stone=buried))
    assert "mine_stone" not in got and "dig_down_to_stone" in got
    shaft = next(a for a in candidates(ob({"wooden_pickaxe": 1}, stone=buried)) if a.id == "dig_down_to_stone")
    digs = [args for tool, args in shaft.steps if tool == "dig_block"]
    assert shaft.tolerant and all(a["x"] == 1 and a["z"] == 0 for a in digs), "one block east, never underfoot"


def test_a_wooden_pick_is_never_offered_iron() -> None:
    got = ids(ob({"wooden_pickaxe": 1}, stone=STONE, iron=IRON))
    assert "mine_stone" in got and "mine_iron" not in got
    assert "mine_iron" in ids(ob({"stone_pickaxe": 1}, iron=IRON))


def test_creepers_are_fled_never_fought() -> None:
    got = ids(ob(entities=[{"name": "creeper", "distance": 4, "position": {"x": 3, "y": 64, "z": 0}}]))
    assert "flee_hostile" in got and "fight_hostile" not in got


#: A realistic previous_state: a mid-game bag makes each aligned step ~600 chars, as in live runs.
BAG = {"oak_log": 12, "birch_log": 7, "oak_planks": 8, "stick": 4, "cobblestone": 31, "wooden_pickaxe": 1, "crafting_table": 1, "dirt": 9}


def _steps(n: int, outcome: str = "x" * 3000) -> list[Step]:
    before = {"health": 20, "food": 18, "day": False, "timeOfDay": 14000, "position": {"x": 1.5, "y": 64.0, "z": -3.5}, "inventory": BAG}
    return [Step(i, "fell_tree_0", "no_effect", "gained nothing (counting oak_log, birch_log)", outcome, before) for i in range(1, n + 1)]


def test_state_is_bounded_trims_oldest_first_and_keeps_history_aligned() -> None:
    import json

    s = decision_state("collect 16 logs", ob(trees=TREE), _steps(12), max_chars=6000)
    assert s["goal"] == "collect 16 logs"
    assert len(json.dumps(s)) <= 6000
    assert len(s["recent_actions"]) == len(s["previous_states"]) >= 1
    assert s["recent_actions"][-1]["step"] == 12, "the newest step survives trimming"
    assert s["action_tally"] == {"fell_tree_0": "12 tried, 0 done, 12 no effect"}, "the tally is never trimmed"


def test_a_bigger_budget_keeps_more_history() -> None:
    """CLM's 2,048-token encoder gets less history than Kev and Jev (operator, 2026-09-28)."""
    steps = _steps(64, "y" * 240)
    clm = decision_state("g", ob(), steps, max_chars=6000, max_steps=12)
    kev = decision_state("g", ob(), steps, max_chars=24000, max_steps=32)
    jev = decision_state("g", ob(), steps, max_chars=48000, max_steps=64)
    assert len(clm["recent_actions"]) < len(kev["recent_actions"]) < len(jev["recent_actions"])


def test_recent_actions_i_and_previous_states_i_are_the_same_step() -> None:
    steps = [Step(i, f"a{i}", "done", "", "", {"health": 20 - i}) for i in range(1, 4)]
    s = decision_state("g", ob(), steps)
    assert [a["step"] for a in s["recent_actions"]] == [1, 2, 3]
    assert [p["health"] for p in s["previous_states"]] == [19, 18, 17]


def test_boilerplate_is_stripped_before_truncation() -> None:
    raw = "Equipped wooden_pickaxe to hand, confirmed in the slot. | Dug 12 block(s) of stone. | No dropped items nearby."
    assert clean_outcome(raw) == "Dug 12 block(s) of stone."


def _tools() -> dict[str, object]:
    return {
        "find_blocks": lambda b, name, **_: TREE if "log" in name else "No 'x' found within 48 blocks.",
        "look_around": {"entities": []},
        "dig_vein": "Dug 5 block(s) of oak_log (vein exhausted). Inventory change: +5 oak_log.",
        "collect_ground_items": "No dropped items nearby.",
    }


async def test_one_step_observes_asks_once_and_executes_the_choice() -> None:
    tools = _tools()

    def dig(b: FakeBody, **_: object) -> str:
        b._state["inv"] = {"oak_log": 5}  # the world really changed
        return "Dug 5 block(s) of oak_log (vein exhausted). Inventory change: +5 oak_log."

    tools["dig_vein"] = dig
    body = FakeBody(tools=tools)
    model = ScriptedDecision(["fell_tree_0"])
    brain = PolicyBrain(model, body, "collect 16 logs")  # type: ignore[arg-type]
    d = await brain.step()
    assert d.action == "fell_tree_0" and d.verdict == "done" and d.confidence == 0.6
    assert "+5 oak_log" in d.check, "the verdict names what the world showed"
    assert d.server_latency_ms == 1.5 and d.usage["inputTokens"] == 100
    ((state, questions),) = model.requests
    assert set(questions) == {"next_action"}, "no last-step question before there is a last step"
    assert "fell_tree_0" in questions["next_action"].options  # type: ignore[union-attr]
    assert ("dig_vein", {"x": 5, "y": 64, "z": 3, "maxBlocks": 24}) in body.calls
    assert body.busy == [True, False], "the body is told a deliberate action is in flight"
    assert d.request["options"]["fell_tree_0"].startswith("Fell the oak_log tree"), "the exact request is recorded"
    assert brain.steps[-1].verdict == "done" and brain.steps[-1].before["inventory"] == {}


async def test_placement_walks_a_ring_instead_of_failing_on_one_blocked_spot() -> None:
    """2026-09-26: a fixed +1x spot inside a trunk failed 24× in a row for every model."""
    from mcbrain.policy import place_near
    from mcbrain.sidecar import ToolOutcome

    def place(b: FakeBody, item: str, x: int, y: int, z: int) -> ToolOutcome:
        if (x, z) == (-3, 5):  # only the spot WEST of a body at x=-3.5 is free
            return ToolOutcome(ok=True, text=f"Placed {item}", data=None)
        return ToolOutcome(ok=False, text="occupied by oak_log", data=None)

    body = FakeBody(state={"pos": {"x": -3.5, "y": 64, "z": 5.2}}, tools={"place_block": place})
    out = place_near(body, "crafting_table")  # type: ignore[arg-type]
    assert out.ok
    tried = [(a["x"], a["z"]) for _, a in body.calls]
    assert tried[0] == (-3, 5), "floor(-3.5)=-4, so +1x is -3 — int() would have said -2"


async def test_failed_actions_stay_on_offer_and_the_model_is_told_and_asked() -> None:
    """v3: no hold-back rule. The model sees every legal option, the world-checked verdict of each
    past step, and is asked whether the last one worked (recorded, not acted on)."""
    from mcbrain.sidecar import ToolOutcome

    tools = {**_tools(), "dig_vein": ToolOutcome(ok=False, text="out of reach", data=None)}
    model = ScriptedDecision(["fell_tree_0", "fell_tree_0", "fell_tree_0"])
    brain = PolicyBrain(model, FakeBody(tools=tools), "logs")  # type: ignore[arg-type]
    for _ in range(3):
        await brain.step()
    third_state, third_q = model.requests[2]
    assert "fell_tree_0" in third_q["next_action"].options, "a failed action is never withheld"  # type: ignore[union-attr]
    assert set(third_q) == {"next_action", "last_step_succeeded"}
    assert [a["verdict"] for a in third_state["recent_actions"]] == ["no_effect", "no_effect"]
    assert all(a["reason"] for a in third_state["recent_actions"]), "every no_effect says why"
    assert third_state["action_tally"]["fell_tree_0"] == "2 tried, 0 done, 2 no effect"
    assert brain.decisions[2].last_step_p == 0.2


async def test_a_gather_that_gains_nothing_is_a_failure_whatever_the_tool_said() -> None:
    body = FakeBody(tools=_tools())  # dig_vein says "+5 oak_log" but the fake bag never changes
    brain = PolicyBrain(ScriptedDecision(["fell_tree_0"]), body, "logs")  # type: ignore[arg-type]
    d = await brain.step()
    assert d.ok, "the transport said ok…"
    assert d.verdict == "no_effect" and "gained nothing" in d.check, "…the world says nothing happened"
    assert d.reason == "nothing reached the bag"


def test_wait_spends_game_time_so_a_night_can_pass() -> None:
    w = next(a for a in candidates(ob()) if a.id == "wait")
    assert w.wait_s > 0 and w.steps == (("stop_moving", {}),)


async def test_a_failed_decision_is_recorded_not_raised() -> None:
    brain = PolicyBrain(ScriptedDecision([RuntimeError("503")]), FakeBody(tools=_tools()), "g")  # type: ignore[arg-type]
    d = await brain.step()
    assert d.action == "error" and not d.ok and "503" in (d.error or "")


async def test_nothing_is_ever_labelled_ok_to_the_model() -> None:
    """v1/v2 told the model ``build_shelter: ok — placed 0/25``. v3 shows only world verdicts."""
    import json

    body = FakeBody(tools=_tools())
    model = ScriptedDecision(["wait", "fell_tree_0", "wait"])
    brain = PolicyBrain(model, body, "g")  # type: ignore[arg-type]
    for _ in range(3):
        await brain.step()
    seen = json.dumps(model.requests[2][0])
    assert '"ok"' not in seen and ": ok" not in seen
    assert [a["verdict"] for a in model.requests[2][0]["recent_actions"]] == ["waited", "no_effect"]


def test_every_resource_carries_position_and_reachability() -> None:
    s = decision_state("g", ob(trees=TREE, stone=[{**STONE[0], "exposed": False}]), [])
    assert s["nearby"]["trees"] == ["oak_log x5 at (5, 64, 3) 6.1m exposed"]
    assert s["nearby"]["stone"] == ["stone x30 at (2, 62, 1) 3.0m buried"]


def test_the_tools_next_step_hint_leads_the_outcome_so_truncation_keeps_it() -> None:
    raw = "Dug 12 block(s) of stone (hit the 12-block cap (vein continues — call again to keep going)). Walked over 0 drop(s)."
    assert clean_outcome(raw).startswith("(vein continues — call again to keep going)")


async def test_body_events_and_the_steps_change_reach_the_model() -> None:
    from mcbrain.sidecar import SidecarEvent

    def dig(b: FakeBody, **_: object) -> str:
        b._state["inv"] = {"oak_log": 5}
        b._state["health"] = 17
        b.queue += [
            SidecarEvent(1, 0.0, "note", "radar: zombie inside 8 blocks — 5 away (3, 64, 1)"),
            SidecarEvent(2, 0.0, "note", "radar: zombie inside 8 blocks — 5 away (3, 64, 1)"),
            SidecarEvent(3, 0.0, "chat", "hi"),
        ]
        return "Dug 5 block(s) of oak_log."

    body = FakeBody(tools={**_tools(), "dig_vein": dig})
    model = ScriptedDecision(["fell_tree_0", "wait"])
    brain = PolicyBrain(model, body, "logs")  # type: ignore[arg-type]
    await brain.step()
    await brain.step()
    line = model.requests[1][0]["recent_actions"][-1]
    assert line["events"] == ["radar: zombie inside 8 blocks — 5 away ×2"], "deduped, coords stripped, chat excluded"
    assert line["change"]["health"] == -3 and line["change"]["bag"] == {"oak_log": "+5"}
    assert "events_since_last_step" not in model.requests[1][0], "nothing is shown twice (and no empty list)"
    assert model.requests[1][0]["budget"]["deaths"] == 0


def test_advancements_are_progress_events_and_accumulate_untrimmed() -> None:
    from mcbrain.policy import advancement

    assert advancement("(server message, not a player) StrandsBot has made the advancement [Stone Age]") == "Stone Age"
    steps = _steps(40, "y" * 240)
    steps[2].events = ["advancement: Getting Wood"]
    steps[30].events = ["advancement: Stone Age"]
    s = decision_state("g", ob(), steps, max_chars=6000, max_steps=12)
    assert s["advancements"] == ["Getting Wood", "Stone Age"], "kept even when their steps are trimmed away"


def test_a_multi_tool_step_leads_with_its_latest_useful_words() -> None:
    from mcbrain.policy import step_outcome

    texts = [
        "Equipped wooden_pickaxe to hand, confirmed in the slot.",
        "Dug leaf_litter at (8, 63, 71). Inventory change: nothing. The drop is on the ground (or there was none).",
        "Dug grass_block at (8, 62, 71). Inventory change: +3 leaf_litter.",
        "Dug stone at (8, 58, 71). Inventory change: +1 cobblestone.",
        "No dropped items nearby.",
    ]
    out = step_outcome(texts)
    assert out.startswith("Dug stone at (8, 58, 71). Inventory change: +1 cobblestone.")
    assert "Inventory change: nothing" not in out and "No dropped items" not in out and "Equipped" not in out


def test_clm_gets_less_text_of_each_kind_but_the_same_facts() -> None:
    """Operator 2026-09-28: CLM's context is trimmed (smoke: 5 kB state tripled its latency)."""
    import json

    from mcbrain.models import context_profile, resolve_brain_spec

    clm = context_profile(resolve_brain_spec({"STRANDS_MODEL_PROVIDER": "clm-sagemaker"}))
    jev = context_profile(resolve_brain_spec({"STRANDS_MODEL_PROVIDER": "typesafe"}))
    kev = context_profile(resolve_brain_spec({"STRANDS_MODEL_PROVIDER": "kev-sagemaker"}))
    assert clm.state_chars < kev.state_chars < jev.state_chars
    assert clm.history_steps < kev.history_steps < jev.history_steps
    assert not clm.previous_states and kev.previous_states and jev.previous_states
    steps = _steps(20, "y" * 240)
    steps[-1].events = ["hurt: (reflex) You just TOOK DAMAGE from ZOMBIE — health 9/20, food 18/20. The attacker is identified"]
    s = decision_state("g", ob(), steps, profile=clm)
    assert "previous_states" not in s and len(s["recent_actions"]) <= clm.history_steps
    assert len(s["recent_actions"][-1]["events"][0]) <= clm.event_chars
    assert "TOOK DAMAGE from ZOMBIE" in s["recent_actions"][-1]["events"][0], "the fact survives the trim"
    assert s["action_tally"]["fell_tree_0"] == "20 tried, 0 done, 20 no effect", "the tally is never trimmed"
    assert len(json.dumps(s)) < 3000


def test_kev_budget_fits_its_row_at_the_measured_worst_chars_per_token() -> None:
    """v3 run 1: 24k chars assumed 4 chars/token; Kev's tokenizer measured 2.11–2.74 → HTTP 422."""
    from strands.models.typesafe import KEV_MAX_STATE_PLUS_QUESTION_TOKENS

    from mcbrain.models import context_profile, resolve_brain_spec

    kev = context_profile(resolve_brain_spec({"STRANDS_MODEL_PROVIDER": "kev-sagemaker"}))
    worst_chars_per_token, question_tokens = 2.11, 1000
    assert kev.state_chars / worst_chars_per_token + question_tokens <= KEV_MAX_STATE_PLUS_QUESTION_TOKENS


def test_the_model_never_sees_an_empty_container() -> None:
    """v3 run 2: step-1 state full of {} / [] flipped CLM to `wait` in 9/9 episodes; without them it acted 9/9."""
    s = decision_state("Craft a crafting table and a wooden pickaxe.", ob(trees=TREE), [])
    empties = [k for k, v in s.items() if isinstance(v, (list, dict)) and not v]
    empties += [f"nearby.{k}" for k, v in s["nearby"].items() if isinstance(v, (list, dict)) and not v]
    assert not empties, empties
    assert "recent_actions" not in s and "action_tally" not in s, "absent on step 1, not empty"
    assert s["nearby"]["trees"], "non-empty facts stay"


def test_hobson_state_fits_its_served_window_at_the_worst_measured_chars_per_token() -> None:
    """hobson truncates an over-long state silently, so the budget must fit (4,096 served tokens)."""
    from mcbrain.models import context_profile, resolve_brain_spec

    hob = context_profile(resolve_brain_spec({"STRANDS_MODEL_PROVIDER": "hobson-sagemaker"}))
    assert hob.state_chars / 2.11 + 500 <= 4096
