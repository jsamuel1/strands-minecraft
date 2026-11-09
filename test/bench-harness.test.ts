/**
 * The benchmark's policy, verifier and episode loops over a scripted body —
 * world-verified verdicts, legal-only candidates, the v3 state projection, and
 * equal budgets — with a fake native DecisionModel.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdirSync, mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { DecisionModel, Model, ModelError, type DecisionCallOptions, type DecisionInput, type DecisionResult, type Question } from '@strands-agents/sdk';
import type { Body, BodyEvent, BodySnapshot, ToolOutcome, Vec } from '../src/bench/body.js';
import { toOutcome } from '../src/bench/body.js';
import { candidates, decisionState, dropEmpty, buildQuestions, PolicyBrain, PROFILES, pyJson, stepOutcome, cleanOutcome, formatEvents, boxPlan, type Observation, type Step } from '../src/bench/policy.js';
import { checkAction, toolReason, type Snapshot } from '../src/bench/verify.js';
import { SUITE, TASKS, UNCAPPED_STEPS, applyMarks } from '../src/bench/tasks.js';
import { runPolicy, runAgent, buildAgent, type Clock } from '../src/bench/loops.js';
import { BENCH_BODY_ENV } from '../src/bench/runner.js';
import { newEpisode, pct } from '../src/bench/episode.js';
import { table } from '../src/bench/report.js';
import { resolveOut, createFreshRunOutput, parseArgs, buildRunManifest } from '../src/bench/cli.js';
import { dockerRunArgs } from '../src/bench/world.js';
import { SelfHostedDecisionModel } from '../src/bench/decider.js';

/** A scripted world: a bag, a position, trees that dig_vein turns into logs. */
class FakeBody implements Body {
  inv: Record<string, number> = {};
  pos: Vec = { x: 0.5, y: 64, z: 0.5 };
  trees = [{ position: { x: 3, y: 64, z: 1 }, distance: 3.2, veinSize: 5, exposed: true, block: 'oak_log' }];
  tables = 0;
  time = 1000;
  deaths = 0;
  calls: Array<[string, Record<string, unknown>]> = [];
  events: BodyEvent[] = [];
  busy: boolean[] = [];
  /** dig_vein reports success text but the world can be told to give nothing (a "transport ok, no effect" case). */
  digGives = 5;

  state(): BodySnapshot {
    return { pos: { ...this.pos }, health: 20, food: 20, inv: { ...this.inv }, foodPortions: 0, timeOfDay: this.time, isDay: this.time < 13000, threats: [], counters: { toolCalls: this.calls.length, toolErrors: 0, deaths: this.deaths, reconnects: 0 } };
  }
  async call(tool: string, args: Record<string, unknown> = {}): Promise<ToolOutcome> {
    this.calls.push([tool, args]);
    if (tool === 'find_blocks') {
      const name = String(args.name);
      if (name.includes('oak_log')) return toOutcome(this.trees);
      if (name === 'crafting_table') return toOutcome(Array.from({ length: this.tables }, () => ({ position: { x: 1, y: 64, z: 0 }, distance: 1, veinSize: 1, exposed: true, block: 'crafting_table' })));
      return toOutcome(`No '${name}' found within ${args.maxDistance} blocks.`);
    }
    if (tool === 'look_around') return toOutcome({ entities: [], standingOn: 'grass_block' });
    if (tool === 'dig_vein') {
      this.inv.oak_log = (this.inv.oak_log ?? 0) + this.digGives;
      return toOutcome(`Dug ${this.digGives} oak_log.`);
    }
    if (tool === 'collect_ground_items') return toOutcome('No dropped items nearby.');
    if (tool === 'go_to') { this.pos = { x: Number(args.x), y: Number(args.y), z: Number(args.z) }; return toOutcome('Arrived.'); }
    if (tool === 'stop_moving') return toOutcome('Stopped.');
    return { ok: false, text: `no fake for ${tool}`, data: null };
  }
  blocks(ps: Vec[]) { return ps.map((p) => ({ ...p, name: 'air', solid: false })); }
  drainEvents() { const e = this.events; this.events = []; return e; }
  setBusy(b: boolean) { this.busy.push(b); }
}

/** A native DecisionModel that answers from a script and records every request. */
class ScriptedModel extends DecisionModel {
  seen: Array<{ state: DecisionInput; questions: Record<string, Question> }> = [];
  constructor(private pick: (choices: readonly string[], n: number) => string) { super(); }
  protected async _ask<const Q extends Record<string, Question>>(state: DecisionInput, questions: Q, _o: DecisionCallOptions): Promise<DecisionResult<Q>> {
    this.seen.push({ state, questions });
    const choices = questions.next_action.choices as readonly string[];
    const c = this.pick(choices, this.seen.length);
    const answers: Record<string, unknown> = { next_action: c };
    const probabilities: Record<string, Record<string, number>> = { next_action: Object.fromEntries(choices.map((x) => [x, x === c ? 0.9 : 0.1 / (choices.length - 1)])) };
    if (questions.last_step_succeeded) { answers.last_step_succeeded = true; probabilities.last_step_succeeded = { true: 0.7, false: 0.3 }; }
    return { answers: answers as any, usage: { inputTokens: 10, outputTokens: 1, totalTokens: 11 }, metadata: { latencyMs: 5, modelId: 'scripted', probabilities: probabilities as any, confidence: { next_action: 0.8 } as any } };
  }
}

const fakeClock = (): Clock & { t: number } => ({ t: 0, now() { return this.t; } });
const noWait = async () => {};

function obs(body: FakeBody, extra: Partial<Observation> = {}): Observation {
  return { state: body.state(), trees: body.trees, stone: [], coal: [], iron: [], tables: [], furnaces: [], entities: [], drops: 0, standingOn: 'grass_block', events: [], ...extra };
}

const snap = (o: Partial<Snapshot> = {}): Snapshot => ({ inv: {}, pos: { x: 0, y: 64, z: 0 }, food: 20, tablesNear: 0, hostiles: {}, solid: new Set(), t: 0, ...o });

// ── verdicts come from the world ─────────────────────────────────────────────

test('verdict: a tree fell only if logs reached the bag — the tool saying "Dug" is not enough', () => {
  const meta = { gain: ['oak_log'] };
  assert.equal(checkAction('fell_tree_0', meta, snap(), snap({ inv: { oak_log: 5 } }), 'Dug 5').verdict, 'done');
  const v = checkAction('fell_tree_0', meta, snap(), snap({ inv: { dirt: 2 } }), 'Dug 0 oak_log. COULD NOT REACH (3, 64, 1)');
  assert.equal(v.verdict, 'no_effect');
  assert.equal(v.reason, 'COULD NOT REACH', 'the tool\'s own words, coordinates stripped');
});

test('verdict: shelter sealed = every plan position solid, whatever fills it; partial is in_progress with ETA', () => {
  const plan = boxPlan({ x: 0, y: 64, z: 0 });
  assert.equal(plan.length, 25, '3×3×3 shell: 8 wall positions × 3 layers + the roof centre (v3 "placed 0/25")');
  const k = (p: Vec) => `${p.x},${p.y},${p.z}`;
  const all = new Set(plan.map(k));
  const half = new Set(plan.slice(0, 8).map(k));
  assert.equal(checkAction('build_shelter', { plan }, snap(), snap({ solid: all })).verdict, 'done');
  const v = checkAction('build_shelter', { plan }, snap(), snap({ solid: half }), 'placed 8/25 … is occupied by stone (1, 64, 0)', 2);
  assert.equal(v.verdict, 'in_progress');
  assert.match(v.check, /8\/25 .*17 open, ~34s to finish/);
  assert.equal(v.reason, 'is occupied by stone');
});

test('verdict: craft, table, explore, eat, wait and unknown actions', () => {
  assert.equal(checkAction('craft_wooden_pickaxe', {}, snap(), snap({ inv: { wooden_pickaxe: 1 } })).verdict, 'done');
  assert.equal(checkAction('craft_wooden_pickaxe', {}, snap(), snap(), 'Missing materials: 2 sticks').reason, 'Missing materials: 2 sticks');
  assert.equal(checkAction('place_crafting_table', {}, snap({ tablesNear: 0 }), snap({ tablesNear: 1 })).verdict, 'done');
  assert.equal(checkAction('explore_north', { minMove: 8 }, snap(), snap({ pos: { x: 0, y: 64, z: -5 } })).verdict, 'no_effect');
  assert.equal(checkAction('eat', {}, snap({ food: 10 }), snap({ food: 14 })).verdict, 'done');
  assert.equal(checkAction('wait', {}, snap(), snap()).verdict, 'waited');
  assert.equal(checkAction('dance', {}, snap(), snap()).verdict, 'unverified');
  assert.equal(toolReason('all fine'), '');
});

// ── candidates are legal and grounded ────────────────────────────────────────

test('candidates: only grounded actions; no mining without a pickaxe; explore+wait always survive the cap', () => {
  const b = new FakeBody();
  const ids = candidates(obs(b)).map((a) => a.id);
  assert.deepEqual(ids, ['fell_tree_0', 'explore_north', 'explore_east', 'wait']);
  b.inv = { oak_log: 2 };
  const withLogs = candidates(obs(b)).map((a) => a.id);
  assert.ok(withLogs.includes('craft_crafting_table'), '2 logs → 8 planks → a table is affordable');
  assert.ok(!withLogs.includes('craft_wooden_pickaxe'), '3×3 recipes need a table in reach');
  assert.ok(!withLogs.some((i) => i.startsWith('mine_')));
  const many = obs(b, { trees: Array.from({ length: 20 }, () => b.trees[0]) });
  assert.ok(candidates(many).length <= 12);
  assert.deepEqual(candidates(many).slice(-3).map((a) => a.id), ['explore_north', 'explore_east', 'wait']);
});

// ── the state the model reads ────────────────────────────────────────────────

test('state: no empty containers on step 1 (v3 empties ablation), budget/tally kept, step_cap null when uncapped', () => {
  const b = new FakeBody();
  const s = decisionState('Collect 16 logs', obs(b), [], { steps_used: 0, step_cap: null }, PROFILES.decider);
  for (const k of ['action_tally', 'advancements', 'recent_actions', 'events_since_last_step', 'threats', 'previous_states']) assert.ok(!(k in s), `${k} absent when empty`);
  assert.deepEqual(Object.keys(s.nearby as object), ['trees', 'dropped_items_within_8']);
  assert.equal((s.budget as any).step_cap, null);
  assert.deepEqual(dropEmpty({ a: [], b: {}, c: 0, nearby: { x: [], y: [1] } }), { c: 0, nearby: { y: [1] } });
});

test('state: recent_actions and previous_states stay aligned and trim oldest-first under the char budget', () => {
  const b = new FakeBody();
  const facts = { health: 20, food: 20, day: true, timeOfDay: 1000, position: { x: 0, y: 64, z: 0 }, standing_on: 'grass', in_water: false, inventory: {} };
  const steps: Step[] = Array.from({ length: 30 }, (_, i) => ({ step: i + 1, action: 'wait', verdict: 'waited', check: 'wait does nothing by design', outcome: 'Stopped.', before: { ...facts, timeOfDay: 1000 + i }, reason: '', change: { seconds: 5 }, events: [] }));
  const prof = { ...PROFILES.decider, stateChars: 3000 };
  const s = decisionState('g', obs(b), steps, {}, prof);
  const ra = s.recent_actions as Array<{ step: number }>;
  const ps = s.previous_states as Array<{ timeOfDay: number }>;
  assert.ok(pyJson(s).length <= 3000);
  assert.equal(ra.length, ps.length);
  assert.equal(ra.at(-1)!.step, 30, 'newest kept');
  assert.equal(ps[0].timeOfDay, 1000 + ra[0].step - 1, 'same step on both lists');
  assert.equal((s.action_tally as any).wait, '30 tried, 0 done, 0 no effect', 'tally covers the whole episode');
});

test('pyJson matches Python json.dumps separators (the budget unit v3 used)', () => {
  assert.equal(pyJson({ a: [1, 'x'], b: { c: null }, d: 1.5, e: true }), '{"a": [1, "x"], "b": {"c": null}, "d": 1.5, "e": true}');
});

test('outcome cleaning: boilerplate stripped, hint first, empty tool lines dropped, newest first', () => {
  assert.equal(cleanOutcome('Equipped stone_pickaxe to hand, confirmed in the slot. | Dug 3. (vein continues — call again to keep going)'),
    '(vein continues — call again to keep going) | Dug 3. (vein continues — call again to keep going)');
  assert.equal(stepOutcome(['Inventory change: nothing.', 'Dug 1 stone.', 'No dropped items nearby.']), 'Dug 1 stone.');
  assert.deepEqual(formatEvents([
    { seq: 1, kind: 'note', text: 'radar: zombie at (1, 2, 3) inside 8' }, { seq: 2, kind: 'note', text: 'radar: zombie at (1, 2, 3) inside 8' },
    { seq: 3, kind: 'chat', text: 'hi' }, { seq: 4, kind: 'note', text: 'StrandsBot has made the advancement [Stone Age]' },
  ], 120, 6), ['radar: zombie at inside 8 ×2', 'advancement: Stone Age'], 'coords stripped exactly as v3\'s _COORDS.sub did');
});

test('questions: descriptions on the wire for a self-hosted decider, folded into text for SDK providers', () => {
  const acts = candidates(obs(new FakeBody()));
  const wire = buildQuestions(acts, true, 'criteria');
  assert.deepEqual(wire.next_action.choices, acts.map((a) => a.id));
  assert.match(wire.next_action.descriptions!.fell_tree_0, /Fell the oak_log tree \(5 logs\) at \(3, 64, 1\)/);
  assert.equal(wire.last_step_succeeded.choices, 'boolean');
  const text = buildQuestions(acts, false, 'question_text');
  assert.match(text.next_action.question, /- wait: Do nothing this step/);
  assert.equal(text.next_action.descriptions, undefined);
  assert.equal(text.last_step_succeeded, undefined, 'no last-step question on step 1');
  assert.equal(new PolicyBrain(new SelfHostedDecisionModel({ baseUrl: 'http://h' }), new FakeBody(), 'g', { profile: PROFILES.decider }).rendering, 'criteria');
});

// ── the episode loop ─────────────────────────────────────────────────────────

test('policy episode: success is the world goal check, not the model; decisions recorded like v3', async () => {
  const b = new FakeBody();
  const model = new ScriptedModel((c) => (c.includes('fell_tree_0') ? 'fell_tree_0' : 'wait'));
  const clock = fakeClock();
  const ep = newEpisode({ run_id: 'r', arm: 'fake', kind: 'policy', model_label: 'scripted', task: 'logs16', episode: 1, seed: '1', mc_version: 'x', brain_commit: null, strands_sdk: '1.24.0' });
  await runPolicy(ep, model, PROFILES.decider, TASKS.logs16, b, {}, clock, { wait: noWait });
  assert.equal(ep.success, true);
  assert.equal(ep.steps, 4, '4 × 5 logs ≥ 16');
  assert.deepEqual(ep.final_inventory, {}, 'the runner, not the loop, fills final_inventory');
  assert.deepEqual(ep.actions, ['fell_tree_0:done', 'fell_tree_0:done', 'fell_tree_0:done', 'fell_tree_0:done']);
  assert.equal(ep.cap_hit, null);
  assert.equal(ep.option_rendering, 'question_text');
  assert.equal(ep.decisions.length, 4);
  assert.equal(ep.decisions[1].last_step_p, 0.7, 'the model\'s own last-step read is recorded beside the world verdict');
  assert.equal(ep.decisions[0].last_step_p, null);
  assert.equal(ep.tokens_in, 40);
  assert.ok(b.busy.includes(true) && b.busy.at(-1) === false, 'busy is set around execution and cleared');
  assert.equal(typeof model.seen[0].state, 'string');
  const sent = JSON.parse(model.seen[0].state as string);
  assert.equal(sent.goal, TASKS.logs16.goal);
  assert.equal(sent.budget.step_cap, 20);
});

test('policy episode: a model that claims success while the world shows none does not succeed (step cap binds)', async () => {
  const b = new FakeBody();
  b.digGives = 0;
  const model = new ScriptedModel(() => 'fell_tree_0');
  const ep = newEpisode({ run_id: 'r', arm: 'fake', kind: 'policy', model_label: 's', task: 'logs16', episode: 1, seed: '1', mc_version: 'x', brain_commit: null, strands_sdk: '1.24.0' });
  await runPolicy(ep, model, PROFILES.decider, TASKS.logs16, b, {}, fakeClock(), { wait: noWait });
  assert.equal(ep.success, false);
  assert.equal(ep.steps, TASKS.logs16.maxSteps);
  assert.equal(ep.cap_hit, 'steps');
  assert.ok(ep.decisions.every((d) => d.verdict === 'no_effect' && d.transport_ok === true), 'transport ok, world says no effect');
  const later = JSON.parse(model.seen[1].state as string);
  assert.equal(later.recent_actions[0].verdict, 'no_effect', 'the model is told the truth, never "ok"');
  assert.equal(later.action_tally.fell_tree_0, '1 tried, 0 done, 1 no effect');
});

test('policy episode: the wall-clock cap binds before the next decision', async () => {
  const b = new FakeBody();
  b.digGives = 0;
  const clock = fakeClock();
  const model = new ScriptedModel(() => { clock.t += 100; return 'fell_tree_0'; });
  const ep = newEpisode({ run_id: 'r', arm: 'fake', kind: 'policy', model_label: 's', task: 'logs16', episode: 1, seed: '1', mc_version: 'x', brain_commit: null, strands_sdk: '1.24.0' });
  await runPolicy(ep, model, PROFILES.decider, TASKS.logs16, b, {}, clock, { wait: noWait });
  assert.equal(ep.steps, 5, '420 s cap: steps start at 0,100,200,300,400; 500 ≥ 420 stops');
  assert.equal(ep.cap_hit, 'wall');
});

test('policy episode: each native decision receives the remaining wall budget as an abort signal', async () => {
  const b = new FakeBody();
  b.digGives = 0;
  const clock = fakeClock();
  class SignalModel extends DecisionModel {
    signals: AbortSignal[] = [];
    protected async _ask<const Q extends Record<string, Question>>(_state: DecisionInput, questions: Q, o: DecisionCallOptions): Promise<DecisionResult<Q>> {
      assert.ok(o.cancelSignal);
      this.signals.push(o.cancelSignal!);
      clock.t += 1;
      const choices = questions.next_action.choices as readonly string[];
      const pick = choices.includes('fell_tree_0') ? 'fell_tree_0' : choices[0];
      return { answers: { next_action: pick } as any, metadata: { latencyMs: 1 } };
    }
  }
  const model = new SignalModel();
  const ep = newEpisode({ run_id: 'r', arm: 'fake', kind: 'policy', model_label: 's', task: 'logs16', episode: 1, seed: '1', mc_version: 'x', brain_commit: null, strands_sdk: '1.24.0' });
  await runPolicy(ep, model, PROFILES.decider, { ...TASKS.logs16, maxSteps: 2, maxWallS: 2 }, b, {}, clock, { wait: noWait });
  assert.equal(model.signals.length, 2);
  assert.ok(model.signals.every((s) => !s.aborted));
  assert.equal(ep.cap_hit, 'steps');
});

test('policy episode: a terminal decision-model failure is recorded once without consuming an embodied step', async () => {
  class Broken extends DecisionModel { protected async _ask(): Promise<never> { const e = new Error('HTTP 503'); e.name = 'ModelThrottledError'; throw e; } }
  const b = new FakeBody();
  const ep = newEpisode({ run_id: 'r', arm: 'fake', kind: 'policy', model_label: 's', task: 'wood_tools', episode: 1, seed: '1', mc_version: 'x', brain_commit: null, strands_sdk: '1.24.0' });
  await runPolicy(ep, new Broken(), PROFILES.decider, { ...TASKS.wood_tools, maxSteps: 2 }, b, {}, fakeClock(), { wait: noWait });
  assert.equal(ep.steps, 0, 'provider failed before any embodied action');
  assert.equal(ep.model_calls, 1, 'terminal provider failure does not retry until the action cap');
  assert.equal(ep.provider_errors.ModelThrottledError, 1);
  assert.equal(ep.throttles, 1);
  assert.equal(ep.latency_p50_ms, null, 'failed calls are not latency samples');
  assert.equal(ep.cap_hit, null, 'provider failure is not a step or wall cap');
  assert.ok(!b.calls.some(([t]) => t === 'dig_vein'));
  const b2 = new FakeBody();
  const brain = new PolicyBrain(new ScriptedModel(() => 'teleport_home'), b2, 'g', { profile: PROFILES.decider, wait: noWait });
  // the SDK base validates questions, not answers; an implementation returning an id outside choices is caught here
  const d = await brain.step();
  assert.equal(d.verdict, 'no_effect');
  assert.match(d.error!, /not one of the offered ids/);
});

test('agent episode: a terminal model failure is recorded once and stops before burning the step budget', async () => {
  class BrokenModel extends Model {
    updateConfig() {}
    getConfig() { return { modelId: 'broken' }; }
    async *stream(): AsyncIterable<never> { throw new ModelError('provider unavailable'); }
  }
  const b = new FakeBody();
  const ep = newEpisode({ run_id: 'r', arm: 'agent', kind: 'agent', model_label: 'broken', task: 'logs16', episode: 1, seed: '1', mc_version: 'x', brain_commit: null, strands_sdk: '1.24.0' });
  await runAgent(ep, buildAgent(new BrokenModel(), [], 'test'), { ...TASKS.logs16, maxSteps: 4 }, b, {}, fakeClock());
  assert.equal(ep.steps, 0, 'provider failed before any embodied tool action');
  assert.equal(ep.model_calls, 1);
  assert.equal(ep.provider_errors.ModelError, 1);
  assert.match(ep.error_samples.ModelError, /provider unavailable/);
  assert.equal(ep.cap_hit, null, 'terminal invocation failure is not mislabelled as budget exhaustion');
});

// ── tasks, marks, budgets, report, guards ────────────────────────────────────

test('task suite is v3\'s: ids, step caps and wall caps', () => {
  assert.deepEqual(SUITE.map((t) => [t.id, t.maxSteps, t.maxWallS]), [['logs16', 20, 420], ['wood_tools', 24, 480], ['cobble8', 32, 600], ['iron1', 48, 900], ['night', UNCAPPED_STEPS, 660]]);
  assert.deepEqual(TASKS.night.setup, ['time set 12500', 'difficulty easy']);
  const m = {};
  assert.equal(applyMarks({ inv: { wooden_pickaxe: 1 }, timeOfDay: 1000, counters: { deaths: 0 } }, m, 1)._placedTable, false, 'a table present at the start does not count');
  assert.equal(applyMarks({ inv: { wooden_pickaxe: 1 }, timeOfDay: 1000, counters: { deaths: 0 } }, m, 2)._placedTable, true);
  assert.deepEqual(TASKS.wood_tools.check(applyMarks({ inv: { wooden_pickaxe: 1 }, timeOfDay: 1000, counters: { deaths: 0 } }, m, 2)), [true, 1]);
  const n = {};
  applyMarks({ inv: {}, timeOfDay: 14000, counters: { deaths: 0 } }, n, 0);
  assert.equal(TASKS.night.check(applyMarks({ inv: {}, timeOfDay: 500, counters: { deaths: 0 } }, n, 0))[0], true);
  assert.equal(TASKS.night.check(applyMarks({ inv: {}, timeOfDay: 500, counters: { deaths: 1 } }, n, 0))[0], false);
  const longTurn = {};
  applyMarks({ inv: {}, timeOfDay: 12500, counters: { deaths: 0 } }, longTurn, 0);
  assert.equal(TASKS.night.check(applyMarks({ inv: {}, timeOfDay: 999, counters: { deaths: 0 } }, longTurn, 0))[0], true,
    'one long turn spanning dusk through clock wrap still proves night passed');
  const backwardsJitter = {};
  applyMarks({ inv: {}, timeOfDay: 5000, counters: { deaths: 0 } }, backwardsJitter, 0);
  assert.equal(TASKS.night.check(applyMarks({ inv: {}, timeOfDay: 4999, counters: { deaths: 0 } }, backwardsJitter, 0))[0], false,
    'a small backwards clock adjustment is not a night wrap');
});

test('reset is reproducible: fixed image tag, version, seed, tmpfs world, localhost-only ports', () => {
  const a = dockerRunArgs('w', 25600, 25601, { image: 'itzg/minecraft-server:latest', version: '1.21.11', seed: '8675309' });
  assert.ok(a.includes('SEED=8675309') && a.includes('VERSION=1.21.11') && a.includes('DIFFICULTY=easy'));
  assert.ok(a.includes('127.0.0.1:25600:25565'));
  assert.deepEqual(a.slice(a.indexOf('--tmpfs'), a.indexOf('--tmpfs') + 2), ['--tmpfs', '/data:size=3g']);
  assert.deepEqual(BENCH_BODY_ENV, {
    REFLEX_MODES_OFF: 'item_magnet,elbow_room,idle_staring',
    REFLEX_MODES_DISABLED: 'false', SENTINEL_DISABLED: 'false', REFLEX_DISABLED: 'false',
    THINKER_DISABLED: 'true', VOICE_DISABLED: 'true',
  }, 'deployment env cannot silently disable safety rails or re-enable autonomous movement for one cohort');
});

test('pct matches Python statistics.quantiles(method="inclusive")', () => {
  assert.equal(pct([], 50), null);
  assert.equal(pct([7], 95), 7);
  assert.equal(pct([1, 2, 3, 4], 50), 2.5);
  assert.equal(pct([10, 20, 30, 40, 50], 95), 48);
});

test('report: median (min–max), budget stars, infra kept out of fail', () => {
  const row = (o: Record<string, unknown>) => ({ _run: 'r', arm: 'decider', task: 'logs16', harness: 'sdk-native-v1', brain_commit: 'abc', strands_sdk: '1.24.0', success: true, error: null, model_calls: 4, decide_s: 1, loop_s: 50, cap_hit: null, ...o });
  const t = table([row({}), row({ success: false, model_calls: 20, loop_s: 100, cap_hit: 'steps' }), row({ error: 'docker: boom' })]);
  assert.match(t, /\| decider \| r \| logs16 \| 3 \| 1 \| 1\+1 infra \| 12 \(4–20\*\) \| 1\.0 \(1\.0–1\.0\) \| 75 \(50–100\) \| 1\.3% \|/);
  assert.match(t, /Harness: sdk-native-v1@abc sdk 1\.24\.0/);
  const sparse = table([{ _run: 'r', arm: 'decider', task: 'night', success: false, error: 'episode process exited SIGKILL without a row', harness: 'sdk-native-v1' }]);
  assert.match(sparse, /\| decider \| r \| night \| 1 \| 0 \| 0\+1 infra \| – \| – \| – \| – \|/, 'a lost episode is shown, never dropped');
});

test('results guard: new results only under bench-results/sdk-native; v3 results and existing run dirs are refused', () => {
  assert.match(resolveOut(undefined, 'x', '/repo'), /\/repo\/bench-results\/sdk-native\/x$/);
  assert.throws(() => resolveOut('bench-results/2026-09-29-v3-policy', 'x', '/repo'), /immutable/);
  assert.throws(() => resolveOut('/repo/bench-results', 'x', '/repo'), /immutable/);
  assert.throws(() => resolveOut('bench-results/RESULTS-v3.md', 'x', '/repo'), /immutable/);
  assert.doesNotThrow(() => resolveOut('/tmp/scratch-run', 'x', '/repo'));
  const root = mkdtempSync(join(tmpdir(), 'mcbench-output-'));
  const existing = join(root, 'existing');
  mkdirSync(existing);
  try {
    assert.throws(() => createFreshRunOutput(existing), /scored runs are append-immutable/);
    const fresh = join(root, 'missing-parent', 'new-run');
    assert.doesNotThrow(() => createFreshRunOutput(fresh), 'a clean checkout has no sdk-native parent yet');
    assert.throws(() => createFreshRunOutput(fresh), /scored runs are append-immutable/);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
  assert.deepEqual(parseArgs(['run', '--arms', 'decider', '--n', '2', 'extra']), { cmd: 'run', flags: { arms: 'decider', n: '2' }, rest: ['extra'] });
  assert.throws(() => parseArgs(['run', '--arms', '--tasks', 'logs16']), /--arms needs a value/);
});

test('run manifest records resolved world, budgets, policy profile and forced body configuration', () => {
  const m = buildRunManifest({ runId: 'r', arms: ['decider', 'agent'], tasks: ['logs16'], n: 2, startedAt: '2026-11-09T00:00:00Z', argv: ['run'] });
  assert.equal(m.harness, 'sdk-native-v1');
  assert.equal(m.world.seed, '8675309');
  assert.deepEqual(m.task_budgets.logs16, { steps: 20, wall_s: 420 });
  assert.equal(m.policy.profiles.decider.stateChars, PROFILES.decider.stateChars);
  assert.equal('agent' in m.policy.profiles, false);
  assert.deepEqual(m.body.env, BENCH_BODY_ENV);
  assert.equal(m.body.config.reflex.unstuckAfterMs, 20_000);
  assert.equal(m.body.config.combat.answerReach, 4);
});
