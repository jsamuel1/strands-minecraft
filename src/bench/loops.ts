/**
 * The episode loops — pure over a `Body`, a model, and a clock, so tests drive
 * them with fakes. The same budgets bind both arm kinds:
 *
 *   - `task.maxSteps` decisions (policy) or turns (agent), and
 *   - `task.maxWallS` seconds of brain loop time, checked before each step and
 *     (agent arm) enforced inside a turn by cancelling it at the deadline.
 *
 * Success is `task.check()` over the WORLD snapshot after every step — never
 * the model's claim. The agent arm gets `limits: { turns: 4 }` per step, as v3.
 */
import { Agent, AfterModelCallEvent, BeforeModelCallEvent, type DecisionModel, type Model, type ToolList } from '@strands-agents/sdk';
import type { Body } from './body.js';
import { applyMarks, UNCAPPED_STEPS, type BodyState, type Task } from './tasks.js';
import { PolicyBrain, type ContextProfile } from './policy.js';
import { decisionRow, pct, type Episode } from './episode.js';

export interface Clock { now(): number }
export const realClock: Clock = { now: () => performance.now() / 1000 };

export interface Marks { tablesAtStart?: number; placedTable?: boolean; sawNight?: boolean; passedNight?: boolean }

async function tablesNear(body: Body): Promise<number> {
  const o = await body.call('find_blocks', { name: 'crafting_table', maxDistance: 8, count: 4 });
  return Array.isArray(o.data) ? o.data.length : 0;
}

/** World truth for the goal check: body state plus the harness's own marks. */
export async function goalState(body: Body, marks: Marks): Promise<BodyState> {
  const s = body.state();
  return applyMarks({ inv: s.inv, timeOfDay: s.timeOfDay, counters: { deaths: s.counters.deaths } }, marks, await tablesNear(body));
}

const budgetFor = (task: Task, steps: number, used: number, progress: number) => ({
  steps_used: steps,
  // null = no step limit (night is wall-clock bound); never a sentinel number the model must decode.
  step_cap: task.maxSteps < UNCAPPED_STEPS ? task.maxSteps : null,
  seconds_used: Math.round(used),
  seconds_cap: task.maxWallS,
  progress,
});

export async function runPolicy(ep: Episode, model: DecisionModel, profile: ContextProfile, task: Task, body: Body, marks: Marks, clock: Clock = realClock, brainOpts: { wait?: (ms: number) => Promise<void> } = {}): Promise<void> {
  const brain = new PolicyBrain(model, body, task.goal, { profile, now: () => clock.now(), ...brainOpts });
  ep.state_chars_budget = profile.stateChars;
  ep.history_steps_budget = profile.historySteps;
  ep.context_profile = { ...profile };
  ep.option_rendering = brain.rendering;
  const lat: number[] = [];
  const srv: number[] = [];
  const t0 = clock.now();
  for (let i = 0; i < task.maxSteps; i++) {
    const used = clock.now() - t0;
    if (used > task.maxWallS) break;
    brain.budget = budgetFor(task, i, used, ep.progress);
    const d = await brain.step();
    ep.steps += 1;
    ep.model_calls += 1;
    if (d.action === 'error' && d.error) {
      const name = d.error.split(':')[0];
      ep.provider_errors[name] = (ep.provider_errors[name] ?? 0) + 1;
      ep.error_samples[name] ??= d.error.slice(0, 300);
      if (/Throttl|RateLimit/.test(d.error)) ep.throttles += 1;
    } else {
      lat.push(d.latencyMs);
      if (d.serverLatencyMs !== null) srv.push(d.serverLatencyMs);
      if (d.modelId && !ep.answering_models.includes(d.modelId)) ep.answering_models.push(d.modelId);
    }
    ep.tokens_in += d.usage.inputTokens ?? 0;
    ep.tokens_out += d.usage.outputTokens ?? 0;
    ep.actions.push(`${d.action}:${d.verdict}`);
    ep.decisions.push(decisionRow(d));
    const [done, progress] = task.check(await goalState(body, marks));
    ep.progress = progress;
    if (done) {
      ep.success = true;
      ep.time_to_goal_s = +(clock.now() - t0).toFixed(1);
      break;
    }
  }
  ep.loop_s = +(clock.now() - t0).toFixed(1);
  ep.decide_s = +(lat.reduce((a, b) => a + b, 0) / 1000).toFixed(2);
  ep.latency_p50_ms = pct(lat, 50);
  ep.latency_p95_ms = pct(lat, 95);
  ep.server_latency_p50_ms = pct(srv, 50);
  if (!ep.success) ep.cap_hit = ep.steps >= task.maxSteps ? 'steps' : 'wall';
}

/** Per-model-call latency and errors from the SDK's own hooks: exact, never estimated. */
export function instrumentAgent(agent: Agent): { latencies: number[]; errors: Record<string, number>; samples: Record<string, string> } {
  const out = { latencies: [] as number[], errors: {} as Record<string, number>, samples: {} as Record<string, string> };
  let started = 0;
  agent.addHook(BeforeModelCallEvent, () => { started = performance.now(); });
  agent.addHook(AfterModelCallEvent, (e) => {
    out.latencies.push(+(performance.now() - started).toFixed(1));
    if (e.error) {
      const n = e.error.name || 'Error';
      out.errors[n] = (out.errors[n] ?? 0) + 1;
      out.samples[n] ??= e.error.message.slice(0, 300);
    }
  });
  return out;
}

export const AGENT_FIRST_PROMPT = (goal: string) =>
  `Your goal: ${goal} This is a benchmark: the goal is verified from your inventory and body, so actually do it. ` +
  'Work step by step with your tools. Use the composite tools (find_blocks, dig_vein, craft_item). ' +
  'Do not start journeys or hire workers — just act.';

export async function runAgent(ep: Episode, agent: Agent, task: Task, body: Body & { digest?: () => string }, marks: Marks, clock: Clock = realClock): Promise<void> {
  const stats = instrumentAgent(agent);
  const t0 = clock.now();
  for (let i = 0; i < task.maxSteps; i++) {
    const used = clock.now() - t0;
    if (used > task.maxWallS) break;
    const prompt = i === 0 ? AGENT_FIRST_PROMPT(task.goal) : `Status:\n${body.digest?.() ?? ''}\nContinue toward the goal: ${task.goal}`;
    const remainingMs = Math.max(1000, (task.maxWallS - used) * 1000);
    body.setBusy(true);
    try {
      const r = await agent.invoke(prompt, { limits: { turns: 4 }, cancelSignal: AbortSignal.timeout(remainingMs) });
      const u = r.metrics?.latestAgentInvocation?.usage;
      if (u) {
        ep.tokens_in += u.inputTokens ?? 0;
        ep.tokens_out += u.outputTokens ?? 0;
        ep.cache_read_tokens += u.cacheReadInputTokens ?? 0;
        ep.cache_write_tokens += u.cacheWriteInputTokens ?? 0;
      }
      if (r.stopReason === 'cancelled') { ep.steps += 1; break; }
    } catch (err) {
      const n = err instanceof Error ? err.name : 'Error';
      ep.error_samples[n] ??= (err instanceof Error ? err.message : String(err)).slice(0, 300);
    } finally {
      body.setBusy(false);
    }
    ep.steps += 1;
    const [done, progress] = task.check(await goalState(body, marks));
    ep.progress = progress;
    if (done) {
      ep.success = true;
      ep.time_to_goal_s = +(clock.now() - t0).toFixed(1);
      break;
    }
  }
  ep.loop_s = +(clock.now() - t0).toFixed(1);
  if (!ep.success) ep.cap_hit = ep.steps >= task.maxSteps ? 'steps' : 'wall';
  ep.model_calls = stats.latencies.length;
  ep.call_latencies_ms = stats.latencies;
  ep.decide_s = +(stats.latencies.reduce((a, b) => a + b, 0) / 1000).toFixed(2);
  ep.latency_p50_ms = pct(stats.latencies, 50);
  ep.latency_p95_ms = pct(stats.latencies, 95);
  for (const [n, c] of Object.entries(stats.errors)) {
    ep.provider_errors[n] = (ep.provider_errors[n] ?? 0) + c;
    if (/Throttl/.test(n)) ep.throttles += c;
  }
  Object.assign(ep.error_samples, Object.fromEntries(Object.entries(stats.samples).filter(([k]) => !(k in ep.error_samples))));
}

/** The agent arm: an SDK Agent with the body's tools, the upstream system prompt, one model. */
export function buildAgent(model: Model, tools: unknown[], systemPrompt: string): Agent {
  return new Agent({ model, tools: tools as ToolList, systemPrompt, printer: false });
}
