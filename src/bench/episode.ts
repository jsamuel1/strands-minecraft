/**
 * One episode's record — field names and meaning match v3's `Episode`
 * (brain/src/mcbench/runner.py) so the two harnesses' JSONL can be read by the
 * same report. New fields are additive; `harness` says which harness wrote it.
 */
import type { Decision } from './policy.js';

export const HARNESS = 'sdk-native-v1';

export interface Episode {
  run_id: string;
  arm: string;
  kind: 'policy' | 'agent';
  model_label: string;
  task: string;
  episode: number;
  seed: string;
  mc_version: string;
  started_at: string;
  success: boolean;
  progress: number;
  time_to_goal_s: number | null;
  wall_s: number;
  steps: number;
  model_calls: number;
  tool_calls: number;
  tool_errors: number;
  deaths: number;
  tokens_in: number;
  tokens_out: number;
  cache_read_tokens: number;
  cache_write_tokens: number;
  cost_usd: number | null;
  latency_p50_ms: number | null;
  latency_p95_ms: number | null;
  server_latency_p50_ms: number | null;
  provider_errors: Record<string, number>;
  error_samples: Record<string, string>;
  instance_type: string | null;
  throttles: number;
  answering_models: string[];
  final_inventory: Record<string, number>;
  actions: string[];
  decisions: Array<Record<string, unknown>>;
  error: string | null;
  /** "steps" | "wall" | null (succeeded, or infra error). */
  cap_hit: 'steps' | 'wall' | null;
  loop_s: number | null;
  decide_s: number | null;
  call_latencies_ms: number[];
  state_chars_budget: number | null;
  history_steps_budget: number | null;
  context_profile: Record<string, unknown>;
  harness: string;
  brain_commit: string | null;
  /** v3 recorded a strands-py fork commit; this harness records the released SDK version. */
  strands_sdk: string;
  /** criteria (descriptions on the wire) | question_text (folded into the question). */
  option_rendering: string | null;
  /** Wall time from episode start to the brain's first step (world + body bootstrap). */
  setup_s: number | null;
}

export function newEpisode(p: Pick<Episode, 'run_id' | 'arm' | 'kind' | 'model_label' | 'task' | 'episode' | 'seed' | 'mc_version' | 'brain_commit' | 'strands_sdk'>): Episode {
  return {
    ...p,
    started_at: new Date().toISOString().replace(/\.\d+Z$/, 'Z'),
    success: false, progress: 0, time_to_goal_s: null, wall_s: 0, steps: 0, model_calls: 0, tool_calls: 0, tool_errors: 0,
    deaths: 0, tokens_in: 0, tokens_out: 0, cache_read_tokens: 0, cache_write_tokens: 0, cost_usd: null,
    latency_p50_ms: null, latency_p95_ms: null, server_latency_p50_ms: null, provider_errors: {}, error_samples: {},
    instance_type: null, throttles: 0, answering_models: [], final_inventory: {}, actions: [], decisions: [], error: null,
    cap_hit: null, loop_s: null, decide_s: null, call_latencies_ms: [], state_chars_budget: null, history_steps_budget: null,
    context_profile: {}, harness: HARNESS, option_rendering: null, setup_s: null,
  };
}

/** Python `statistics.quantiles(xs, n=100, method="inclusive")[q-1]`, rounded to 0.1 — v3's _pct. */
export function pct(xs: number[], q: number): number | null {
  if (!xs.length) return null;
  if (xs.length === 1) return Math.round(xs[0] * 10) / 10;
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length - 1;
  const pos = (q / 100) * m;
  const lo = Math.floor(pos);
  const hi = Math.min(m, lo + 1);
  return Math.round((s[lo] + (s[hi] - s[lo]) * (pos - lo)) * 10) / 10;
}

/** The per-decision row v3 stored, keyed identically. */
export function decisionRow(d: Decision): Record<string, unknown> {
  return {
    step: d.step, action: d.action, verdict: d.verdict, check: d.check, reason: d.reason, transport_ok: d.transportOk,
    last_step_p: d.lastStepP, conf: d.confidence, top: d.top, ms: d.latencyMs, server_ms: d.serverLatencyMs,
    options: d.options, state_chars: d.stateChars, outcome: (d.outcome || d.error || '').slice(0, 400), request: d.request,
    prob_sum: d.probabilitySum,
  };
}
