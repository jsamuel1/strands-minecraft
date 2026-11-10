/**
 * 🧮 Decision models for the benchmark — the SDK's native `DecisionModel`, plus
 * one subclass for a self-hosted System One server.
 *
 * @strands-agents/sdk ≥ 1.21 ships `DecisionModel` (the base: validation,
 * `decision` span, usage) and, from 1.23, the System One providers
 * `TypeSafeDecisionModel` (api.typesafe.ai, Jev) and `DecisionsAPIDecisionModel`
 * (decisionapi.net). Their endpoints are constants: neither takes a base URL.
 * A strands-decider server (`strands-decider serve`, github.com/strands-labs/
 * strands-decider) speaks the same `POST /v1/systemone` wire, so the harness
 * reaches it with a small `DecisionModel` subclass. Everything above the wire —
 * `ask()`, question validation, the trace span, usage — is the SDK's.
 *
 * Why a subclass and not `TypeSafeDecisionModel({ fetch })` pointed elsewhere:
 *   1. The native `Question` carries `choices` as bare ids. The macro-action
 *      alphabet (policy.ts) is ids PLUS a description each ("Fell the oak_log
 *      tree (5 logs) at (12, 70, -3)…"), which is what v3 sent and what the
 *      decider renders as `1. fell_tree_0 — Fell the …`. An id alone loses the
 *      grounding. `BenchQuestion.descriptions` carries them on the wire as the
 *      choice `criteria`; the SDK sends `null` there.
 *   2. strands-decider rounds each probability to 4 dp, so a 12-option
 *      distribution can sum to 1 ± 6e-4; the SDK's parser rejects anything past
 *      1e-6 ("probabilities … must sum to 1"). This parser accepts ≤ 1e-3 and
 *      records the raw sum. Nothing is renormalised.
 *   3. The server's own `latency_ms` (forward-pass time) is kept beside the
 *      client round trip, as v3 recorded it.
 *
 * Everything else (wire field names, Uncertain below min confidence, HTTP
 * error classes) follows the SDK's system-one-http-adapter at 1.24.0. This
 * adapter also treats HTTP 503 as throttling, matching current SDK mainline;
 * released TS 1.24.0 itself maps only 429/529 that way.
 */
import {
  ContextWindowOverflowError,
  DecisionModel,
  LLMDecisionModel,
  ModelError,
  ModelThrottledError,
  TypeSafeDecisionModel,
  Uncertain,
  type DecisionCallOptions,
  type DecisionInput,
  type DecisionResult,
  type Model,
  type Question,
} from '@strands-agents/sdk';

/** A native Question, with an optional description per choice id (sent as `criteria`). */
export type BenchQuestion = Question & { descriptions?: Readonly<Record<string, string>> };

/** What the policy needs back from any decision model, beyond the SDK result. */
export interface DecisionExtras {
  /** Server-reported model latency, ms (self-hosted only). */
  serverLatencyMs?: number;
  /** Sum of the returned probabilities per choice question (self-hosted only). */
  probabilitySums?: Record<string, number>;
}

export interface SelfHostedDecisionModelOptions {
  /** Base URL of the server, e.g. http://127.0.0.1:8099. `/v1/systemone` is appended. */
  baseUrl: string;
  /** Sent as `model`; strands-decider ignores it and reports the checkpoint it loaded. */
  modelId?: string;
  /** Bearer token, when the server is behind one. Local `strands-decider serve` needs none. */
  apiKey?: string;
  /** Below this confidence the answer is Uncertain (SDK semantics). Default 0 here: the policy must act. */
  defaultMinConfidence?: number;
  /** Largest request body, bytes. Over it → ContextWindowOverflowError before sending. */
  maxBodyBytes?: number;
  /** Provider name recorded in metadata. */
  providerName?: string;
  fetch?: typeof globalThis.fetch;
  /** Abort a request after this many ms. */
  timeoutMs?: number;
}

/** Probabilities from the server are rounded (4 dp): tolerate the rounding, not more. */
export const PROBABILITY_SUM_TOLERANCE = 1e-3;
const MAX_CHOICES = 255;

/** SDK `Question` → System One wire question. Descriptions become `criteria` values. */
export function toWireQuestion(q: BenchQuestion): Record<string, unknown> {
  if (q.choices === 'boolean') return { type: 'noul', instructions: q.question };
  if (q.choices.length > MAX_CHOICES) throw new Error(`System One allows at most ${MAX_CHOICES} choices`);
  const criteria = Object.fromEntries(q.choices.map((c) => [c, q.descriptions?.[c] ?? null]));
  return { type: 'choice', instructions: q.question, criteria };
}

const isRecord = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null && !Array.isArray(v);

function prob(v: unknown, what: string): number {
  if (typeof v !== 'number' || !Number.isFinite(v) || v < 0 || v > 1) throw new ModelError(`${what} must be a number in [0, 1]`);
  return v;
}

/** Parse a `/v1/systemone` response into the SDK result shape. Pure; exported for tests. */
export function parseSystemOne<Q extends Record<string, BenchQuestion>>(
  payload: unknown,
  questions: Q,
  minConfidence: number,
  latencyMs: number,
  providerName: string,
): { result: DecisionResult<Q>; extras: DecisionExtras } {
  if (!isRecord(payload)) throw new ModelError('decision provider response must be an object');
  const modelId = payload.model;
  if (typeof modelId !== 'string' || !modelId.trim()) throw new ModelError('decision provider model must be a non-empty string');
  const raw = payload.answers;
  if (!isRecord(raw)) throw new ModelError('decision provider answers must be an object');
  const answers: Record<string, unknown> = {};
  const probabilities: Record<string, Record<string, number>> = {};
  const confidence: Record<string, number> = {};
  const sums: Record<string, number> = {};
  for (const [id, q] of Object.entries(questions)) {
    const a = raw[id];
    if (!isRecord(a)) throw new ModelError(`decision provider omitted answer for question id=<${id}>`);
    const min = q.uncertainty === false ? undefined
      : (isRecord(q.uncertainty) && typeof q.uncertainty.minConfidence === 'number') ? q.uncertainty.minConfidence : minConfidence;
    if (q.choices === 'boolean') {
      if (a.type !== 'noul') throw new ModelError(`answer for boolean question id=<${id}> must have type 'noul'`);
      const p = prob(a.noul, `noul for id=<${id}>`);
      const c = Math.abs(2 * p - 1);
      answers[id] = min !== undefined && c < min ? new Uncertain() : p >= 0.5;
      probabilities[id] = { true: p, false: 1 - p };
      confidence[id] = c;
      continue;
    }
    if (a.type !== 'choice') throw new ModelError(`answer for choice question id=<${id}> must have type 'choice'`);
    const choices = q.choices as readonly string[];
    if (typeof a.choice !== 'string' || !choices.includes(a.choice)) {
      throw new ModelError(`answer for choice question id=<${id}> selected an unknown choice`);
    }
    if (!isRecord(a.probabilities)) throw new ModelError(`probabilities for id=<${id}> must be an object`);
    const keys = Object.keys(a.probabilities);
    if (keys.length !== choices.length || choices.some((c) => !Object.hasOwn(a.probabilities as object, c))) {
      throw new ModelError(`probabilities for id=<${id}> must contain exactly the requested choices`);
    }
    const ps = Object.fromEntries(choices.map((c) => [c, prob((a.probabilities as Record<string, unknown>)[c], `p(${c})`)]));
    const sum = Object.values(ps).reduce((s, x) => s + x, 0);
    if (Math.abs(sum - 1) > PROBABILITY_SUM_TOLERANCE) throw new ModelError(`probabilities for id=<${id}> sum to ${sum}, not 1`);
    const top = Math.max(...Object.values(ps));
    if (ps[a.choice] + PROBABILITY_SUM_TOLERANCE < top) throw new ModelError(`choice for id=<${id}> must have the highest probability`);
    const conf = prob(a.confidence, `confidence for id=<${id}>`);
    answers[id] = min !== undefined && conf < min ? new Uncertain() : a.choice;
    probabilities[id] = ps;
    confidence[id] = conf;
    sums[id] = +sum.toFixed(6);
  }
  const u = payload.usage;
  const usage = isRecord(u) && typeof u.input_tokens === 'number' && typeof u.output_tokens === 'number'
    ? { inputTokens: u.input_tokens, outputTokens: u.output_tokens, totalTokens: u.input_tokens + u.output_tokens }
    : undefined;
  const serverLatencyMs = typeof payload.latency_ms === 'number' ? payload.latency_ms : undefined;
  return {
    result: {
      answers: answers as DecisionResult<Q>['answers'],
      ...(usage ? { usage } : {}),
      metadata: {
        latencyMs,
        providerName,
        modelId,
        probabilities: probabilities as DecisionResult<Q>['metadata']['probabilities'],
        confidence: confidence as DecisionResult<Q>['metadata']['confidence'],
      },
    },
    extras: { ...(serverLatencyMs !== undefined ? { serverLatencyMs } : {}), probabilitySums: sums },
  };
}

/**
 * A strands-decider (or any `/v1/systemone`) server as a native SDK DecisionModel.
 * `ask()` is the SDK's; this class only supplies `_ask`.
 */
export class SelfHostedDecisionModel extends DecisionModel {
  readonly endpoint: string;
  private readonly o: Required<Pick<SelfHostedDecisionModelOptions, 'modelId' | 'defaultMinConfidence' | 'maxBodyBytes' | 'providerName' | 'timeoutMs'>> & SelfHostedDecisionModelOptions;
  private readonly f: typeof globalThis.fetch;
  /** Extras from the most recent call (the SDK result shape has no slot for them). */
  last: DecisionExtras = {};

  constructor(opts: SelfHostedDecisionModelOptions) {
    super();
    if (!opts.baseUrl) throw new Error('SelfHostedDecisionModel needs a baseUrl');
    this.endpoint = `${opts.baseUrl.replace(/\/+$/, '')}/v1/systemone`;
    this.o = {
      modelId: 'strands-decider-latest', defaultMinConfidence: 0, maxBodyBytes: 4 * 1024 * 1024,
      providerName: 'strands-decider', timeoutMs: 120_000, ...opts,
    };
    const f = opts.fetch ?? globalThis.fetch;
    this.f = (input, init) => f(input, init);
  }

  protected async _ask<const Q extends Record<string, Question>>(state: DecisionInput, questions: Q, options: DecisionCallOptions): Promise<DecisionResult<Q>> {
    if (typeof state !== 'string') throw new Error('SelfHostedDecisionModel sends string state only (the harness serializes it)');
    const body = JSON.stringify({
      state,
      model: this.o.modelId,
      questions: Object.fromEntries(Object.entries(questions).map(([id, q]) => [id, toWireQuestion(q as BenchQuestion)])),
    });
    const bytes = Buffer.byteLength(body);
    if (bytes > this.o.maxBodyBytes) throw new ContextWindowOverflowError(`decision request is ${bytes} bytes; limit ${this.o.maxBodyBytes}`);
    const signal = options.cancelSignal
      ? AbortSignal.any([options.cancelSignal, AbortSignal.timeout(this.o.timeoutMs)])
      : AbortSignal.timeout(this.o.timeoutMs);
    const t0 = performance.now();
    let res: Response;
    try {
      res = await this.f(this.endpoint, {
        method: 'POST',
        headers: { 'content-type': 'application/json', ...(this.o.apiKey ? { authorization: `Bearer ${this.o.apiKey}` } : {}) },
        body,
        signal,
      });
    } catch (err) {
      throw new ModelError(`${this.o.providerName} request failed: ${err instanceof Error ? err.message : String(err)}`, { cause: err });
    }
    let text: string;
    try {
      text = await res.text();
    } catch (err) {
      throw new ModelError(`${this.o.providerName} response body failed: ${err instanceof Error ? err.message : String(err)}`, { cause: err });
    }
    const latencyMs = Math.round(performance.now() - t0);
    if (!res.ok) {
      const msg = `${this.o.providerName} HTTP ${res.status}: ${text.slice(0, 300)}`;
      if (res.status === 429 || res.status === 503 || res.status === 529) throw new ModelThrottledError(msg);
      if (res.status === 413) throw new ContextWindowOverflowError(msg);
      throw new ModelError(msg);
    }
    let payload: unknown;
    try { payload = JSON.parse(text); } catch (err) { throw new ModelError('decision provider returned non-JSON', { cause: err }); }
    const { result, extras } = parseSystemOne(payload, questions as unknown as Record<string, BenchQuestion>, this.o.defaultMinConfidence, latencyMs, this.o.providerName);
    this.last = extras;
    return result as unknown as DecisionResult<Q>;
  }
}

// ── factory ─────────────────────────────────────────────────────────────────

export const DECISION_PROVIDERS = ['decider', 'typesafe', 'llm'] as const;
export type DecisionProvider = (typeof DECISION_PROVIDERS)[number];

export interface DecisionSpec {
  provider: DecisionProvider;
  modelId: string;
  /** decider: server base URL. */
  baseUrl?: string;
  /** Recorded per episode; never contains secrets. */
  label: string;
}

/** env → spec. BENCH_DECIDER_PROVIDER decider (default) | typesafe | llm. Pure. */
export function resolveDecisionSpec(env: Record<string, string | undefined> = process.env): DecisionSpec {
  const provider = (env.BENCH_DECIDER_PROVIDER || 'decider').trim().toLowerCase();
  if (!(DECISION_PROVIDERS as readonly string[]).includes(provider)) {
    throw new Error(`BENCH_DECIDER_PROVIDER=${JSON.stringify(provider)} — use one of ${DECISION_PROVIDERS.join(' | ')}`);
  }
  if (provider === 'decider') {
    const baseUrl = env.STRANDS_DECIDER_URL || 'http://127.0.0.1:8099';
    const modelId = env.BENCH_DECIDER_MODEL_ID || 'strands-decider-latest';
    return { provider, modelId, baseUrl, label: `decider:${modelId}@${baseUrl}` };
  }
  if (provider === 'typesafe') {
    const modelId = env.BENCH_DECIDER_MODEL_ID || 'jev-latest';
    return { provider, modelId, label: `typesafe:${modelId}` };
  }
  const modelId = env.STRANDS_MODEL_ID || 'global.anthropic.claude-sonnet-5';
  return { provider: 'llm', modelId, label: `llm-decision:${modelId}` };
}

/** Build the native DecisionModel for a spec. `llm` wraps a tool-calling Model in the SDK's LLMDecisionModel. */
export function createDecisionModel(spec: DecisionSpec, deps: { llm?: () => Model; fetch?: typeof globalThis.fetch } = {}): DecisionModel {
  if (spec.provider === 'decider') {
    return new SelfHostedDecisionModel({ baseUrl: spec.baseUrl!, modelId: spec.modelId, ...(deps.fetch ? { fetch: deps.fetch } : {}) });
  }
  if (spec.provider === 'typesafe') {
    // Reads TYPESAFE_API_KEY. Policy must act, so the confidence floor is 0 (Uncertain never chosen silently).
    return new TypeSafeDecisionModel({ modelId: spec.modelId, defaultMinConfidence: 0, ...(deps.fetch ? { fetch: deps.fetch } : {}) });
  }
  if (!deps.llm) throw new Error('llm decision provider needs an llm() factory');
  return new LLMDecisionModel({ model: deps.llm() });
}
