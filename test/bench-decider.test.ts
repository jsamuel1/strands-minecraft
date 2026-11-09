/**
 * The SDK-native decision seam: SelfHostedDecisionModel is a real
 * @strands-agents/sdk DecisionModel (ask() is the SDK's), its wire matches the
 * System One shape strands-decider serves, and its parser accepts the server's
 * 4-dp rounding that the SDK's own parser rejects.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { DecisionModel, TypeSafeDecisionModel, Uncertain, ModelThrottledError, ContextWindowOverflowError, ModelError } from '@strands-agents/sdk';
import { SelfHostedDecisionModel, toWireQuestion, parseSystemOne, resolveDecisionSpec, createDecisionModel, PROBABILITY_SUM_TOLERANCE } from '../src/bench/decider.js';

type Call = { url: string; body: any; headers: Record<string, string> };

function fakeServer(reply: (body: any) => { status?: number; json?: unknown; text?: string }) {
  const calls: Call[] = [];
  const fetch = (async (url: string, init: RequestInit) => {
    const body = JSON.parse(String(init.body));
    calls.push({ url, body, headers: init.headers as Record<string, string> });
    const r = reply(body);
    return new Response(r.text ?? JSON.stringify(r.json), { status: r.status ?? 200 });
  }) as unknown as typeof globalThis.fetch;
  return { fetch, calls };
}

// The exact shape strands-decider 0.3.0 returned on the live 12-option probe: 4-dp, sum 0.9999.
const ROUNDED = Object.fromEntries(Array.from({ length: 12 }, (_, i) => [`opt_${i}`, i === 3 ? 0.4499 : 0.05]));
const ids = Object.keys(ROUNDED) as [string, ...string[]];
const decider = (extra: Record<string, unknown> = {}) => ({
  model: 'strands-decider-2B-qwen3.5-v1-2610',
  answers: { next_action: { type: 'choice', choice: 'opt_3', probabilities: ROUNDED, confidence: 0.41 }, ...extra },
  usage: { input_tokens: 812, output_tokens: 1 },
  latency_ms: 1905.2,
});

test('SelfHostedDecisionModel is an SDK DecisionModel: ask() validates questions before any request', async () => {
  const s = fakeServer(() => ({ json: decider() }));
  const m = new SelfHostedDecisionModel({ baseUrl: 'http://127.0.0.1:8099/', fetch: s.fetch });
  assert.ok(m instanceof DecisionModel);
  await assert.rejects(m.ask('s', {}), /at least one question/);
  await assert.rejects(m.ask('s', { q: { question: 'x', choices: ['uncertain'] } }), /reserved/);
  assert.equal(s.calls.length, 0, 'the SDK base validates; nothing reached the wire');
});

test('wire: POST /v1/systemone with state, model and choice criteria carrying the option descriptions', async () => {
  const s = fakeServer(() => ({ json: decider({ ok: { type: 'noul', noul: 0.8 } }) }));
  const m = new SelfHostedDecisionModel({ baseUrl: 'http://127.0.0.1:8099/', fetch: s.fetch, modelId: 'x' });
  const r = await m.ask('{"goal": "logs"}', {
    next_action: { question: 'Which?', choices: ids, uncertainty: false, descriptions: { opt_3: 'Fell the oak tree.' } } as any,
    ok: { question: 'Did it work?', choices: 'boolean' },
  });
  assert.equal(s.calls[0].url, 'http://127.0.0.1:8099/v1/systemone');
  assert.equal(s.calls[0].body.state, '{"goal": "logs"}');
  assert.equal(s.calls[0].body.model, 'x');
  assert.deepEqual(s.calls[0].body.questions.ok, { type: 'noul', instructions: 'Did it work?' });
  assert.equal(s.calls[0].body.questions.next_action.type, 'choice');
  assert.equal(s.calls[0].body.questions.next_action.criteria.opt_3, 'Fell the oak tree.');
  assert.equal(s.calls[0].body.questions.next_action.criteria.opt_0, null, 'no description → null, as the SDK sends');
  assert.equal(s.calls[0].headers.authorization, undefined, 'a local server needs no token');
  assert.equal(r.answers.next_action, 'opt_3');
  assert.equal(r.answers.ok, true);
  assert.equal(r.metadata.modelId, 'strands-decider-2B-qwen3.5-v1-2610');
  assert.deepEqual(r.usage, { inputTokens: 812, outputTokens: 1, totalTokens: 813 });
  assert.equal(m.last.serverLatencyMs, 1905.2);
  assert.equal(m.last.probabilitySums?.next_action, 0.9999);
});

test('the SDK 1.24 TypeSafeDecisionModel rejects the same rounded answer — why the subclass exists', async () => {
  const s = fakeServer(() => ({ json: decider() }));
  const native = new TypeSafeDecisionModel({ apiKey: 'k', fetch: s.fetch });
  await assert.rejects(native.ask('s', { next_action: { question: 'Which?', choices: ids, uncertainty: false } }), /must sum to 1/);
});

test('parser: tolerance is the rounding, not more; unknown and non-top choices are refused', () => {
  const q = { next_action: { question: 'Which?', choices: ['a', 'b'] as const, uncertainty: false as const } };
  const ans = (p: Record<string, number>, choice = 'a') => ({ model: 'm', answers: { next_action: { type: 'choice', choice, probabilities: p, confidence: 0.5 } } });
  assert.equal(parseSystemOne(ans({ a: 0.6, b: 0.3995 }), q, 0, 1, 'x').result.answers.next_action, 'a');
  assert.throws(() => parseSystemOne(ans({ a: 0.6, b: 0.398 }), q, 0, 1, 'x'), /sum to/);
  assert.throws(() => parseSystemOne(ans({ a: 0.6, b: 0.4 }, 'c'), q, 0, 1, 'x'), /unknown choice/);
  assert.throws(() => parseSystemOne(ans({ a: 0.4, b: 0.6 }), q, 0, 1, 'x'), /highest probability/);
  assert.throws(() => parseSystemOne(ans({ a: 1 }), q, 0, 1, 'x'), /exactly the requested/);
  assert.ok(PROBABILITY_SUM_TOLERANCE <= 1e-3);
});

test('parser: Uncertain below the confidence floor unless uncertainty is false (SDK semantics)', () => {
  const p = { model: 'm', answers: { a: { type: 'choice', choice: 'x', probabilities: { x: 0.5, y: 0.5 }, confidence: 0.2 } } };
  const r1 = parseSystemOne(p, { a: { question: 'q', choices: ['x', 'y'] } }, 0.5, 1, 'x').result;
  assert.ok(r1.answers.a instanceof Uncertain);
  const r2 = parseSystemOne(p, { a: { question: 'q', choices: ['x', 'y'], uncertainty: false } }, 0.5, 1, 'x').result;
  assert.equal(r2.answers.a, 'x');
});

test('HTTP errors map to the SDK error classes; a body over the limit never leaves', async () => {
  for (const [status, cls] of [[429, ModelThrottledError], [503, ModelThrottledError], [413, ContextWindowOverflowError], [422, ModelError]] as const) {
    const s = fakeServer(() => ({ status, text: 'nope' }));
    const m = new SelfHostedDecisionModel({ baseUrl: 'http://h', fetch: s.fetch });
    await assert.rejects(m.ask('s', { q: { question: 'q', choices: 'boolean' } }), cls);
  }
  const s = fakeServer(() => ({ json: {} }));
  const m = new SelfHostedDecisionModel({ baseUrl: 'http://h', fetch: s.fetch, maxBodyBytes: 100 });
  await assert.rejects(m.ask('x'.repeat(200), { q: { question: 'q', choices: 'boolean' } }), ContextWindowOverflowError);
  assert.equal(s.calls.length, 0);
});

test('toWireQuestion: noul for boolean, choice with criteria; 256 choices refused', () => {
  assert.deepEqual(toWireQuestion({ question: 'q', choices: 'boolean' }), { type: 'noul', instructions: 'q' });
  assert.throws(() => toWireQuestion({ question: 'q', choices: Array.from({ length: 256 }, (_, i) => `c${i}`) }), /at most 255/);
});

test('resolveDecisionSpec: decider default, typesafe and llm; unknown names the valid ones; labels carry no secrets', () => {
  assert.deepEqual(resolveDecisionSpec({}), { provider: 'decider', modelId: 'strands-decider-latest', baseUrl: 'http://127.0.0.1:8099', label: 'decider:strands-decider-latest@http://127.0.0.1:8099' });
  assert.equal(resolveDecisionSpec({ BENCH_DECIDER_PROVIDER: 'typesafe', TYPESAFE_API_KEY: 'sekrit' }).label, 'typesafe:jev-latest');
  assert.equal(resolveDecisionSpec({ BENCH_DECIDER_PROVIDER: 'llm', STRANDS_MODEL_ID: 'm' }).label, 'llm-decision:m');
  assert.throws(() => resolveDecisionSpec({ BENCH_DECIDER_PROVIDER: 'hobson' }), /decider \| typesafe \| llm/);
  assert.ok(createDecisionModel(resolveDecisionSpec({})) instanceof SelfHostedDecisionModel);
  const saved = process.env.TYPESAFE_API_KEY;
  process.env.TYPESAFE_API_KEY = 'test-key';
  try {
    assert.ok(createDecisionModel({ provider: 'typesafe', modelId: 'jev-latest', label: '' }) instanceof TypeSafeDecisionModel);
  } finally {
    if (saved === undefined) delete process.env.TYPESAFE_API_KEY; else process.env.TYPESAFE_API_KEY = saved;
  }
  assert.throws(() => createDecisionModel({ provider: 'llm', modelId: 'm', label: '' }), /llm\(\) factory/);
});
