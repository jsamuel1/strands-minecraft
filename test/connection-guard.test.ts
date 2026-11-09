/**
 * Tests for connection-guard — the reconnect-orphan fix (issue #20).
 *
 * The failure being pinned: a tool awaiting an event on a socket that just died
 * never settles, so the agent turn never ends and the human message that started
 * it is lost with no answer and no error. Everything here is offline and uses a
 * hand-cranked epoch counter, because "the socket died" is one number changing.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  Agent,
  FunctionTool,
  Model,
  type JSONValue,
  type Message,
  type StreamOptions,
  type Tool,
  type ToolContext,
} from '@strands-agents/sdk';
import { guardEpoch, guardTools, connectionLostMessage } from '../src/tools/connection-guard.js';

const POLL = 10; // tests do not need a quarter-second of patience

test('guardEpoch: ordinary work passes straight through, value and rejection alike', async () => {
  let epoch = 7;
  assert.equal(await guardEpoch(Promise.resolve('ok'), { tool: 'get_status', epoch: () => epoch, pollMs: POLL }), 'ok');
  await assert.rejects(
    () => guardEpoch(Promise.reject(new Error('No block at (1,2,3)')), { tool: 'inspect_block', epoch: () => epoch, pollMs: POLL }),
    /No block at \(1,2,3\)/,
    "a tool's own error must survive the guard unchanged",
  );
  assert.equal(epoch, 7);
});

test('guardEpoch: a hung call loses to a reconnect instead of hanging the turn', async () => {
  let epoch = 1;
  // The real shape: openContainer awaiting window_items from a dead socket.
  const neverSettles = new Promise<string>(() => {});
  setTimeout(() => { epoch = 2; }, POLL * 2);
  await assert.rejects(
    () => guardEpoch(neverSettles, {
      tool: 'container_transact',
      epoch: () => epoch,
      pollMs: POLL,
      detail: () => 'The body was StrandsBot at the time of the drop.',
    }),
    (err: Error) => {
      assert.match(err.message, /connection dropped while 'container_transact' was in flight/);
      assert.match(err.message, /result is UNKNOWN/);
      assert.match(err.message, /may have half-happened server-side/, 'a chest click can land and still be lost');
      assert.match(err.message, /do NOT assume success or failure/);
      assert.match(err.message, /Re-check the world first/);
      assert.match(err.message, /StrandsBot at the time of the drop/, 'the detail hook is included');
      return true;
    },
  );
});

test('guardEpoch: work that finished beats a reconnect that follows', async () => {
  let epoch = 1;
  const work = new Promise<string>((r) => setTimeout(() => r('withdrew 5x iron_ingot'), POLL));
  const p = guardEpoch(work, { tool: 'container_transact', epoch: () => epoch, pollMs: POLL });
  setTimeout(() => { epoch = 2; }, POLL * 3); // kicked a moment later — too late to matter
  assert.equal(await p, 'withdrew 5x iron_ingot');
});

test('guardEpoch: a slow-but-honest call is not interrupted just for being slow', async () => {
  const epoch = 1;
  const slow = new Promise<string>((r) => setTimeout(() => r('dug 30 blocks'), POLL * 8));
  assert.equal(await guardEpoch(slow, { tool: 'dig_block', epoch: () => epoch, pollMs: POLL }), 'dug 30 blocks');
});

/** A tool with a private field: the wrapper must not break `this`. */
class FakeTool {
  #calls = 0;
  toolSpec = { name: 'place_entity' };
  hang = false;
  constructor(hang = false) { this.hang = hang; }
  async invoke(input: unknown) {
    this.#calls++;
    if (this.hang) return new Promise(() => {}); // awaiting an entity that will never spawn
    return `placed ${JSON.stringify(input)}`;
  }
  calls() { return this.#calls; }
}

test('guardTools: invoke is guarded, everything else reaches the real tool intact', async () => {
  let epoch = 4;
  const real = new FakeTool();
  const [wrapped] = guardTools([real], { epoch: () => epoch, pollMs: POLL });
  const w = wrapped as FakeTool;
  assert.equal(w.toolSpec.name, 'place_entity', 'the SDK reads the spec off the object it was handed');
  assert.equal(await w.invoke({ ref: 'boat' }), 'placed {"ref":"boat"}', 'arguments pass through');
  assert.equal(w.calls(), 1, 'a method using a PRIVATE field still runs with the real `this`');
  assert.equal(epoch, 4);
});

test('guardTools: the boat case — the tool that killed the socket stops waiting on it', async () => {
  let epoch = 1;
  const [wrapped] = guardTools([new FakeTool(true)], { epoch: () => epoch, pollMs: POLL });
  setTimeout(() => { epoch = 2; }, POLL * 2);
  await assert.rejects(
    () => (wrapped as FakeTool).invoke({ ref: 'boat' }) as Promise<unknown>,
    /connection dropped while 'place_entity' was in flight/,
    'issue #21: the boat packet got us kicked, and the boat tool then waited forever',
  );
});

function toolContext(name: string, input: JSONValue = {}): ToolContext {
  return {
    toolUse: { name, toolUseId: `${name}-1`, input },
    agent: {} as never,
    invocationState: {},
    cancelSignal: new AbortController().signal,
    interrupt: () => { throw new Error('not used'); },
  };
}

async function finish<T>(stream: AsyncGenerator<unknown, T, unknown>): Promise<T> {
  let next = await stream.next();
  while (!next.done) next = await stream.next();
  return next.value;
}

class OneToolModel extends Model {
  private calls = 0;
  constructor(private readonly name: string) { super(); }
  updateConfig() {}
  getConfig() { return { modelId: 'one-tool' }; }
  async *stream(_messages: Message[], _options?: StreamOptions) {
    this.calls += 1;
    yield { type: 'modelMessageStartEvent' as const, role: 'assistant' as const };
    if (this.calls === 1) {
      yield { type: 'modelContentBlockStartEvent' as const, start: { type: 'toolUseStart' as const, name: this.name, toolUseId: `${this.name}-1` } };
      yield { type: 'modelContentBlockDeltaEvent' as const, delta: { type: 'toolUseInputDelta' as const, input: '{}' } };
      yield { type: 'modelContentBlockStopEvent' as const };
      yield { type: 'modelMessageStopEvent' as const, stopReason: 'toolUse' as const };
    } else {
      yield { type: 'modelContentBlockStartEvent' as const };
      yield { type: 'modelContentBlockDeltaEvent' as const, delta: { type: 'textDelta' as const, text: 'done' } };
      yield { type: 'modelContentBlockStopEvent' as const };
      yield { type: 'modelMessageStopEvent' as const, stopReason: 'endTurn' as const };
    }
    yield { type: 'modelMetadataEvent' as const, usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2 }, metrics: { latencyMs: 1 } };
  }
}

test('guardTools: SDK FunctionTool stream preserves events, context, and final result', async () => {
  let epoch = 4;
  let seen: ToolContext | undefined;
  const real = new FunctionTool({
    name: 'place_entity', description: 'place', inputSchema: { type: 'object' },
    callback: async function* (_input, context) {
      seen = context;
      yield 'placing';
      return 'placed';
    },
  });
  const [wrapped] = guardTools([real], { epoch: () => epoch, pollMs: POLL });
  const context = toolContext('place_entity');
  const stream = wrapped.stream(context);
  const event = await stream.next();
  assert.equal(event.done, false);
  assert.equal(event.value.data, 'placing');
  const result = await stream.next();
  assert.equal(result.done, true);
  assert.equal(result.value.status, 'success');
  assert.equal(seen, context, 'full SDK ToolContext reaches the real callback');
});

test('guardTools: reconnect between stream events rejects the original execution epoch', async () => {
  let epoch = 1;
  const real = new FunctionTool({
    name: 'dig_vein', description: 'dig', inputSchema: { type: 'object' },
    callback: async function* () {
      yield 'first block';
      return 'done';
    },
  });
  const [wrapped] = guardTools([real], { epoch: () => epoch, pollMs: POLL });
  const stream = wrapped.stream(toolContext('dig_vein'));
  const event = await stream.next();
  assert.equal(event.done, false);
  epoch = 2;
  await assert.rejects(
    () => stream.next(),
    /connection dropped while 'dig_vein' was in flight/,
  );
});

class StreamOnlyTool {
  readonly name = 'stream_only';
  readonly description = 'stream-only SDK tool';
  readonly toolSpec = { name: this.name, description: this.description, inputSchema: { type: 'object' } };
  async *stream(context: ToolContext) {
    await new Promise<void>(() => {});
    return { status: 'success', toolUseId: context.toolUse.toolUseId, content: [] } as never;
  }
}

test('guardTools: SDK stream-only Tool is guarded without an invoke method', async () => {
  let epoch = 1;
  const real = new StreamOnlyTool();
  const [wrapped] = guardTools([real], { epoch: () => epoch, pollMs: POLL });
  assert.notEqual(wrapped, real, 'Tool.stream is the SDK contract; invoke is optional');
  setTimeout(() => { epoch = 2; }, POLL * 2);
  await assert.rejects(
    () => finish(wrapped.stream(toolContext(real.name))),
    /connection dropped while 'stream_only' was in flight/,
  );
});

test('guardTools: SDK stream waiting on a dead socket rejects after reconnect', async () => {
  let epoch = 1;
  const real = new FunctionTool({
    name: 'container_transact', description: 'open', inputSchema: { type: 'object' },
    callback: async () => new Promise(() => {}),
  });
  const [wrapped] = guardTools([real], {
    epoch: () => epoch,
    pollMs: POLL,
    detail: () => 'The body was StrandsBot at the time of the drop.',
  });
  setTimeout(() => { epoch = 2; }, POLL * 2);
  await assert.rejects(
    () => finish(wrapped.stream(toolContext('container_transact'))),
    /connection dropped while 'container_transact' was in flight.*result is UNKNOWN.*StrandsBot/s,
  );
});

test('guardTools: real SDK Agent executes the guarded stream path after reconnect', async () => {
  let epoch = 1;
  let invoked = 0;
  const real = new FunctionTool({
    name: 'container_transact', description: 'open', inputSchema: { type: 'object' },
    callback: async () => {
      invoked += 1;
      return new Promise(() => {});
    },
  });
  const [wrapped] = guardTools([real], { epoch: () => epoch, pollMs: POLL });
  const agent = new Agent({ model: new OneToolModel(real.name), tools: [wrapped as Tool], printer: false });
  setTimeout(() => { epoch = 2; }, POLL * 2);
  const result = await agent.invoke('open it', { limits: { turns: 2 } });
  assert.equal(result.stopReason, 'endTurn');
  assert.equal(invoked, 1);
  const errors = agent.messages.flatMap((message) => message.content)
    .filter((block) => block.type === 'toolResultBlock')
    .filter((block) => block.status === 'error');
  assert.equal(errors.length, 1, 'Agent ToolExecutor consumed guarded Tool.stream and surfaced one error result');
  assert.match(JSON.stringify(errors[0]), /connection dropped while 'container_transact' was in flight/);
});

test('guardTools: skip leaves a tool that is meant to span a reconnect alone', async () => {
  let epoch = 1;
  const real = new FakeTool(true);
  real.toolSpec = { name: 'respawn' };
  const [wrapped] = guardTools([real], { epoch: () => epoch, pollMs: POLL, skip: ['respawn'] });
  assert.equal(wrapped, real, 'not even wrapped');
  epoch = 2;
});

test('guardTools: an entry with no invoke is passed through untouched', () => {
  const odd = { toolSpec: { name: 'weird' } };
  const [wrapped] = guardTools([odd], { epoch: () => 1 });
  assert.equal(wrapped, odd);
});

test('connectionLostMessage: usable without a detail', () => {
  const m = connectionLostMessage({ tool: 'walk_to' });
  assert.match(m, /'walk_to'/);
  assert.doesNotMatch(m, /undefined/);
});
