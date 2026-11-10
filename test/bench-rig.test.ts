/**
 * The in-process rig drives the REAL tool layer (allTools + connection guard),
 * counts every call the way v3's sidecar did, and probes blocks from the body's
 * own world. Rails are off here: no server, so no sentinel/reflex ticks.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Agent, FunctionTool, Model, ToolResultBlock, type JSONValue, type Message, type ModelStreamEvent, type StreamOptions, type ToolContext } from '@strands-agents/sdk';
import { FakeWorld, fakeBot } from './fake-bot.js';
import { Rig, probeBlocks, countingTools } from '../src/bench/rig.js';
import type { LiveBody } from '../src/body.js';

function rigFor(world: FakeWorld, inventory: Record<string, number> = {}) {
  const { bot } = fakeBot(world, { position: [0, 64, 0], inventory, blockNames: ['oak_log', 'stone', 'dirt'] });
  const live = { bot, onEachBot: (w: (b: unknown) => void) => w(bot), epoch: () => 0, retire: () => {} } as unknown as LiveBody;
  return new Rig(live, { rails: false });
}

test('rig: the benchmark mounts the game bot\'s own tools (minus capture_view) and counts calls', async () => {
  const world = new FakeWorld();
  world.set(2, 64, 0, 'oak_log');
  const rig = rigFor(world, { oak_log: 1 });
  const names = rig.tools.map((t) => t.toolSpec.name);
  for (const n of ['find_blocks', 'look_around', 'dig_vein', 'craft_item', 'go_to', 'build_structure', 'attack_entity', 'collect_ground_items']) assert.ok(names.includes(n), n);
  assert.ok(!names.includes('capture_view'));
  const found = await rig.call('find_blocks', { name: 'oak_log', maxDistance: 16, count: 3 });
  assert.equal(found.ok, true);
  assert.ok(Array.isArray(found.data) && (found.data as Array<{ block: string }>)[0].block === 'oak_log', 'objects arrive as data, like v3\'s decoded MCP JSON');
  const missing = await rig.call('no_such_tool');
  assert.equal(missing.ok, false);
  assert.equal(rig.counters.toolCalls, 1, 'an unknown name never reached a tool');
  assert.equal(rig.toolCounts.find_blocks, 1);
  assert.deepEqual(rig.state().inv, { oak_log: 1 });
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

test('rig: agent stream executions preserve events/context and count success/error exactly once', async () => {
  const counters = { toolCalls: 0, toolErrors: 0 };
  const per: Record<string, number> = {};
  let seen: ToolContext | undefined;
  const ok = new FunctionTool({
    name: 'ok', description: '', inputSchema: { type: 'object' },
    callback: async function* (_input, context) {
      seen = context;
      yield 'working';
      return 'done';
    },
  });
  const bad = new FunctionTool({
    name: 'bad', description: '', inputSchema: { type: 'object' },
    callback: () => { throw new Error('stream failed'); },
  });
  const [countedOk, countedBad] = countingTools([ok, bad], counters, per);
  const context = toolContext('ok');
  const okStream = countedOk.stream(context);
  const event = await okStream.next();
  assert.equal(event.done, false);
  assert.equal(event.value.data, 'working');
  const next = await okStream.next();
  assert.equal(next.done, true);
  assert.equal(next.value.status, 'success');
  assert.equal(seen, context, 'SDK ToolContext, including cancel signal, reaches callback unchanged');

  const badStream = countedBad.stream(toolContext('bad'));
  let failed = await badStream.next();
  while (!failed.done) failed = await badStream.next();
  assert.equal(failed.value.status, 'error');
  assert.deepEqual(counters, { toolCalls: 2, toolErrors: 1 });
  assert.deepEqual(per, { ok: 1, bad: 1 });
});

class OneToolModel extends Model {
  private calls = 0;
  constructor(private readonly name: string) { super(); }
  updateConfig() {}
  getConfig() { return { modelId: 'one-tool' }; }
  async *stream(_messages: Message[], _options?: StreamOptions): AsyncIterable<ModelStreamEvent> {
    this.calls += 1;
    yield { type: 'modelMessageStartEvent', role: 'assistant' };
    if (this.calls === 1) {
      yield { type: 'modelContentBlockStartEvent', start: { type: 'toolUseStart', name: this.name, toolUseId: `${this.name}-1` } };
      yield { type: 'modelContentBlockDeltaEvent', delta: { type: 'toolUseInputDelta', input: '{}' } };
      yield { type: 'modelContentBlockStopEvent' };
      yield { type: 'modelMessageStopEvent', stopReason: 'toolUse' };
    } else {
      yield { type: 'modelContentBlockStartEvent' };
      yield { type: 'modelContentBlockDeltaEvent', delta: { type: 'textDelta', text: 'done' } };
      yield { type: 'modelContentBlockStopEvent' };
      yield { type: 'modelMessageStopEvent', stopReason: 'endTurn' };
    }
    yield { type: 'modelMetadataEvent', usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2 }, metrics: { latencyMs: 1 } };
  }
}

test('rig: real SDK Agent reaches countingTools stream path exactly once', async () => {
  const counters = { toolCalls: 0, toolErrors: 0 };
  const per: Record<string, number> = {};
  let seen: ToolContext | undefined;
  const action = new FunctionTool({
    name: 'act', description: 'act', inputSchema: { type: 'object' },
    callback: (_input, context) => { seen = context; return 'acted'; },
  });
  const [counted] = countingTools([action], counters, per);
  const agent = new Agent({ model: new OneToolModel(action.name), tools: [counted], printer: false });
  const result = await agent.invoke('act', { limits: { turns: 2 } });
  assert.equal(result.stopReason, 'endTurn');
  assert.equal(seen?.toolUse.name, 'act');
  assert.deepEqual(counters, { toolCalls: 1, toolErrors: 0 });
  assert.deepEqual(per, { act: 1 });
});

test('rig: countingTools admits SDK stream-only tools and counts their error result', async () => {
  const counters = { toolCalls: 0, toolErrors: 0 };
  const per: Record<string, number> = {};
  const streamOnly = {
    name: 'stream_only',
    description: 'stream',
    toolSpec: { name: 'stream_only', description: 'stream', inputSchema: { type: 'object' as const } },
    async *stream(context: ToolContext) {
      return new ToolResultBlock({ status: 'error', toolUseId: context.toolUse.toolUseId, content: [] });
    },
  };
  const [counted] = countingTools([streamOnly], counters, per);
  const stream = counted.stream(toolContext('stream_only'));
  const result = await stream.next();
  assert.equal(result.done, true);
  assert.equal(result.value.status, 'error');
  assert.deepEqual(counters, { toolCalls: 1, toolErrors: 1 });
  assert.deepEqual(per, { stream_only: 1 });
});
test('rig: direct invoke remains one counted call and preserves optional context', async () => {
  const counters = { toolCalls: 0, toolErrors: 0 };
  const per: Record<string, number> = {};
  let seen: ToolContext | undefined;
  const raw = new FunctionTool({
    name: 'direct', description: '', inputSchema: { type: 'object' },
    callback: (_input, context) => { seen = context; return 'done'; },
  });
  const [counted] = countingTools([raw], counters, per);
  const context = toolContext('direct');
  assert.equal(await counted.invoke({}, context), 'done');
  assert.equal(seen, context);
  assert.deepEqual(counters, { toolCalls: 1, toolErrors: 0 });
  assert.deepEqual(per, { direct: 1 });
});

test('rig: the block probe reads world truth (solid = full collision box) and caps the batch', () => {
  const world = new FakeWorld();
  world.set(1, 64, 0, 'stone');
  const { bot } = fakeBot(world);
  const r = probeBlocks(bot as never, [{ x: 1.7, y: 64.2, z: 0.1 }, { x: 2, y: 64, z: 0 }]);
  assert.deepEqual(r.map((b) => [b.name, b.solid]), [['stone', true], ['air', false]]);
  assert.equal(probeBlocks(bot as never, Array.from({ length: 300 }, () => ({ x: 0, y: 0, z: 0 }))).length, 256);
});
