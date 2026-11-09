/**
 * The in-process rig drives the REAL tool layer (allTools + connection guard),
 * counts every call the way v3's sidecar did, and probes blocks from the body's
 * own world. Rails are off here: no server, so no sentinel/reflex ticks.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
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

test('rig: a tool that throws is ok:false with its message, and counts as a tool error', async () => {
  const counters = { toolCalls: 0, toolErrors: 0 };
  const per: Record<string, number> = {};
  const [t] = countingTools([{ toolSpec: { name: 'boom' }, invoke: async (_input: unknown) => { throw new Error('missing 2 sticks'); } }], counters, per);
  await assert.rejects(t.invoke({}), /missing 2 sticks/);
  assert.deepEqual(counters, { toolCalls: 1, toolErrors: 1 });
});

test('rig: the block probe reads world truth (solid = full collision box) and caps the batch', () => {
  const world = new FakeWorld();
  world.set(1, 64, 0, 'stone');
  const { bot } = fakeBot(world);
  const r = probeBlocks(bot as never, [{ x: 1.7, y: 64.2, z: 0.1 }, { x: 2, y: 64, z: 0 }]);
  assert.deepEqual(r.map((b) => [b.name, b.solid]), [['stone', true], ['air', false]]);
  assert.equal(probeBlocks(bot as never, Array.from({ length: 300 }, () => ({ x: 0, y: 0, z: 0 }))).length, 256);
});
