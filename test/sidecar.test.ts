/**
 * The body sidecar's contract, without a Minecraft server: the MCP surface
 * lists the tools' own generated schemas and calls them, results convert the
 * way the SDK's FunctionTool would, and the event ring stays bounded while
 * telling a lagging poller exactly how much it missed.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { z } from 'zod';
import { tool, ImageBlock } from '@strands-agents/sdk';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { InMemoryTransport } from '@modelcontextprotocol/sdk/inMemory.js';
import { EventLog, mcpServerFor, probeBlocks, toMcpResult } from '../src/sidecar.js';
import { FakeWorld, fakeBot } from './fake-bot.js';
import { allTools } from '../src/tools/index.js';

const echo = tool({
  name: 'echo_status',
  description: 'returns what it was given',
  inputSchema: z.object({ n: z.number().default(3) }),
  callback: ({ n }) => ({ n, ok: true }),
});
const boom = tool({
  name: 'boom',
  description: 'fails with advice',
  inputSchema: z.object({}),
  callback: () => { throw new Error('missing 2 sticks — gather then re-call'); },
});

async function connect(tools: unknown[]) {
  const rig = { tools: tools as never, counters: { toolCalls: 0, toolErrors: 0, deaths: 0, reconnects: 0, chat: 0, reflexAsks: 0 }, toolCounts: {} as Record<string, number>, inFlight: 0 };
  const server = mcpServerFor(rig);
  const [a, b] = InMemoryTransport.createLinkedPair();
  await server.connect(a);
  const client = new Client({ name: 'test', version: '0' });
  await client.connect(b);
  return { client, rig };
}

test('MCP lists every tool with the schema the SDK generated from its zod definition', async () => {
  const { client } = await connect([echo, boom]);
  const { tools } = await client.listTools();
  assert.deepEqual(tools.map((t) => t.name), ['echo_status', 'boom']);
  const schema = tools[0]!.inputSchema as { properties?: Record<string, { type?: string; default?: unknown }> };
  assert.equal(schema.properties?.n?.type, 'number');
  assert.equal(schema.properties?.n?.default, 3);
});

test('the full game roster crosses the wire unchanged (count pinned by docs/numbers.json)', async () => {
  const { bot } = fakeBot(new FakeWorld());
  const game = allTools(bot as never);
  const { client } = await connect(game);
  const { tools } = await client.listTools();
  assert.equal(tools.length, game.length);
  assert.ok(tools.some((t) => t.name === 'dig_vein') && tools.some((t) => t.name === 'craft_item'));
});

test('calls run the tool (zod defaults applied) and count themselves', async () => {
  const { client, rig } = await connect([echo, boom]);
  const r = await client.callTool({ name: 'echo_status', arguments: {} });
  assert.deepEqual(JSON.parse((r.content as Array<{ text: string }>)[0]!.text), { n: 3, ok: true });
  assert.equal(rig.counters.toolCalls, 1);
  assert.equal(rig.toolCounts.echo_status, 1);
  assert.equal(rig.inFlight, 0, 'in-flight is released after the call');
});

test('a tool that throws returns isError with its advice, never a transport failure', async () => {
  const { client, rig } = await connect([boom]);
  const r = await client.callTool({ name: 'boom', arguments: {} });
  assert.equal(r.isError, true);
  assert.match((r.content as Array<{ text: string }>)[0]!.text, /missing 2 sticks/);
  assert.equal(rig.counters.toolErrors, 1);
  const missing = await client.callTool({ name: 'nope', arguments: {} });
  assert.equal(missing.isError, true);
});

test('result conversion: images become MCP image content, empty strings never become empty text (#39)', () => {
  const img = toMcpResult(new ImageBlock({ format: 'png', source: { bytes: new Uint8Array([1, 2, 3]) } }));
  assert.equal(img.content[0]!.type, 'image');
  assert.equal((img.content[0] as { mimeType: string }).mimeType, 'image/png');
  assert.equal((toMcpResult('') .content[0] as { text: string }).text, '(empty result)');
  assert.equal((toMcpResult(undefined).content[0] as { text: string }).text, 'null');
  assert.equal((toMcpResult(7).content[0] as { text: string }).text, '7');
});

test('event ring: bounded, seq survives eviction, a lagging poller is told what it missed', () => {
  const log = new EventLog(3, () => 1000);
  for (let i = 1; i <= 5; i++) log.push('note', `(sentinel) note ${i}`);
  assert.equal(log.size, 3);
  assert.equal(log.dropped, 2);
  const { events, last, missed } = log.since(0);
  assert.deepEqual(events.map((e) => e.seq), [3, 4, 5]);
  assert.equal(last, 5);
  assert.equal(missed, 2);
  assert.equal(log.since(5).events.length, 0);
  assert.equal(log.since(4).missed, 0);
});

test('event ring refuses a wordless event — the empty-text brick one hop earlier (#39)', () => {
  const log = new EventLog(10);
  assert.equal(log.push('note', '   '), undefined);
  assert.equal(log.size, 0);
});

test('long-poll wakes on the next push instead of sleeping out its window', async () => {
  const log = new EventLog(10);
  const t0 = Date.now();
  const waiting = log.wait(0, 5_000);
  setTimeout(() => log.push('chat', 'hello', 'Cagatay'), 20);
  await waiting;
  assert.ok(Date.now() - t0 < 1_000, 'woke on push');
  assert.equal(log.since(0).events[0]!.name, 'Cagatay');
});

test('block probe: solid means a full collision box, whatever the block is called', () => {
  // A shelter dug into stone is sealed by stone, not by the cobblestone the plan named (v3 smoke, 2026-09-28).
  const world: Record<string, { name: string; boundingBox: string }> = {
    '0,64,0': { name: 'stone', boundingBox: 'block' },
    '1,64,0': { name: 'air', boundingBox: 'empty' },
    '2,64,0': { name: 'leaf_litter', boundingBox: 'empty' },
  };
  const bot = { blockAt: (p: { x: number; y: number; z: number }) => world[`${p.x},${p.y},${p.z}`] ?? null };
  const out = probeBlocks(bot as never, [{ x: 0, y: 64, z: 0 }, { x: 1.7, y: 64.2, z: 0 }, { x: 2, y: 64, z: 0 }, { x: 9, y: 9, z: 9 }]);
  assert.deepEqual(out.map((b) => [b.name, b.solid]), [['stone', true], ['air', false], ['leaf_litter', false], ['unloaded', false]]);
});
