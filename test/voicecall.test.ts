/**
 * The shared voice-call plumbing both rails (CLI call, web phone call) sit
 * on: the tool executor's image interception + error prose, and the rule for
 * absorbing spoken turns into the typed agent's history.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Agent, FunctionTool, Model, TextBlock, ToolResultBlock, type JSONValue, type Message, type ModelStreamEvent, type StreamOptions, type ToolContext } from '@strands-agents/sdk';
import { makeExecuteTool, absorbTurn, type AgentLike } from '../src/voicecall.js';

const fakeTool = (name: string, impl: (args: unknown) => unknown) => ({
  toolSpec: { name },
  invoke: async (args: unknown) => impl(args),
});

test('makeExecuteTool: dispatches by toolSpec.name, strings pass through, objects stringify', async () => {
  const exec = makeExecuteTool([
    fakeTool('greet', () => 'hello'),
    fakeTool('status', () => ({ hp: 20 })),
  ]);
  assert.equal(await exec('greet', {}), 'hello');
  assert.equal(await exec('status', {}), '{"hp":20}');
  assert.match(await exec('nope', {}), /no tool named nope/);
});

test('makeExecuteTool: stream-only SDK Tool preserves context and returns final result', async () => {
  let seen: ToolContext | undefined;
  const progress: unknown[] = [];
  const streamOnly = {
    name: 'stream_only',
    description: 'stream-only SDK tool',
    toolSpec: { name: 'stream_only', description: 'stream-only SDK tool', inputSchema: { type: 'object' as const } },
    async *stream(context: ToolContext) {
      seen = context;
      progress.push('working');
      yield { type: 'toolStreamEvent' as const, data: 'working' };
      return new ToolResultBlock({
        toolUseId: context.toolUse.toolUseId,
        status: 'success',
        content: [new TextBlock('done')],
      });
    },
  };
  const controller = new AbortController();
  const agent = {} as ToolContext['agent'];
  const exec = makeExecuteTool([streamOnly], undefined, { agent, cancelSignal: controller.signal, invocationState: { rail: 'voice' } });
  assert.equal(await exec('stream_only', { n: 1 }), 'done');
  assert.deepEqual(progress, ['working']);
  assert.equal(seen?.toolUse.name, 'stream_only');
  assert.deepEqual(seen?.toolUse.input, { n: 1 });
  assert.equal(seen?.agent, agent);
  assert.equal(seen?.cancelSignal, controller.signal);
  assert.equal(seen?.invocationState.rail, 'voice');
});

class OneVoiceToolModel extends Model {
  private calls = 0;
  updateConfig() {}
  getConfig() { return { modelId: 'voice-tool' }; }
  async *stream(_messages: Message[], _options?: StreamOptions): AsyncIterable<ModelStreamEvent> {
    this.calls += 1;
    yield { type: 'modelMessageStartEvent', role: 'assistant' };
    if (this.calls === 1) {
      yield { type: 'modelContentBlockStartEvent', start: { type: 'toolUseStart', name: 'act', toolUseId: 'act-1' } };
      yield { type: 'modelContentBlockDeltaEvent', delta: { type: 'toolUseInputDelta', input: '{"n":1}' } };
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

test('makeExecuteTool: same SDK tool works through Agent stream and direct voice stream', async () => {
  const contexts: ToolContext[] = [];
  const action = new FunctionTool({
    name: 'act', description: 'act', inputSchema: { type: 'object' },
    callback: (input, context) => { contexts.push(context); return { input: input as JSONValue, path: contexts.length }; },
  });
  const sdkAgent = new Agent({ model: new OneVoiceToolModel(), tools: [action], printer: false });
  assert.equal((await sdkAgent.invoke('act', { limits: { turns: 2 } })).stopReason, 'endTurn');
  const direct = makeExecuteTool([action], undefined, { agent: sdkAgent });
  assert.equal(await direct('act', { n: 2 }), '{"input":{"n":2},"path":2}');
  assert.equal(contexts.length, 2);
  assert.equal(contexts[0].toolUse.name, 'act');
  assert.equal(contexts[1].toolUse.name, 'act');
  assert.equal(contexts[1].agent, sdkAgent);
});

test('makeExecuteTool: stream error result becomes spoken error prose', async () => {
  const streamOnly = {
    name: 'bad', description: 'bad',
    toolSpec: { name: 'bad', description: 'bad', inputSchema: { type: 'object' as const } },
    async *stream(context: ToolContext) {
      return new ToolResultBlock({ status: 'error', toolUseId: context.toolUse.toolUseId, content: [], error: new Error('stream failed') });
    },
  };
  assert.equal(await makeExecuteTool([streamOnly])('bad', {}), 'error: stream failed');
});

test('makeExecuteTool: intercepts ImageBlocks — spoken context never eats base64', async () => {
  const exec = makeExecuteTool([
    fakeTool('capture_view', () => ({ type: 'imageBlock', source: { bytes: new Uint8Array(1024) } })),
  ]);
  const out = await exec('capture_view', {});
  assert.match(out, /cannot inspect images mid-call/);
  assert.doesNotMatch(out, /imageBlock/);
});

test('makeExecuteTool: a thrown tool answers in prose — silence hangs a call', async () => {
  const exec = makeExecuteTool([
    fakeTool('dig', () => { throw new Error('no pickaxe equipped'); }),
  ]);
  assert.equal(await exec('dig', {}), 'error: no pickaxe equipped');
});

test('makeExecuteTool: touches the thinker clock on every call', async () => {
  let touches = 0;
  const exec = makeExecuteTool([fakeTool('t', () => 'ok')], () => touches++);
  await exec('t', {});
  await exec('missing', {});
  assert.equal(touches, 2, 'even a missing tool proves a human is on the line');
});

test('absorbTurn: a real exchange lands as a user/assistant pair, tagged (voice)', () => {
  const agent: AgentLike = { messages: [] };
  const ok = absorbTurn(agent, { user: 'come here', assistant: 'on my way', continuation: false });
  assert.equal(ok, true);
  assert.equal(agent.messages.length, 2);
  const [u, a] = agent.messages as Array<{ role: string; content: Array<{ text: string }> }>;
  assert.equal(u.role, 'user');
  assert.equal(u.content[0].text, '(voice) come here');
  assert.equal(a.role, 'assistant');
  assert.equal(a.content[0].text, 'on my way');
});

test('absorbTurn: continuations and userless turns never fake a question', () => {
  const agent: AgentLike = { messages: [] };
  assert.equal(absorbTurn(agent, { user: '', assistant: 'done digging!', continuation: true }), false);
  assert.equal(absorbTurn(agent, { user: '', assistant: 'stray line', continuation: false }), false);
  assert.equal(agent.messages.length, 0);
});

test('absorbTurn pushes real Message instances — the SDK clones every one (#26)', () => {
  const agent = { messages: [] } as unknown as AgentLike;
  absorbTurn(agent, { user: 'dig down', assistant: 'digging', continuation: false });
  for (const m of agent.messages) {
    assert.equal(typeof (m as { clone?: unknown }).clone, 'function',
      'a plain object here throws msg.clone is not a function on the NEXT turn, forever');
    assert.doesNotThrow(() => (m as { clone: () => unknown }).clone());
  }
});

// ── issue #39: the empty text block that bricks the mind ──────────────────
//
// A voice turn whose assistant side is EMPTY (the model answered by calling a
// tool, or the transcript never arrived) used to push `new TextBlock('')`.
// Bedrock refuses the whole history from then on:
//   400 messages: text content blocks must be non-empty  (retryable=false)
// and the block sits EARLY in history, so every rail — journey, thinker, web
// ask, reflex turn — dies until the window slides past it. Live cost: 14
// refusals in 4 minutes, 3 deaths, 42 pending briefings, a bot at 4.3/20 hp.
const textsOf = (agent: AgentLike) =>
  (agent.messages as Array<{ content: Array<{ text?: string }> }>)
    .flatMap((m) => m.content.map((b) => b.text ?? ''));

test('absorbTurn: a tool-only voice turn never pushes an empty text block (#39)', () => {
  const agent: AgentLike = { messages: [] };
  const ok = absorbTurn(agent, { user: 'go mine iron', assistant: '', continuation: false });
  assert.equal(ok, false, 'nothing to absorb: the pair would carry an empty assistant block');
  assert.equal(agent.messages.length, 0);
  assert.deepEqual(textsOf(agent), []);
});

test('absorbTurn: whitespace-only sides are empty too (#39)', () => {
  const agent: AgentLike = { messages: [] };
  assert.equal(absorbTurn(agent, { user: 'hello?', assistant: '  \n\t ', continuation: false }), false);
  assert.equal(absorbTurn(agent, { user: '   ', assistant: 'on my way', continuation: false }), false);
  assert.equal(agent.messages.length, 0, 'a provider counts whitespace-only text as empty');
});

test('absorbTurn: every block it does push has real text (#39)', () => {
  const agent: AgentLike = { messages: [] };
  absorbTurn(agent, { user: ' come here ', assistant: ' on my way ', continuation: false });
  for (const t of textsOf(agent)) assert.ok(t.trim().length > 0, `empty block reached history: ${JSON.stringify(t)}`);
});
