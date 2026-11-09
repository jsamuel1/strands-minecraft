/**
 * 📞 One call factory for every voice surface.
 *
 * The CLI's 'call' rail and the web dashboard's phone call are the SAME
 * agent on the same realtime socket — same tools, same image interception,
 * same absorption of spoken turns into the typed agent's history. Only the
 * audio differs (host child processes vs a browser over WebSocket), and that
 * is a transport, not a call. This factory holds everything the two rails
 * share so they cannot drift apart.
 *
 * Absorption rule (learned on the first live tool call): only completed
 * exchanges with a real user side land in agent.messages. A continuation —
 * the model speaking again after a tool, with no new user turn — has no
 * question to pair with; writing one in would fake the history.
 */
import type { Bot } from 'mineflayer';
import { randomUUID } from 'node:crypto';
import type {
  InvokableTool,
  JSONValue,
  LocalAgent,
  Message,
  Tool,
  ToolContext,
  ToolResultBlock,
} from '@strands-agents/sdk';
import { textMessage } from './history-doctor.js';
import { RealtimeCall, type RealtimeEvent } from './realtime/realtime.js';
import type { TransportFactories } from './realtime/transport.js';

/** The one slice of a Strands Agent this file touches. */
/**
 * Just the history, and typed as `Message[]` on purpose: this is the seam where a
 * plain object literal once entered the transcript and bricked every later turn
 * (`msg.clone is not a function` — the SDK clones the whole history per call).
 * `unknown[]` is what allowed it, so the type is the guard now. Anything pushed
 * here must be a real Message, and the compiler says so.
 */
export interface AgentLike { messages: Message[] }

/** Runtime-tolerant view: production entries are SDK Tools; older callers may pass invoke-only tools. */
type VoiceTool = Partial<Pick<Tool, 'stream' | 'toolSpec'>> & Partial<Pick<InvokableTool<unknown, unknown>, 'invoke'>>;

const blockValue = (block: { type?: string; text?: string; json?: JSONValue }): unknown => {
  if (block.type === 'textBlock') return block.text ?? '';
  if (block.type === 'jsonBlock') return block.json;
  return block;
};

/** Turn an SDK ToolResultBlock into the direct executor's raw-value shape. */
function voiceToolResult(result: ToolResultBlock): unknown {
  const values = result.content.map(blockValue);
  const value = values.length === 1 ? values[0] : values;
  if (result.status === 'error') {
    const detail = values.map((v) => typeof v === 'string' ? v : JSON.stringify(v)).join(' | ');
    throw result.error ?? new Error(detail.replace(/^Error:\s*/, '') || 'tool failed');
  }
  return value;
}

/** Execute a stream-only SDK Tool to its terminal result, preserving its ToolContext. */
async function invokeVoiceTool(t: VoiceTool, name: string, args: unknown, context?: Partial<ToolContext>): Promise<unknown> {
  const toolContext = context?.toolUse && context.agent && context.invocationState && context.cancelSignal && context.interrupt
    ? context as ToolContext
    : {
      toolUse: { name, toolUseId: context?.toolUse?.toolUseId ?? `voice-${randomUUID()}`, input: args as JSONValue },
      agent: context?.agent ?? ({} as LocalAgent),
      invocationState: context?.invocationState ?? {},
      cancelSignal: context?.cancelSignal ?? new AbortController().signal,
      ...(context?.invocationContext ? { invocationContext: context.invocationContext } : {}),
      interrupt: context?.interrupt ?? (() => { throw new Error(`tool ${name} requested an interrupt outside an SDK Agent turn`); }),
    } as ToolContext;
  if (typeof t.invoke === 'function') return t.invoke(args, toolContext);
  if (typeof t.stream !== 'function') throw new Error(`tool ${name} has neither invoke nor stream`);
  const stream = t.stream(toolContext);
  let next = await stream.next();
  while (!next.done) next = await stream.next();
  return voiceToolResult(next.value);
}

/**
 * The tool executor both rails share. Voice can't look at images: catch
 * capture_view's ImageBlock before it becomes 200KB of base64 read into the
 * model's spoken context. Errors come back as prose — a thrown tool that
 * answers nothing leaves a call hanging in silence forever.
 */
export function makeExecuteTool(tools: unknown[], touch?: () => void, context?: Partial<ToolContext>) {
  return async (name: string, args: unknown, callContext?: Partial<ToolContext>): Promise<string> => {
    touch?.();
    const t = tools.find((x) => (x as { toolSpec?: { name?: string } }).toolSpec?.name === name) as VoiceTool | undefined;
    if (!t) return `error: no tool named ${name}`;
    try {
      const executionContext = callContext
        ? (callContext.agent ? callContext : { ...context, ...callContext, agent: context?.agent })
        : context;
      const r = await invokeVoiceTool(t, name, args, executionContext);
      if (r && typeof r === 'object' && (r as { type?: string }).type === 'imageBlock') {
        return 'Screenshot captured, but I cannot inspect images mid-call — ask me over text chat to describe what I see.';
      }
      return typeof r === 'string' ? r : JSON.stringify(r ?? {});
    } catch (e) {
      return `error: ${e instanceof Error ? e.message : String(e)}`;
    }
  };
}

/**
 * Fold a completed spoken exchange into the typed agent's history.
 *
 * These MUST be real Message instances. The SDK clones the whole history on
 * every model call (`this.messages.map((msg) => msg.clone())`), so a plain object
 * literal in there is a landmine that detonates on the NEXT turn, not this one —
 * and on every turn after it, forever. The live soak paid the full price: two
 * `as never` casts here, one voice call, and then `msg.clone is not a function`
 * on every journey step, every thinker cycle and every web ask until the process
 * died. The bot's body kept walking around with no mind at all, and was finally
 * kicked for `disconnect.timeout`.
 *
 * `as never` is what let it compile: it silenced the one check that knew these
 * were the wrong shape.
 */
export function absorbTurn(agent: AgentLike, e: { user: string; assistant: string; continuation: boolean }): boolean {
  if (e.continuation) return false;
  // Both sides go through textMessage, which returns null for empty or
  // whitespace-only text. A pair is absorbed ONLY if both halves have words:
  //
  //  - no user words  → there is no question to record (a stray line).
  //  - no assistant words → the turn answered by CALLING A TOOL and said
  //    nothing. Recording it as `TextBlock('')` is what bricked the mind in
  //    issue #39: the provider then refuses the whole history with
  //    `messages: text content blocks must be non-empty`, retryable=false,
  //    on EVERY rail, until the window slides past the block. A tool-only
  //    voice turn is normal, so the right amount of history to write is none.
  const spoken = (e.user ?? '').trim();
  const user = spoken ? textMessage('user', `(voice) ${spoken}`) : null;
  const assistant = textMessage('assistant', e.assistant);
  if (!user || !assistant) return false;
  agent.messages.push(user, assistant);
  return true;
}

export interface VoiceCallDeps {
  apiKey: string;
  bot: Bot;
  agent: AgentLike;
  /** The Strands tools to mount — caller decides the roster. */
  tools: unknown[];
  /** Reset the thinker's idle clock — a human is on the line. */
  touch?: () => void;
  /** Surface hook: rendering, web mirroring. Absorption already happened. */
  onEvent?: (e: RealtimeEvent) => void;
  /** Browser audio (web rail). Omitted = host sox/ffmpeg (CLI rail). */
  transport?: TransportFactories;
}

export function createVoiceCall(deps: VoiceCallDeps): RealtimeCall {
  return new RealtimeCall({
    apiKey: deps.apiKey,
    tools: deps.tools,
    agent: deps.agent,
    executeTool: makeExecuteTool(deps.tools, deps.touch, { agent: deps.agent as unknown as LocalAgent }),
    context:
      `You are ${deps.bot.username} in a Minecraft world at ${deps.bot.entity?.position}. ` +
      `Players online: ${Object.keys(deps.bot.players ?? {}).filter((n) => n !== deps.bot.username).join(', ') || 'none'}.`,
    onEvent: (e) => {
      if (e.type === 'user_transcript') deps.touch?.();
      if (e.type === 'turn') absorbTurn(deps.agent, e);
      deps.onEvent?.(e);
    },
    ...(deps.transport ?? {}),
  });
}
