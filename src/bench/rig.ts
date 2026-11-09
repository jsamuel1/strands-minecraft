/**
 * 🦾 The benchmark rig — one real body, in-process, wired like `npm start`.
 *
 * v3 ran this in a separate Node sidecar and drove it from Python over MCP.
 * Here the harness is TypeScript, so the body, the 60 tools (`allTools`, with
 * the connection guard), the reflexes, the sentinel and the pain/hunger/death
 * rails (`body-rails.ts`) are the SAME modules the game bot uses, called
 * directly. Nothing about the body is re-implemented for the benchmark.
 *
 * Benchmark settings, identical for every arm (v3's):
 *   - idle reflexes that move or collect are off (REFLEX_MODES_OFF), the safety
 *     spinal cord stays on;
 *   - no thinker, no chat rail, no journeys, no fleet: the task is the only ask;
 *   - MEMORY_DIR is a fresh temp dir per episode (no waypoints, no graves).
 */
import { Vec3 } from 'vec3';
import type { Bot } from 'mineflayer';
import type { LiveBody } from '../body.js';
import { allTools } from '../tools/index.js';
import { guardTools } from '../tools/connection-guard.js';
import { startSentinel, type SentinelHandle } from '../sentinel.js';
import { startReflexes, type ReflexHandle } from '../reflexes.js';
import { legs, registerLegs } from '../legs.js';
import { wireBodyRails } from '../body-rails.js';
import { systemNoteFor, classifyMessage } from '../chatrail.js';
import { inventoryHighlights } from '../digest.js';
import { carriedView, countCarried } from '../tools/helpers.js';
import { errorOutcome, toOutcome, type Body, type BodyEvent, type BodySnapshot, type ToolOutcome, type Vec } from './body.js';

/** Reflexes that move the body or collect on their own — off for every arm, so a gain is the brain's. */
export const REFLEX_MODES_OFF = 'item_magnet,elbow_room,idle_staring';

export interface MountedTool {
  toolSpec: { name: string; description?: string; inputSchema?: unknown };
  invoke: (input: unknown) => Promise<unknown>;
}

/** What occupies each position, and whether it blocks movement (a full collision box). */
export function probeBlocks(bot: Pick<Bot, 'blockAt'>, positions: Vec[], max = 256): Array<Vec & { name: string; solid: boolean }> {
  return positions.slice(0, max).map((p) => {
    const b = bot.blockAt(new Vec3(Math.floor(p.x), Math.floor(p.y), Math.floor(p.z)));
    return { x: p.x, y: p.y, z: p.z, name: b?.name ?? 'unloaded', solid: b?.boundingBox === 'block' };
  });
}

/** Counts every tool call (policy steps, perception, agent calls alike): the body's own counter. */
export function countingTools<T extends MountedTool>(tools: T[], counters: { toolCalls: number; toolErrors: number }, perTool: Record<string, number>): T[] {
  return tools.map((t) => new Proxy(t, {
    get(obj, prop) {
      if (prop === 'invoke') {
        return async (input: unknown) => {
          counters.toolCalls += 1;
          perTool[obj.toolSpec.name] = (perTool[obj.toolSpec.name] ?? 0) + 1;
          try { return await obj.invoke(input); } catch (err) { counters.toolErrors += 1; throw err; }
        };
      }
      const v = Reflect.get(obj, prop, obj);
      return typeof v === 'function' ? (v as (...a: unknown[]) => unknown).bind(obj) : v;
    },
  }));
}

export class Rig implements Body {
  readonly tools: MountedTool[];
  readonly counters = { toolCalls: 0, toolErrors: 0, deaths: 0, reconnects: 0 };
  readonly toolCounts: Record<string, number> = {};
  private events: BodyEvent[] = [];
  private seq = 0;
  private busy = false;
  private inFlight = 0;
  sentinel: SentinelHandle | null = null;
  reflexes: ReflexHandle | null = null;

  constructor(readonly body: LiveBody, opts: { rails?: boolean } = {}) {
    const bot = body.bot;
    // capture_view needs the prismarine viewer (a port + headless Chrome) — not part of the benchmark.
    const base = (allTools(bot) as unknown as MountedTool[]).filter((t) => t.toolSpec.name !== 'capture_view');
    const guarded = guardTools(base, { epoch: body.epoch, detail: () => `The body was ${bot.username} at the time of the drop.` });
    this.tools = countingTools(guarded, this.counters, this.toolCounts);
    if (opts.rails === false) return;
    const note = (text: string) => this.push('note', text);
    const reflex = (name: string, prompt: string) => this.push('reflex', prompt, name);
    body.onEachBot((b) => {
      b.on('death', () => { this.counters.deaths += 1; });
      b.on('messagestr', ((raw: string, position: string, _j: unknown, senderUuid?: string) => {
        const v = classifyMessage({ position, raw, senderUuid, resolveName: () => undefined, selfUsername: b.username, selfUuid: b.player?.uuid, peerBots: [] });
        if (v.kind !== 'system') return;
        const sys = systemNoteFor(v.text, b.username);
        if (sys) note(sys);
      }) as Parameters<typeof b.on<'messagestr'>>[1]);
    });
    body.onRevive = (_b, cause) => { this.counters.reconnects += 1; this.push('revive', `(system) Your connection dropped ("${cause}") and you reconnected. Pathfinding goals were lost.`); };
    body.onGaveUp = (cause) => this.push('gave_up', `(system) body gave up reconnecting: ${cause}`);
    wireBodyRails(body, { note, reflex, log: (t) => console.log(`[bench] ${t}`) });
    registerLegs(body, legs);
    this.sentinel = startSentinel(body, { note, reflex, log: () => {} });
    this.reflexes = startReflexes(body, { deliberateBusy: () => this.busy || this.inFlight > 0, note, legs, log: () => {} });
  }

  private push(kind: string, text: string, name?: string) {
    if (!text?.trim()) return;
    this.events.push({ seq: ++this.seq, kind, text, ...(name ? { name } : {}) });
    if (this.events.length > 500) this.events.shift();
  }

  drainEvents(): BodyEvent[] { const e = this.events; this.events = []; return e; }

  setBusy(b: boolean) { this.busy = b; }

  async call(tool: string, args: Record<string, unknown> = {}): Promise<ToolOutcome> {
    const t = this.tools.find((x) => x.toolSpec.name === tool);
    if (!t) return { ok: false, text: `No tool named '${tool}' on this body.`, data: null };
    this.inFlight += 1;
    try { return toOutcome(await t.invoke(args)); } catch (err) { return errorOutcome(err); } finally { this.inFlight -= 1; }
  }

  blocks(positions: Vec[]) { return probeBlocks(this.body.bot, positions); }

  state(): BodySnapshot {
    const bot = this.body.bot;
    const p = bot.entity?.position;
    const inv = bot.entity ? countCarried(carriedView(bot)) : {};
    return {
      pos: p ? { x: +p.x.toFixed(2), y: +p.y.toFixed(2), z: +p.z.toFixed(2) } : null,
      health: bot.health ?? null,
      food: bot.food ?? null,
      inv,
      foodPortions: inventoryHighlights(Object.entries(inv).map(([name, count]) => ({ name, count }))).foodPortions,
      timeOfDay: bot.time?.timeOfDay ?? null,
      isDay: bot.time?.isDay ?? null,
      threats: this.sentinel?.threats() ?? [],
      counters: { ...this.counters },
    };
  }

  stop() {
    this.sentinel?.stop();
    this.reflexes?.stop();
    this.body.retire();
  }
}
