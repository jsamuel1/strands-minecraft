/**
 * 🦾 The body sidecar — mineflayer stays in Node, the brain moves out.
 *
 * No Python Minecraft client speaks 1.21, so the Python brain (brain/, on
 * strands-py) drives bodies that live here. This process keeps everything that
 * must run next to the socket and must NOT wait for a model:
 *
 *   body.ts      reconnect proxy (signed-chat kicks, mineflayer#3838)
 *   legs.ts      pathfinder arbitration lock
 *   reflexes.ts  300 ms LLM-free spinal cord
 *   sentinel.ts  event-driven senses
 *   chatrail.ts  chat classification (messagestr, not chat)
 *   body-rails   pain / hunger / death
 *   tools/*      the 60 game tools
 *
 * and exposes, per body, on one loopback HTTP port (SIDECAR_PORT, default 3010):
 *
 *   POST /mcp/<body>        MCP streamable HTTP (stateless, JSON responses). The
 *                           tools' JSON schemas are the ones the SDK generates
 *                           from their zod definitions — nothing is re-declared.
 *   GET  /events/<body>?after=N   the event stream: notes, reflex asks, player
 *                           chat, revive/death/gave-up. Seq-numbered, capped;
 *                           the brain long-polls with `wait=ms`.
 *   GET  /state/<body>      the digest block, a Δ-critic snapshot and counters.
 *   POST /blocks/<body>     {positions:[{x,y,z}]} → [{x,y,z,name,solid}] from the body's
 *                           loaded chunks. A harness probe for world-truth post-conditions
 *                           (is the shelter sealed?), NOT an MCP tool: no arm sees it.
 *   POST /busy/<body>       {busy:true|false} — the brain says a turn is in
 *                           flight, so idle reflexes keep off the legs.
 *   POST /bodies            {name, viewDistance?} hire a worker body.
 *   DELETE /bodies/<body>   retire one.
 *   GET  /health            liveness + the body roster.
 *
 * Doctrine carried over, unchanged: notes are EVENTS the brain queues in front
 * of its next turn (never pushed into a live history), peer bots are logged and
 * never answered (they never become a `chat` event), and every rail says its
 * number — the boot line prints the port, bodies and tool count.
 */
import 'dotenv/config';
import http from 'node:http';
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StreamableHTTPServerTransport } from '@modelcontextprotocol/sdk/server/streamableHttp.js';
import { CallToolRequestSchema, ListToolsRequestSchema, type CallToolResult } from '@modelcontextprotocol/sdk/types.js';
import { ImageBlock } from '@strands-agents/sdk';
import { Vec3 } from 'vec3';
import { createLiveBody, type LiveBody } from './body.js';
import { allTools } from './tools/index.js';
import { guardTools } from './tools/connection-guard.js';
import { startSentinel, type SentinelHandle } from './sentinel.js';
import { startReflexes, type ReflexHandle } from './reflexes.js';
import { legs, registerLegs, LegsLock } from './legs.js';
import { classifyMessage, buildChatPrompt, systemNoteFor } from './chatrail.js';
import { wireBodyRails } from './body-rails.js';
import { worldDigest, inventoryHighlights } from './digest.js';
import { carriedView, countCarried } from './tools/helpers.js';
import { checkMemory } from './memcheck.js';
import { startLoopWatch } from './loopwatch.js';
import { stopBody } from './web.js';
import { cfg } from './config.js';

export type SidecarEventKind = 'note' | 'reflex' | 'chat' | 'peer' | 'system' | 'revive' | 'death' | 'gave_up' | 'log';

export interface SidecarEvent {
  seq: number;
  at: number;
  kind: SidecarEventKind;
  text: string;
  /** reflex: the reflex name; chat: the speaker. */
  name?: string;
}

/**
 * The per-body event log. A ring, not an array: a brain that stops polling must
 * not turn the sidecar into issue #44. Seq numbers are monotonic across
 * evictions, so a poller that fell behind learns exactly how much it missed.
 */
export class EventLog {
  private items: SidecarEvent[] = [];
  private seq = 0;
  private waiters = new Set<() => void>();
  dropped = 0;
  constructor(private cap = cfg.sidecar.eventCap, private clock: () => number = Date.now) {}

  push(kind: SidecarEventKind, text: string, name?: string): SidecarEvent | undefined {
    // A wordless event is a wordless note — issue #39's brick, one hop earlier.
    if (!(text ?? '').trim()) return undefined;
    const e: SidecarEvent = { seq: ++this.seq, at: this.clock(), kind, text, ...(name ? { name } : {}) };
    this.items.push(e);
    while (this.items.length > this.cap) { this.items.shift(); this.dropped += 1; }
    for (const w of this.waiters) w();
    this.waiters.clear();
    return e;
  }

  /** Events after `after`, plus how many the ring had already evicted past it. */
  since(after: number): { events: SidecarEvent[]; last: number; missed: number } {
    const events = this.items.filter((e) => e.seq > after);
    const first = this.items[0]?.seq ?? this.seq + 1;
    const missed = Math.max(0, first - after - 1);
    return { events, last: this.seq, missed };
  }

  /** Resolve when something newer than `after` exists, or after `ms`. */
  wait(after: number, ms: number): Promise<void> {
    if (this.seq > after || ms <= 0) return Promise.resolve();
    return new Promise((resolve) => {
      const done = () => { clearTimeout(t); this.waiters.delete(done); resolve(); };
      const t = setTimeout(done, ms);
      t.unref?.();
      this.waiters.add(done);
    });
  }

  get size(): number { return this.items.length; }
  get last(): number { return this.seq; }
}

/** The one conversion from a Strands tool's raw return to MCP content. */
export function toMcpResult(value: unknown): CallToolResult {
  if (value instanceof ImageBlock) {
    const bytes = (value.source as { bytes?: Uint8Array }).bytes;
    if (bytes) return { content: [{ type: 'image', data: Buffer.from(bytes).toString('base64'), mimeType: `image/${value.format}` }] };
  }
  if (typeof value === 'string') return { content: [{ type: 'text', text: value || '(empty result)' }] };
  if (value === undefined || value === null) return { content: [{ type: 'text', text: 'null' }] };
  if (typeof value === 'number' || typeof value === 'boolean') return { content: [{ type: 'text', text: String(value) }] };
  // Objects travel as JSON text — the model reads the same keys the TS brain saw.
  return { content: [{ type: 'text', text: JSON.stringify(value) }] };
}

interface MountedTool {
  toolSpec: { name: string; description?: string; inputSchema?: Record<string, unknown> };
  invoke: (input: unknown) => Promise<unknown>;
}

export interface Rig {
  name: string;
  body: LiveBody;
  tools: MountedTool[];
  events: EventLog;
  sentinel: SentinelHandle | null;
  reflexes: ReflexHandle | null;
  brainBusy: boolean;
  inFlight: number;
  counters: { toolCalls: number; toolErrors: number; deaths: number; reconnects: number; chat: number; reflexAsks: number };
  toolCounts: Record<string, number>;
  startedAt: number;
  primary: boolean;
}

/** List + call handlers over one rig's tools. Pure over the rig: tests hand it fakes. */
export function mcpServerFor(rig: Pick<Rig, 'tools' | 'counters' | 'toolCounts'> & { inFlight: number }): Server {
  const server = new Server({ name: 'strands-minecraft-body', version: '0.1.0' }, { capabilities: { tools: {} } });
  server.setRequestHandler(ListToolsRequestSchema, async () => ({
    tools: rig.tools.map((t) => ({
      name: t.toolSpec.name,
      description: t.toolSpec.description ?? '',
      inputSchema: (t.toolSpec.inputSchema ?? { type: 'object', properties: {} }) as { type: 'object'; [k: string]: unknown },
    })),
  }));
  server.setRequestHandler(CallToolRequestSchema, async (req) => {
    const tool = rig.tools.find((t) => t.toolSpec.name === req.params.name);
    if (!tool) return { isError: true, content: [{ type: 'text', text: `No tool named '${req.params.name}' on this body.` }] };
    rig.counters.toolCalls += 1;
    rig.toolCounts[tool.toolSpec.name] = (rig.toolCounts[tool.toolSpec.name] ?? 0) + 1;
    rig.inFlight += 1;
    try {
      return toMcpResult(await tool.invoke(req.params.arguments ?? {}));
    } catch (err) {
      rig.counters.toolErrors += 1;
      // The error text is the tool's advice ("missing 2 sticks …") — the model reads it.
      return { isError: true, content: [{ type: 'text', text: err instanceof Error ? err.message : String(err) }] };
    } finally {
      rig.inFlight -= 1;
    }
  });
  return server;
}

const PEER_BOTS = (process.env.PEER_BOTS ?? '').split(',').map((s) => s.trim()).filter(Boolean);

/** Build a rig around a live body: tools, senses, reflexes, chat — every rail into `events`. */
export function rigBody(name: string, body: LiveBody, opts: { primary: boolean }): Rig {
  const bot = body.bot;
  const events = new EventLog();
  const base = allTools(bot) as unknown as MountedTool[];
  // The prismarine-viewer is a one-port singleton owned by the primary body.
  const mounted = opts.primary ? base : base.filter((t) => t.toolSpec.name !== 'capture_view');
  const tools = guardTools(mounted, {
    epoch: body.epoch,
    detail: () => `The body was ${bot.username ?? name} at the time of the drop.`,
  });
  const rig: Rig = {
    name, body, tools, events, sentinel: null, reflexes: null, brainBusy: false, inFlight: 0,
    counters: { toolCalls: 0, toolErrors: 0, deaths: 0, reconnects: 0, chat: 0, reflexAsks: 0 },
    toolCounts: {}, startedAt: Date.now(), primary: opts.primary,
  };
  const note = (text: string) => { events.push('note', text); };
  const reflex = (rname: string, prompt: string) => {
    if (process.env.REFLEX_DISABLED === 'true') return;
    rig.counters.reflexAsks += 1;
    events.push('reflex', prompt, rname);
  };

  body.onEachBot((b) => {
    const onMessage = (raw: string, position: string, _json: unknown, senderUuid?: string) => {
      const verdict = classifyMessage({
        position, raw, senderUuid,
        resolveName: (uuid) => Object.values(b.players).find((p) => p?.uuid === uuid)?.username,
        selfUsername: b.username, selfUuid: b.player?.uuid, peerBots: PEER_BOTS,
      });
      if (verdict.kind === 'self') return;
      if (verdict.kind === 'peer') {
        // Logged, NEVER a chat event: bot↔bot chat is the cascade that burns a shared quota.
        console.log(`🤖 [${name}] <${verdict.username}> ${verdict.text}`);
        events.push('peer', verdict.text, verdict.username);
        return;
      }
      if (verdict.kind === 'system') {
        const sys = systemNoteFor(verdict.text, b.username);
        if (sys) note(sys);
        return;
      }
      rig.counters.chat += 1;
      console.log(`💬 [${name}] <${verdict.username}> ${verdict.text}`);
      events.push('chat', buildChatPrompt(verdict.username, verdict.text, verdict.verified), verdict.username);
    };
    b.on('messagestr', onMessage as Parameters<typeof b.on<'messagestr'>>[1]);
    b.on('death', () => { rig.counters.deaths += 1; });
    b.on('kicked', (reason) => console.error(`⛔ [${name}] kicked:`, reason));
    b.on('error', (err) => console.error(`⛔ [${name}] error:`, err.message));
  });
  body.onRevive = (b, cause) => {
    rig.counters.reconnects += 1;
    console.log(`🔁 [${name}] back in the world at ${b.entity?.position}`);
    events.push('revive', `(system) Your connection dropped ("${cause}" — the known signed-chat kick, mineflayer#3838, not your fault) and you reconnected. Pathfinding goals were lost; re-issue movement if you were mid-errand.`);
  };
  body.onGaveUp = (cause) => {
    console.error(`💀 [${name}] body gave up reconnecting (${cause})`);
    events.push('gave_up', `(system) body gave up reconnecting: ${cause}`);
    rig.sentinel?.stop();
    rig.reflexes?.stop();
    // Corpse guard: the primary body dying takes the sidecar down, so a
    // supervisor (or the benchmark runner) brings a FRESH process.
    if (opts.primary) setTimeout(() => process.exit(1), 1_500).unref();
  };

  wireBodyRails(body, {
    note,
    reflex,
    deathWaypoint: opts.primary ? 'last_death' : `${name.toLowerCase()}_death`,
    log: (t) => console.log(`[${name}] ${t}`),
  });
  // Primary legs are the process singleton the tools look up; a worker gets its own.
  const lock = opts.primary ? legs : new LegsLock();
  registerLegs(body, lock);
  rig.sentinel = opts.primary
    ? startSentinel(body, { note, reflex, log: (who, text) => console.log(`👂 [${name}] ${who}: ${text}`) })
    : null;
  rig.reflexes = startReflexes(body, {
    deliberateBusy: () => rig.brainBusy || rig.inFlight > 0,
    note,
    legs: lock,
    log: (_who, text) => console.log(`⚡ [${name}] ${text}`),
  }, { idleModes: opts.primary });
  return rig;
}

/** The numbers the brain's digest, Δ critic and the benchmark all read. */
export function rigState(rig: Rig) {
  const bot = rig.body.bot;
  const p = bot.entity?.position;
  const inv = bot.entity ? countCarried(carriedView(bot)) : {};
  return {
    name: rig.name,
    username: bot.username,
    connected: !!bot.entity && !!bot.player,
    epoch: rig.body.epoch(),
    pos: p ? { x: +p.x.toFixed(2), y: +p.y.toFixed(2), z: +p.z.toFixed(2) } : null,
    health: bot.health ?? null,
    food: bot.food ?? null,
    inv,
    foodPortions: inventoryHighlights(Object.entries(inv).map(([n, c]) => ({ name: n, count: c }))).foodPortions,
    timeOfDay: bot.time?.timeOfDay ?? null,
    isDay: bot.time?.isDay ?? null,
    dimension: bot.game?.dimension ?? null,
    digest: bot.entity ? worldDigest(bot, {
      threats: () => rig.sentinel?.threats() ?? [],
      reflexRecent: (n) => rig.reflexes?.recent(n) ?? [],
    }) : '(not spawned)',
    threats: rig.sentinel?.threats() ?? [],
    counters: rig.counters,
    toolCounts: rig.toolCounts,
    events: { last: rig.events.last, pending: rig.events.size, dropped: rig.events.dropped },
    busy: { brain: rig.brainBusy, toolsInFlight: rig.inFlight },
    upMs: Date.now() - rig.startedAt,
  };
}

const readBody = (req: http.IncomingMessage): Promise<unknown> => new Promise((resolve, reject) => {
  const chunks: Buffer[] = [];
  req.on('data', (c: Buffer) => chunks.push(c));
  req.on('end', () => {
    const s = Buffer.concat(chunks).toString('utf8');
    if (!s) return resolve(undefined);
    try { resolve(JSON.parse(s)); } catch (e) { reject(e); }
  });
  req.on('error', reject);
});

const json = (res: http.ServerResponse, status: number, body: unknown) => {
  res.writeHead(status, { 'content-type': 'application/json' });
  res.end(JSON.stringify(body));
};

export interface SidecarOptions {
  port?: number;
  host?: string;
  /** Build a body — injected so tests run without a server. */
  spawn?: (name: string, viewDistance?: number | string) => Promise<LiveBody>;
}

/** The HTTP front door over a roster of rigs. Returns the server and roster. */
/** What occupies each position, and whether it blocks movement (a full collision box). */
export function probeBlocks(
  bot: { blockAt(p: Vec3): { name: string; boundingBox: string } | null },
  positions: Array<{ x: number; y: number; z: number }>,
): Array<{ x: number; y: number; z: number; name: string; solid: boolean }> {
  return positions.map((p) => {
    const b = bot.blockAt(new Vec3(Math.floor(p.x), Math.floor(p.y), Math.floor(p.z)));
    return { x: p.x, y: p.y, z: p.z, name: b?.name ?? 'unloaded', solid: b?.boundingBox === 'block' };
  });
}

export function startSidecarServer(rigs: Map<string, Rig>, opts: SidecarOptions = {}) {
  const spawn = opts.spawn ?? ((name, viewDistance) => createLiveBody({
    username: name,
    ...(viewDistance !== undefined ? { viewDistance: viewDistance as number } : {}),
  }));
  const server = http.createServer(async (req, res) => {
    const url = new URL(req.url ?? '/', 'http://sidecar');
    const [, route, name] = url.pathname.split('/');
    const rig = name ? rigs.get(name) : undefined;
    try {
      if (route === 'health') return json(res, 200, { ok: true, bodies: [...rigs.keys()] });
      if (route === 'bodies' && req.method === 'POST') {
        const b = (await readBody(req)) as { name?: string; viewDistance?: number | string } | undefined;
        if (!b?.name) return json(res, 400, { error: 'body needs a name' });
        if (rigs.has(b.name)) return json(res, 409, { error: `body ${b.name} already exists` });
        const live = await spawn(b.name, b.viewDistance ?? cfg.fleet.workerViewDistance);
        rigs.set(b.name, rigBody(b.name, live, { primary: false }));
        return json(res, 201, { name: b.name });
      }
      if (!rig) return json(res, 404, { error: `no body '${name ?? ''}' (have: ${[...rigs.keys()].join(', ')})` });
      if (route === 'bodies' && req.method === 'DELETE') {
        if (rig.primary) return json(res, 400, { error: 'the primary body is retired by stopping the sidecar' });
        rig.sentinel?.stop(); rig.reflexes?.stop(); rig.body.retire();
        try { rig.body.bot.quit(); } catch { /* already gone */ }
        rigs.delete(rig.name);
        return json(res, 200, { retired: rig.name });
      }
      if (route === 'state') return json(res, 200, rigState(rig));
      if (route === 'events') {
        const after = Number(url.searchParams.get('after') ?? 0) || 0;
        const wait = Math.min(Number(url.searchParams.get('wait') ?? 0) || 0, cfg.sidecar.maxPollMs);
        await rig.events.wait(after, wait);
        return json(res, 200, rig.events.since(after));
      }
      if (route === 'blocks' && req.method === 'POST') {
        const b = (await readBody(req)) as { positions?: Array<{ x: number; y: number; z: number }> } | undefined;
        return json(res, 200, probeBlocks(rig.body.bot, (b?.positions ?? []).slice(0, cfg.sidecar.blockProbeMax)));
      }
      if (route === 'busy' && req.method === 'POST') {
        const b = (await readBody(req)) as { busy?: boolean } | undefined;
        rig.brainBusy = !!b?.busy;
        return json(res, 200, { busy: rig.brainBusy });
      }
      if (route === 'stop' && req.method === 'POST') return json(res, 200, { stopped: stopBody(rig.body.bot) });
      if (route === 'mcp') {
        // Stateless: a fresh server + transport per request (the SDK's rule for
        // sessionIdGenerator: undefined). Tool objects are shared, so every
        // request drives the same body.
        const mcp = mcpServerFor(rig);
        const transport = new StreamableHTTPServerTransport({ sessionIdGenerator: undefined, enableJsonResponse: true });
        res.on('close', () => { void transport.close(); void mcp.close(); });
        await mcp.connect(transport);
        const body = req.method === 'POST' ? await readBody(req) : undefined;
        await transport.handleRequest(req, res, body);
        return;
      }
      return json(res, 404, { error: `unknown route ${url.pathname}` });
    } catch (err) {
      if (!res.headersSent) json(res, 500, { error: err instanceof Error ? err.message : String(err) });
    }
  });
  server.listen(opts.port ?? cfg.sidecar.port, opts.host ?? cfg.sidecar.host);
  return server;
}

const main = async () => {
  console.log('🦾 strands-minecraft body sidecar — connecting…');
  const mem = checkMemory();
  if (mem.level !== 'ok') console.log(`${mem.level === 'warn' ? '🚨' : '⚠️ '} ${mem.text}`);
  const body = await createLiveBody();
  const name = body.bot.username;
  console.log(`✅ Spawned as ${name} at ${body.bot.entity.position}`);
  const rigs = new Map<string, Rig>([[name, rigBody(name, body, { primary: true })]]);
  const loopWatch = startLoopWatch({ emit: (v) => console.log(`🩺 ${v.text}`) });
  const server = startSidecarServer(rigs);
  console.log(`🦾 sidecar on http://${cfg.sidecar.host}:${cfg.sidecar.port} — body '${name}', ${rigs.get(name)!.tools.length} tools at /mcp/${name}, events at /events/${name} (ring ${cfg.sidecar.eventCap})`);
  const shutdown = () => {
    loopWatch.stop();
    for (const r of rigs.values()) { r.sentinel?.stop(); r.reflexes?.stop(); r.body.retire(); try { r.body.bot.quit(); } catch { /* closing */ } }
    server.close();
    setTimeout(() => process.exit(0), 500).unref();
  };
  process.on('SIGTERM', shutdown);
  process.on('SIGINT', shutdown);
};

if (process.argv[1] && /sidecar\.(ts|js)$/.test(process.argv[1])) {
  main().catch((err) => {
    console.error('sidecar failed to start:', err instanceof Error ? err.message : err);
    process.exit(1);
  });
}
