/**
 * The body as the benchmark sees it — an interface, so the policy, verifier and
 * runner are pure over it and tests hand them a fake.
 *
 * v3 reached the body over MCP + HTTP (a Node sidecar driven from Python). Here
 * the body is in-process: `rig.ts` implements this against the same LiveBody,
 * the same 60 tools (`allTools`) and the same reflex/sentinel rails as
 * `npm start`. A tool's return value is the same object an MCP client would
 * have decoded from JSON text, and a thrown tool is `ok: false` (MCP isError).
 */

export interface ToolOutcome {
  /** Transport: the call returned without throwing. NOT whether it worked (verify.ts decides that). */
  ok: boolean;
  text: string;
  data: unknown;
}

export const short = (o: ToolOutcome, limit = 400) => {
  const t = o.text.replace(/\s+/g, ' ').trim();
  return t.length <= limit ? t : `${t.slice(0, limit - 1)}…`;
};

export interface Vec { x: number; y: number; z: number }

export interface BodyEvent { seq: number; kind: string; text: string; name?: string }

/** The facts the policy and the goal checks read, each step. */
export interface BodySnapshot {
  pos: Vec | null;
  health: number | null;
  food: number | null;
  inv: Record<string, number>;
  /** The body's own food portions (digest.inventoryHighlights). */
  foodPortions: number;
  timeOfDay: number | null;
  isDay: boolean | null;
  threats: string[];
  counters: { toolCalls: number; toolErrors: number; deaths: number; reconnects: number };
}

export interface Body {
  state(): BodySnapshot;
  call(tool: string, args?: Record<string, unknown>): Promise<ToolOutcome>;
  /** World-truth probe (NOT a tool; no arm sees it): what occupies each position. */
  blocks(positions: Vec[]): Array<Vec & { name: string; solid: boolean }>;
  /** Body events since the last drain (notes, reflexes, revive, death). */
  drainEvents(): BodyEvent[];
  /** A brain turn is in flight: idle reflexes keep off the legs. */
  setBusy(busy: boolean): void;
}

/** A tool's raw return → outcome. Strings stay strings; objects travel as JSON text, as over MCP. */
export function toOutcome(value: unknown): ToolOutcome {
  if (typeof value === 'string') {
    let data: unknown = value;
    try { data = JSON.parse(value); } catch { /* plain text */ }
    return { ok: true, text: value || '(empty result)', data };
  }
  if (value === undefined || value === null) return { ok: true, text: 'null', data: null };
  if (typeof value === 'number' || typeof value === 'boolean') return { ok: true, text: String(value), data: value };
  return { ok: true, text: JSON.stringify(value), data: value };
}

export function errorOutcome(err: unknown): ToolOutcome {
  const text = err instanceof Error ? err.message : String(err);
  return { ok: false, text, data: text };
}
