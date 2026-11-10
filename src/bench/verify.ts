/**
 * Did the action actually happen? A post-condition per macro-action, read from
 * the WORLD — ported from v3 (brain/src/mcbrain/verify.py) without changes in
 * meaning.
 *
 * A tool returning without throwing only means the request was answered. The
 * tools report partial or no effect as ordinary text ("placed 0/25 … FAILED
 * 24", "COULD NOT REACH", "gave up after 20s"). So the verdict is computed from
 * world state before and after the step:
 *
 * - `done`: the post-condition held (the bag gained the item, the table is in reach, …).
 * - `in_progress`: a multi-step effect moved forward but is not complete (a shelter part-built).
 * - `no_effect`: no change toward the post-condition, or a tool call threw.
 * - `unverified`: no checkable post-condition.
 * - `waited`: `wait` did nothing, by design.
 *
 * `reason` is WHY, in the tool's own words when it gave one, otherwise what the
 * check found. The verdict informs the model; it never removes an option.
 */
import type { Body, Vec } from './body.js';

export const VERDICTS = ['done', 'in_progress', 'no_effect', 'unverified', 'waited'] as const;
export type VerdictName = (typeof VERDICTS)[number];

export const FOOD_ITEMS = ['beef', 'porkchop', 'mutton', 'chicken', 'rabbit', 'cooked_beef', 'cooked_porkchop', 'cooked_mutton', 'cooked_chicken'];

export interface Verdict { verdict: VerdictName; check: string; reason: string }

export interface Snapshot {
  inv: Record<string, number>;
  pos: Vec;
  food: number;
  tablesNear: number;
  /** hostile name → nearest distance */
  hostiles: Record<string, number>;
  /** Plan positions that are solid: the shelter's world truth. Keys "x,y,z". */
  solid: Set<string>;
  t: number;
}

const key = (p: Vec) => `${p.x},${p.y},${p.z}`;

export async function takeSnapshot(body: Body, hostile: readonly string[], plan: Vec[] | null, t: number): Promise<Snapshot> {
  const s = body.state();
  const tables = await body.call('find_blocks', { name: 'crafting_table', maxDistance: 4, count: 4 });
  const around = await body.call('look_around', { radius: 24 });
  const ents = (around.data && typeof around.data === 'object' && Array.isArray((around.data as { entities?: unknown[] }).entities))
    ? (around.data as { entities: Array<Record<string, unknown>> }).entities : [];
  const near: Record<string, number> = {};
  for (const e of ents) {
    const name = String(e?.name ?? '');
    if (hostile.includes(name)) near[name] = Math.min(Number(e.distance ?? 99), near[name] ?? 99);
  }
  const solid = new Set(plan ? body.blocks(plan).filter((b) => b.solid).map(key) : []);
  return {
    inv: { ...s.inv },
    pos: s.pos ?? { x: 0, y: 0, z: 0 },
    food: Number(s.food ?? 0),
    tablesNear: Array.isArray(tables.data) ? tables.data.length : 0,
    hostiles: near,
    solid,
    t,
  };
}

/** The first thing a tool says went wrong, e.g. "is occupied by stone", "COULD NOT REACH …". */
const REFUSAL = new RegExp(
  "(is occupied by [\\w ]+|COULD NOT REACH[^.|]*|gave up[^.|]*|out of time[^.|]*|Missing materials[^|]*|No '[\\w' /]+' found[^.|]*"
  + '|Cannot [^.|]*|can\'t [^.|]*|not enough [^.|]*|was NOT placed[^.|]*|No solid neighbor[^.|]*)',
  'i',
);

/** The tool's own first refusal, coordinates stripped, or '' when it gave none. */
export function toolReason(outcome: string): string {
  const m = REFUSAL.exec(outcome ?? '');
  if (!m) return '';
  const text = m[1].replace(/\s*\(-?\d+(?:\.\d+)?, -?\d+(?:\.\d+)?, -?\d+(?:\.\d+)?\)/g, '');
  return text.replace(/\s+/g, ' ').replace(/\s+([:,;])/g, '$1').replace(/^[ :]+|[ :]+$/g, '').slice(0, 160);
}

function gained(a: Snapshot, b: Snapshot, pred: (n: string) => boolean = () => true): Record<string, number> {
  const out: Record<string, number> = {};
  for (const [n, c] of Object.entries(b.inv)) if (c > (a.inv[n] ?? 0) && pred(n)) out[n] = c - (a.inv[n] ?? 0);
  return out;
}

const moved = (a: Snapshot, b: Snapshot) => Math.hypot(a.pos.x - b.pos.x, a.pos.z - b.pos.z);
const fmt = (g: Record<string, number>) => Object.entries(g).map(([n, c]) => `+${c} ${n}`).join(', ') || 'nothing';
const yes = (ok: boolean): VerdictName => (ok ? 'done' : 'no_effect');
const v = (verdict: VerdictName, check: string, reason = ''): Verdict => ({ verdict, check, reason });

export interface ActionMeta {
  gain?: readonly string[];
  mob?: string;
  minMove?: number;
  item?: string;
  plan?: Vec[];
}

export function shelterVerdict(meta: ActionMeta, a: Snapshot, b: Snapshot, outcome: string, sPerBlock: number | null): Verdict {
  const plan = (meta.plan ?? []).map(key);
  const total = plan.length;
  const before = plan.filter((p) => a.solid.has(p)).length;
  const after = plan.filter((p) => b.solid.has(p)).length;
  const open = total - after;
  const check = `${after}/${total} shelter positions solid (was ${before})`;
  if (total && open === 0) return v('done', `${check} — sealed`);
  const eta = sPerBlock ? `, ~${(open * sPerBlock).toFixed(0)}s to finish at ${sPerBlock.toFixed(1)}s/block` : '';
  if (after > before) return v('in_progress', `${check}, ${open} open${eta}`, toolReason(outcome));
  return v('no_effect', `${check}, ${open} open${eta}`, toolReason(outcome) || 'no plan position became solid');
}

/** The verdict for one executed macro-action. Pure: facts in, verdict out. */
export function checkAction(action: string, meta: ActionMeta, a: Snapshot, b: Snapshot, outcome = '', sPerBlock: number | null = null): Verdict {
  const why = toolReason(outcome);
  if (action === 'wait') return v('waited', 'wait does nothing by design');
  if (action === 'build_shelter') return shelterVerdict(meta, a, b, outcome, sPerBlock);
  if (action.startsWith('craft_')) {
    const item = action.slice('craft_'.length);
    const n = (b.inv[item] ?? 0) - (a.inv[item] ?? 0);
    return v(yes(n > 0), `bag ${item} ${a.inv[item] ?? 0}→${b.inv[item] ?? 0}`, n > 0 ? '' : why || `no ${item} appeared`);
  }
  if (action === 'place_crafting_table') {
    const ok = b.tablesNear > a.tablesNear;
    return v(yes(ok), `crafting tables within 4 blocks ${a.tablesNear}→${b.tablesNear}`, ok ? '' : why || 'no new table in reach');
  }
  if (action === 'walk_to_table') {
    const ok = b.tablesNear > 0;
    return v(yes(ok), `crafting tables within 4 blocks after: ${b.tablesNear}`, ok ? '' : why || 'still out of reach');
  }
  if (action.startsWith('explore_')) {
    const d = moved(a, b);
    const need = meta.minMove ?? 1;
    return v(yes(d >= need), `moved ${d.toFixed(1)} blocks, need ≥${need.toFixed(0)}`, d >= need ? '' : why || 'barely moved');
  }
  if (action === 'flee_hostile') {
    const mob = String(meta.mob);
    const da = a.hostiles[mob] ?? 99;
    const db = b.hostiles[mob] ?? 99;
    return v(yes(db > da), `${mob} distance ${da.toFixed(1)}→${db.toFixed(1)}`, db > da ? '' : why || `the ${mob} is no farther`);
  }
  if (action === 'fight_hostile') {
    const mob = String(meta.mob);
    const fa = a.hostiles[mob];
    const fb = b.hostiles[mob];
    const gone = fb === undefined || (fa !== undefined && fb > fa + 4);
    return v(yes(gone), `${mob} nearest ${fa ?? 'None'}→${fb ?? 'None'} (gone = done)`, gone ? '' : why || `the ${mob} is still there`);
  }
  if (action === 'eat') {
    const ok = b.food > a.food;
    return v(yes(ok), `food ${a.food.toFixed(0)}→${b.food.toFixed(0)}`, ok ? '' : why || 'food did not rise');
  }
  if (action === 'hunt_animal') {
    const g = gained(a, b, (n) => FOOD_ITEMS.includes(n));
    const ok = Object.keys(g).length > 0;
    return v(yes(ok), `food items gained: ${fmt(g)}`, ok ? '' : why || 'no food dropped into the bag');
  }
  if (/^(fell_tree|mine_|dig_down|collect_drops)/.test(action)) {
    const want = meta.gain;
    const g = gained(a, b, want ? (n) => want.includes(n) : () => true);
    const ok = Object.keys(g).length > 0;
    const counting = want ? ` (counting ${want.join(', ')})` : '';
    return v(yes(ok), `gained ${fmt(g)}${counting}`, ok ? '' : why || 'nothing reached the bag');
  }
  return v('unverified', 'no post-condition defined for this action');
}
