/**
 * The System One brain: a native SDK DecisionModel picks the next macro-action;
 * code executes it; the world says whether it worked.
 *
 * Ported from v3 (brain/src/mcbrain/policy.py) with the same alphabet, the same
 * candidate rules, the same decision-state projection and the same one-ask
 * shape (next_action + last_step_succeeded). What changed is the model seam:
 * v3 called `strands.experimental.decisions` `Choice`/`YesNo` from a fork of
 * strands-py; this calls `DecisionModel.ask(state, questions)` from the
 * released @strands-agents/sdk.
 *
 * - Facts are code: `observe()` reads the body and `candidates()` builds only
 *   LEGAL, GROUNDED macro-actions.
 * - Judgment is the model: one choice over those ids.
 * - Execution is code: each macro-action is one to a few tool calls on the same
 *   60 tools the LLM agent arm calls.
 */
import { DecisionModel, Uncertain, type Question } from '@strands-agents/sdk';
import { short, type Body, type BodyEvent, type BodySnapshot, type ToolOutcome, type Vec } from './body.js';
import { checkAction, takeSnapshot, type ActionMeta } from './verify.js';
import { SelfHostedDecisionModel, type BenchQuestion } from './decider.js';

export const LOGS = ['oak_log', 'birch_log', 'spruce_log', 'jungle_log', 'acacia_log', 'dark_oak_log', 'mangrove_log', 'cherry_log'];
export const STONE = ['stone', 'cobblestone', 'deepslate', 'andesite', 'diorite', 'granite'];
export const COAL = ['coal_ore', 'deepslate_coal_ore'];
export const IRON = ['iron_ore', 'deepslate_iron_ore'];
export const FOOD_MOBS = ['cow', 'pig', 'sheep', 'chicken', 'rabbit'];
export const HOSTILE = ['zombie', 'skeleton', 'spider', 'creeper', 'husk', 'drowned', 'stray', 'witch', 'cave_spider', 'slime', 'phantom', 'enderman'];
const PICKS = ['netherite_pickaxe', 'diamond_pickaxe', 'iron_pickaxe', 'stone_pickaxe', 'wooden_pickaxe'];
const SWORDS = ['netherite_sword', 'diamond_sword', 'iron_sword', 'stone_sword', 'wooden_sword'];
const AXES = ['netherite_axe', 'diamond_axe', 'iron_axe', 'stone_axe', 'wooden_axe'];
const STONE_DROPS = ['cobblestone', 'cobbled_deepslate', 'andesite', 'diorite', 'granite'];
const COAL_DROPS = ['coal'];
const IRON_DROPS = ['raw_iron', 'iron_ore', 'deepslate_iron_ore'];
const CRAFT_ORDER = ['crafting_table', 'wooden_pickaxe', 'wooden_sword', 'stone_pickaxe', 'stone_sword', 'furnace', 'torch', 'wooden_axe'];
const SHELTER_BLOCKS = ['cobblestone', 'dirt', 'cobbled_deepslate'];

/** v3 PolicyCfg defaults (brain/src/mcbrain/config.py), env-overridable with the same names. */
const num = (k: string, d: number) => { const v = Number(process.env[k]); return Number.isFinite(v) && process.env[k] !== '' && process.env[k] !== undefined ? v : d; };
export const policyCfg = {
  maxCandidates: num('POLICY_CANDIDATES', 12),
  searchRadius: num('POLICY_SEARCH_RADIUS', 48),
  exploreMinMove: num('POLICY_EXPLORE_MIN_MOVE', 8),
  waitS: num('POLICY_WAIT_S', 5),
};

/** How much decision state one model is shown — its operating point, recorded per episode. */
export interface ContextProfile {
  name: string;
  stateChars: number;
  historySteps: number;
  eventChars: number;
  maxEvents: number;
  previousStates: boolean;
  outcomeChars: number;
}

/**
 * v3's profiles (config.py). `decider` is v3's hobson profile: strands-decider
 * serves a 4,096-token window; 7,500 chars of state is ≤ ~3.6k Qwen3.5 tokens at
 * the measured 2.11–2.74 chars/token, leaving room for a 12-option question.
 */
export const PROFILES: Readonly<Record<string, ContextProfile>> = Object.freeze({
  decider: { name: 'decider', stateChars: num('POLICY_STATE_CHARS_DECIDER', 7_500), historySteps: num('POLICY_HISTORY_STEPS_DECIDER', 12), eventChars: 120, maxEvents: 6, previousStates: true, outcomeChars: 240 },
  jev: { name: 'jev', stateChars: num('POLICY_STATE_CHARS_JEV', 48_000), historySteps: num('POLICY_HISTORY_STEPS_JEV', 64), eventChars: 120, maxEvents: 6, previousStates: true, outcomeChars: 240 },
  clm: { name: 'clm', stateChars: 6_000, historySteps: 6, eventChars: 60, maxEvents: 6, previousStates: false, outcomeChars: 120 },
});

export interface Cluster { position: Vec; distance: number; veinSize: number; exposed: boolean; block: string }
export interface Entity { name?: string; distance?: number; position?: Vec }

export interface Observation {
  state: BodySnapshot;
  trees: Cluster[];
  stone: Cluster[];
  coal: Cluster[];
  iron: Cluster[];
  tables: Cluster[];
  furnaces: Cluster[];
  entities: Entity[];
  drops: number;
  standingOn: string;
  events: string[];
}

const clusters = (o: ToolOutcome): Cluster[] =>
  Array.isArray(o.data) ? (o.data as unknown[]).filter((c): c is Cluster => !!c && typeof c === 'object' && 'position' in c) : [];

const BODY_EVENT_KINDS = ['note', 'reflex', 'revive', 'death', 'gave_up', 'system'];
const COORDS = /\s*\(-?\d+(?:\.\d+)?, -?\d+(?:\.\d+)?, -?\d+(?:\.\d+)?\)/g;
const ADVANCEMENT = /has made the advancement \[([^\]]+)\]/;

export const advancement = (text: string): string | null => ADVANCEMENT.exec(text)?.[1] ?? null;

/** Body events, deduplicated (×N) and capped, oldest first. Advancements → `advancement: <name>`. */
export function formatEvents(raw: BodyEvent[], eventChars: number, maxEvents: number): string[] {
  const counts = new Map<string, number>();
  for (const e of raw) {
    if (!BODY_EVENT_KINDS.includes(e.kind)) continue;
    const adv = advancement(e.text);
    const text = adv ? `advancement: ${adv}` : e.text.replace(/\s+/g, ' ').replace(COORDS, '').trim().slice(0, eventChars);
    const k = e.kind === 'reflex' && e.name ? `${e.name}: ${text}` : text;
    counts.set(k, (counts.get(k) ?? 0) + 1);
  }
  return [...counts].map(([k, n]) => (n === 1 ? k : `${k} ×${n}`)).slice(-maxEvents);
}

export async function observe(body: Body, profile: ContextProfile): Promise<Observation> {
  const state = body.state();
  const r = policyCfg.searchRadius;
  const trees = clusters(await body.call('find_blocks', { name: LOGS.join(','), maxDistance: r, count: 3 }));
  const stone = clusters(await body.call('find_blocks', { name: STONE.join(','), maxDistance: Math.min(r, 24), count: 3 }));
  const ores = clusters(await body.call('find_blocks', { name: [...COAL, ...IRON].join(','), maxDistance: Math.min(r, 32), count: 6 }));
  const stations = clusters(await body.call('find_blocks', { name: 'crafting_table,furnace', maxDistance: 24, count: 4 }));
  const around = await body.call('look_around', { radius: 16 });
  const ad = (around.data && typeof around.data === 'object' ? around.data : {}) as { entities?: Entity[]; standingOn?: string };
  const entities = (ad.entities ?? []).filter((e) => e && typeof e === 'object');
  return {
    state,
    trees,
    stone,
    coal: ores.filter((c) => COAL.includes(c.block)).slice(0, 2),
    iron: ores.filter((c) => IRON.includes(c.block)).slice(0, 2),
    tables: stations.filter((c) => c.block === 'crafting_table'),
    furnaces: stations.filter((c) => c.block === 'furnace'),
    entities,
    // Only drops collect_ground_items can reach (radius 8).
    drops: entities.filter((e) => e.name === 'item' && Number(e.distance ?? 99) <= 8).length,
    standingOn: String(ad.standingOn ?? ''),
    events: formatEvents(body.drainEvents(), profile.eventChars, profile.maxEvents),
  };
}

const has = (inv: Record<string, number>, names: string[]) => names.find((n) => (inv[n] ?? 0) > 0) ?? null;
const count = (inv: Record<string, number>, pred: (n: string) => boolean) => Object.entries(inv).reduce((t, [n, c]) => t + (pred(n) ? c : 0), 0);
export const fmtPos = (p: Vec) => `(${Math.trunc(p.x)}, ${Math.trunc(p.y)}, ${Math.trunc(p.z)})`;
const nearTable = (ob: Observation) => ob.tables.find((t) => Number(t.distance ?? 99) <= 4) ?? null;
const xyz = (p: Vec) => ({ x: p.x, y: p.y, z: p.z });
const pos = (ob: Observation) => ob.state.pos ?? { x: 0, y: 0, z: 0 };

/** A conservative, code-owned affordability check (craft_item itself is the final word). */
export function craftable(ob: Observation, item: string): boolean {
  const inv = ob.state.inv;
  const logs = count(inv, (n) => n.endsWith('_log'));
  const planks = count(inv, (n) => n.endsWith('_planks')) + 4 * logs;
  const sticks = (inv.stick ?? 0) + 2 * planks;
  const cobble = (inv.cobblestone ?? 0) + (inv.cobbled_deepslate ?? 0);
  const table = nearTable(ob) !== null;
  const need: Record<string, boolean> = {
    crafting_table: planks >= 4,
    stick: planks >= 2,
    wooden_pickaxe: table && planks >= 5,
    wooden_sword: table && planks >= 3,
    wooden_axe: table && planks >= 5,
    stone_pickaxe: table && cobble >= 3 && sticks >= 2,
    stone_sword: table && cobble >= 2 && sticks >= 1,
    furnace: table && cobble >= 8,
    torch: (inv.coal ?? 0) + (inv.charcoal ?? 0) >= 1 && sticks >= 1,
  };
  return need[item] ?? false;
}

export type ToolStep = [string, Record<string, unknown>];

export interface MacroAction {
  id: string;
  description: string;
  steps: ToolStep[];
  /** Placement that searches a ring of spots around the body (v3 place_near). */
  placeRing?: boolean;
  /** Keep going past a failed step (a shaft meets air pockets); the bag decides success. */
  tolerant?: boolean;
  /** Seconds of game time this action spends after its steps (only `wait`). */
  waitS?: number;
  meta: ActionMeta;
}

/** Placement ring: the 8 neighbours at foot level, then one block up. */
export const RING: Array<[number, number, number]> = [0, 1].flatMap((dy) =>
  ([[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, -1], [1, -1], [-1, 1]] as const).map(([dx, dz]) => [dx, dy, dz] as [number, number, number]));

export async function placeNear(body: Body, item: string): Promise<ToolOutcome> {
  const p = body.state().pos ?? { x: 0, y: 0, z: 0 };
  const [x0, y0, z0] = [Math.floor(p.x), Math.floor(p.y), Math.floor(p.z)];
  let last: ToolOutcome = { ok: false, text: 'no spot tried', data: null };
  for (const [dx, dy, dz] of RING) {
    last = await body.call('place_block', { item, x: x0 + dx, y: y0 + dy, z: z0 + dz });
    if (last.ok) return last;
  }
  return last;
}

export const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

/** Run the steps in order; stop at the first failure unless tolerant. */
export async function runAction(a: MacroAction, body: Body, wait: (ms: number) => Promise<void> = sleep): Promise<ToolOutcome[]> {
  if (a.placeRing) return [await placeNear(body, 'crafting_table')];
  const outs: ToolOutcome[] = [];
  for (const [tool, args] of a.steps) {
    const out = await body.call(tool, args);
    outs.push(out.ok || !a.tolerant ? out : { ok: true, text: `(skipped: ${short(out, 80)})`, data: null });
    if (!out.ok && !a.tolerant) break;
  }
  if (a.waitS) await wait(a.waitS * 1000);
  return outs;
}

const COLLECT: ToolStep = ['collect_ground_items', { radius: 8 }];

function digColumn(p: Vec, depth: number): ToolStep[] {
  const [x, y, z] = [Math.floor(p.x) + 1, Math.floor(p.y), Math.floor(p.z)];
  return Array.from({ length: depth + 1 }, (_, d) => ['dig_block', { x, y: y - d, z }] as ToolStep);
}

function treeActions(ob: Observation, axe: string | null): MacroAction[] {
  const equip: ToolStep[] = axe ? [['equip_item', { item: axe }]] : [];
  return ob.trees.slice(0, 2).map((t, i) => ({
    id: `fell_tree_${i}`,
    description: `Fell the ${t.block} tree (${t.veinSize} logs) at ${fmtPos(t.position)}, ${t.distance} blocks away, and pick up the logs.`,
    steps: [...equip, ['dig_vein', { ...xyz(t.position), maxBlocks: 24 }], COLLECT],
    meta: { gain: LOGS },
  }));
}

function craftActions(ob: Observation): MacroAction[] {
  const out: MacroAction[] = [];
  for (const item of CRAFT_ORDER) {
    if ((ob.state.inv[item] ?? 0) && item !== 'torch') continue;
    if (item === 'crafting_table' && nearTable(ob)) continue;
    if (!craftable(ob, item)) continue;
    const n = item === 'torch' ? 4 : 1;
    out.push({
      id: `craft_${item}`,
      description: `Craft ${n} ${item} (intermediates such as planks and sticks are chained automatically).`,
      steps: [['craft_item', { item, count: n }]],
      meta: {},
    });
  }
  return out;
}

function tableActions(ob: Observation): MacroAction[] {
  if (nearTable(ob)) return [];
  const out: MacroAction[] = [];
  if (ob.state.inv.crafting_table) {
    out.push({ id: 'place_crafting_table', description: 'Place a crafting table next to you so 3×3 recipes (pickaxes, swords, furnace) become craftable here.', steps: [], placeRing: true, meta: {} });
  }
  if (ob.tables.length) {
    const t = ob.tables[0];
    out.push({ id: 'walk_to_table', description: `Walk to the crafting table at ${fmtPos(t.position)} (${t.distance} blocks) to use 3×3 recipes.`, steps: [['go_to', { ...xyz(t.position), range: 2 }]], meta: {} });
  }
  return out;
}

function miningActions(ob: Observation, pick: string | null): MacroAction[] {
  if (!pick) return [];
  const equip: ToolStep = ['equip_item', { item: pick }];
  const out: MacroAction[] = [];
  const exposed = ob.stone.filter((s) => s.exposed);
  if (exposed.length) {
    const s = exposed[0];
    out.push({
      id: 'mine_stone',
      description: `Mine the exposed ${s.block} patch (${s.veinSize} blocks) at ${fmtPos(s.position)} with the ${pick} and pick up the cobblestone.`,
      steps: [equip, ['dig_vein', { ...xyz(s.position), also: 'cobblestone,andesite,diorite,granite', maxBlocks: 12 }], COLLECT],
      meta: { gain: STONE_DROPS },
    });
  } else if (ob.stone.length) {
    out.push({
      id: 'dig_down_to_stone',
      description: `Dig straight down through the ground under you with the ${pick} to reach the buried stone ~${Math.trunc(pos(ob).y - ob.stone[0].position.y)} blocks below, collecting what drops.`,
      steps: [equip, ...digColumn(pos(ob), 6), COLLECT],
      tolerant: true,
      meta: {},
    });
  }
  for (const [kind, veins] of [['coal', ob.coal], ['iron', ob.iron]] as const) {
    if (kind === 'iron' && pick === 'wooden_pickaxe') continue;
    for (const vn of veins.filter((x) => x.exposed).slice(0, 1)) {
      out.push({
        id: `mine_${kind}`,
        description: `Mine the ${vn.block} vein (${vn.veinSize} blocks, exposed=${vn.exposed ? 'True' : 'False'}) at ${fmtPos(vn.position)}, ${vn.distance} blocks away, with the ${pick}.`,
        steps: [equip, ['dig_vein', { ...xyz(vn.position), also: vn.block, maxBlocks: 16 }], COLLECT],
        meta: { gain: kind === 'coal' ? COAL_DROPS : IRON_DROPS },
      });
    }
  }
  return out;
}

function nearest(ob: Observation, names: string[]): Entity | null {
  return [...ob.entities].filter((e) => names.includes(String(e.name))).sort((a, b) => Number(a.distance ?? 99) - Number(b.distance ?? 99))[0] ?? null;
}

function dangerActions(ob: Observation, sword: string | null): MacroAction[] {
  const h = nearest(ob, HOSTILE);
  if (!h) return [];
  const out: MacroAction[] = [];
  if (h.name !== 'creeper') {
    out.push({
      id: 'fight_hostile',
      description: `Fight the ${h.name} ${h.distance} blocks away until it dies (auto-equips your best weapon${sword ? '' : ' — you have no sword'}).`,
      steps: [['attack_entity', { entity: String(h.name), until: 'dead', minHealth: 8, timeoutSec: 20 }]],
      meta: { mob: h.name },
    });
  }
  const me = pos(ob);
  const hp = h.position ?? me;
  const dx = me.x - hp.x;
  const dz = me.z - hp.z;
  const norm = Math.max(1e-6, Math.hypot(dx, dz));
  const away = { x: Math.round(me.x + 14 * dx / norm), y: Math.round(me.y), z: Math.round(me.z + 14 * dz / norm), range: 3 };
  out.push({ id: 'flee_hostile', description: `Run ~14 blocks directly away from the ${h.name}.`, steps: [['go_to', away]], meta: { mob: h.name } });
  return out;
}

function foodActions(ob: Observation): MacroAction[] {
  const out: MacroAction[] = [];
  const food = Number(ob.state.food ?? 20);
  const hasMeal = ob.state.foodPortions > 0;
  const a = nearest(ob, FOOD_MOBS);
  if (a && (food < 16 || !hasMeal)) {
    out.push({
      id: 'hunt_animal',
      description: `Kill the ${a.name} ${a.distance} blocks away for food and pick up the drop.`,
      steps: [['attack_entity', { entity: String(a.name), until: 'dead', timeoutSec: 20 }], COLLECT],
      meta: {},
    });
  }
  if (hasMeal && food <= 16) out.push({ id: 'eat', description: 'Eat the best food in your bag.', steps: [['eat', {}]], meta: {} });
  return out;
}

/** The positions build_structure(shape='box', door='none') fills: walls plus a flat roof, no floor. */
export function boxPlan(anchor: Vec, w = 3, d = 3, h = 3): Vec[] {
  const out: Vec[] = [];
  for (let x = 0; x < w; x++) for (let z = 0; z < d; z++) for (let y = 0; y < h; y++) {
    const shell = x === 0 || x === w - 1 || z === 0 || z === d - 1;
    if (shell || y === h - 1) out.push({ x: anchor.x + x, y: anchor.y + y, z: anchor.z + z });
  }
  return out;
}

function shelterAction(ob: Observation): MacroAction[] {
  const usable = Object.fromEntries(Object.entries(ob.state.inv).filter(([n]) => SHELTER_BLOCKS.includes(n) || n.endsWith('_planks')));
  const total = Object.values(usable).reduce((a, b) => a + b, 0);
  if (total < 28 || ob.state.isDay !== false) return [];
  const item = Object.keys(usable).reduce((a, b) => (usable[b] > usable[a] ? b : a));
  const me = pos(ob);
  const anchor = { x: Math.floor(me.x) - 1, y: Math.floor(me.y), z: Math.floor(me.z) - 1 };
  return [{
    id: 'build_shelter',
    description: `Build a 3×3×3 ${item} box shelter around yourself for the night (no door).`,
    steps: [['build_structure', { shape: 'box', item, anchor, width: 3, depth: 3, height: 3, door: 'none' }]],
    meta: { item, plan: boxPlan(anchor) },
  }];
}

function movementActions(ob: Observation): MacroAction[] {
  const me = pos(ob);
  const out: MacroAction[] = ([['north', 0, -24], ['east', 24, 0]] as const).map(([label, dx, dz]) => ({
    id: `explore_${label}`,
    description: `Walk ~24 blocks ${label} to find new resources.`,
    steps: [['go_to', { x: Math.round(me.x + dx), y: Math.round(me.y), z: Math.round(me.z + dz), range: 4 }]] as ToolStep[],
    meta: { minMove: policyCfg.exploreMinMove },
  }));
  // `wait` passes game time: a no-op step is instantaneous and would burn the decision budget.
  out.push({
    id: 'wait',
    description: 'Do nothing this step (only when every other option is worse, e.g. safe inside a shelter at night).',
    steps: [['stop_moving', {}]],
    waitS: policyCfg.waitS,
    meta: {},
  });
  return out;
}

/** Only the actions that are legal and grounded in THIS observation, most urgent first. */
export function candidates(ob: Observation): MacroAction[] {
  const inv = ob.state.inv;
  const [pick, sword, axe] = [has(inv, PICKS), has(inv, SWORDS), has(inv, AXES)];
  const drops: MacroAction[] = ob.drops ? [{ id: 'collect_drops', description: `Pick up the ${ob.drops} dropped item(s) lying within 8 blocks.`, steps: [COLLECT], meta: {} }] : [];
  const ordered = [...dangerActions(ob, sword), ...foodActions(ob), ...treeActions(ob, axe), ...craftActions(ob), ...tableActions(ob), ...miningActions(ob, pick), ...drops, ...shelterAction(ob)];
  const tail = movementActions(ob); // explore + wait always survive the cap
  return [...ordered.slice(0, Math.max(0, policyCfg.maxCandidates - tail.length)), ...tail];
}

// ── what the model reads ─────────────────────────────────────────────────────

const BOILERPLATE = /Equipped [\w ]+ to hand, confirmed in the slot\.\s*\|?\s*|\s*\|\s*No dropped items nearby\./g;
const HINT = /\((?:[^()]*?)(?:call again|re-run|run again)[^()]*\)|(?:Fix|Next|Try): [^.|]+\.?/gi;
const EMPTY = /^(?:No dropped items nearby\.?|\(skipped: .*\)|.*Inventory change: nothing\..*)$/i;

/** Outcome with boilerplate removed, whitespace collapsed, the tool's own next-step hint first. */
export function cleanOutcome(text: string): string {
  const t = text.replace(BOILERPLATE, '').replace(/\s+/g, ' ').replace(/^[ |]+|[ |]+$/g, '');
  const hints = [...new Set((t.match(HINT) ?? []).map((h) => h.trim()))];
  return hints.length ? `${hints.join(' ')} | ${t}` : t;
}

/** Each tool call's text cleaned, empty ones dropped, NEWEST FIRST. */
export function stepOutcome(texts: string[]): string {
  const parts = texts.map(cleanOutcome);
  const nonEmpty = parts.filter(Boolean);
  const useful = parts.filter((t) => t && !EMPTY.test(t));
  return (useful.length ? useful : nonEmpty.slice(-1)).reverse().join(' | ');
}

const res = (t: Cluster) => `${t.block} x${t.veinSize}${t.position ? ` at ${fmtPos(t.position)}` : ''} ${t.distance}m ${t.exposed ? 'exposed' : 'buried'}`;
const mob = (e: Entity) => `${e.name} ${e.distance}m${e.position && 'x' in e.position ? ` at ${fmtPos(e.position)}` : ''}`;
const r1 = (n: number) => Math.round(n * 10) / 10;

export interface Facts {
  health: number | null;
  food: number | null;
  day: boolean | null;
  timeOfDay: number | null;
  position: Vec;
  standing_on: string;
  in_water: boolean;
  inventory: Record<string, number>;
}

export function snapshotFacts(ob: Pick<Observation, 'state' | 'standingOn' | 'events'>): Facts {
  const s = ob.state;
  const p = s.pos ?? { x: 0, y: 0, z: 0 };
  return {
    health: s.health,
    food: s.food,
    day: s.isDay,
    timeOfDay: s.timeOfDay,
    position: { x: r1(p.x), y: r1(p.y), z: r1(p.z) },
    standing_on: ob.standingOn,
    in_water: ob.standingOn === 'water' || ob.standingOn === 'bubble_column' || ob.events.some((e) => e.includes('drowning')),
    inventory: { ...s.inv },
  };
}

export function delta(a: Facts, b: Facts, seconds: number): Record<string, unknown> {
  const moved = Math.hypot(a.position.x - b.position.x, a.position.z - b.position.z);
  const bag: Record<string, string> = {};
  for (const n of [...new Set([...Object.keys(a.inventory), ...Object.keys(b.inventory)])].sort()) {
    const d = (b.inventory[n] ?? 0) - (a.inventory[n] ?? 0);
    if (d) bag[n] = d > 0 ? `+${d}` : String(d);
  }
  const out: Record<string, unknown> = { seconds: r1(seconds) };
  for (const k of ['health', 'food'] as const) {
    const x = a[k];
    const y = b[k];
    if (typeof x === 'number' && typeof y === 'number' && x !== y) out[k] = r1(y - x);
  }
  if (moved >= 0.5) out.moved = r1(moved);
  if (Math.abs(b.position.y - a.position.y) >= 1) out.dy = r1(b.position.y - a.position.y);
  if (Object.keys(bag).length) out.bag = bag;
  return out;
}

export interface Step {
  step: number;
  action: string;
  verdict: string;
  check: string;
  outcome: string;
  before: Facts;
  reason: string;
  change: Record<string, unknown>;
  events: string[];
}

export function stepLine(st: Step, chars: number, eventChars: number): Record<string, unknown> {
  const out: Record<string, unknown> = { step: st.step, action: st.action, verdict: st.verdict, check: st.check };
  if (st.reason) out.reason = st.reason;
  if (Object.keys(st.change).length) out.change = st.change;
  if (st.events.length) out.events = st.events.map((e) => e.slice(0, eventChars));
  out.outcome = st.outcome.slice(0, chars);
  return out;
}

/** Per action over the whole episode: tried / done / no effect. Never trimmed. */
export function tally(steps: Step[]): Record<string, string> {
  const t = new Map<string, [number, number, number]>();
  for (const s of steps) {
    const x = t.get(s.action) ?? [0, 0, 0];
    x[0] += 1;
    x[1] += s.verdict === 'done' ? 1 : 0;
    x[2] += s.verdict === 'no_effect' ? 1 : 0;
    t.set(s.action, x);
  }
  return Object.fromEntries([...t].map(([k, [a, d, f]]) => [k, `${a} tried, ${d} done, ${f} no effect`]));
}

/** Remove empty lists/dicts, one level into `nearby` too (v3 run-2 empties ablation). */
export function dropEmpty(d: Record<string, unknown>): Record<string, unknown> {
  const empty = (v: unknown) => (Array.isArray(v) ? v.length === 0 : !!v && typeof v === 'object' && Object.keys(v).length === 0);
  const out: Record<string, unknown> = {};
  for (const [k, v0] of Object.entries(d)) {
    let v = v0;
    if (k === 'nearby' && v && typeof v === 'object') v = Object.fromEntries(Object.entries(v).filter(([, nv]) => !empty(nv)));
    if (empty(v)) continue;
    out[k] = v;
  }
  return out;
}

/**
 * JSON text in Python `json.dumps(v, ensure_ascii=False)` default form (`, ` and
 * `: ` separators). v3 measured its state budget with exactly this, so the
 * budget here means the same number of characters.
 */
export function pyJson(v: unknown): string {
  if (v === null || v === undefined) return 'null';
  if (Array.isArray(v)) return `[${v.map(pyJson).join(', ')}]`;
  if (typeof v === 'object') return `{${Object.entries(v as Record<string, unknown>).filter(([, x]) => x !== undefined).map(([k, x]) => `${JSON.stringify(k)}: ${pyJson(x)}`).join(', ')}}`;
  if (typeof v === 'number' && Number.isInteger(v) && !Object.is(v, -0)) return String(v);
  return JSON.stringify(v);
}

/**
 * The projection the model judges from, bounded by the profile's char budget.
 * `recent_actions[i]` and `previous_states[i]` describe the same step. Trimmed
 * oldest-first; `action_tally`, `advancements` and `budget` always stay.
 * Size is measured as v3 measured it (`pyJson`).
 */
export function decisionState(goal: string, ob: Observation, steps: Step[], budget: Record<string, unknown>, profile: ContextProfile): Record<string, unknown> {
  const window = steps.slice(-profile.historySteps);
  let state: Record<string, unknown> = {
    goal,
    ...snapshotFacts(ob),
    nearby: {
      trees: ob.trees.map(res),
      stone: ob.stone.map(res),
      ores: [...ob.coal, ...ob.iron].map(res),
      crafting_table: ob.tables.map((t) => `at ${fmtPos(t.position)} ${t.distance}m`),
      furnace: ob.furnaces.map((t) => `at ${fmtPos(t.position)} ${t.distance}m`),
      mobs: ob.entities.filter((e) => e.name !== 'item' && e.name !== 'player').slice(0, 6).map(mob),
      dropped_items_within_8: ob.drops,
    },
    threats: ob.state.threats ?? [],
    events_since_last_step: ob.events,
    budget: { ...budget, deaths: ob.state.counters?.deaths ?? 0 },
    action_tally: tally(steps),
    advancements: steps.flatMap((s) => s.events.filter((e) => e.startsWith('advancement: ')).map((e) => e.slice('advancement: '.length))),
    recent_actions: window.map((s) => stepLine(s, profile.outcomeChars, profile.eventChars)),
  };
  if (profile.previousStates) state.previous_states = window.map((s) => s.before);
  state = dropEmpty(state);
  while (pyJson(state).length > profile.stateChars && Array.isArray(state.recent_actions) && state.recent_actions.length) {
    state.recent_actions = (state.recent_actions as unknown[]).slice(1);
    if (Array.isArray(state.previous_states)) state.previous_states = (state.previous_states as unknown[]).slice(1);
  }
  return dropEmpty(state);
}

/** One instruction for every model (v3 ablation: the short one is equal-or-best for all). */
export const INSTRUCTION = 'Which action best advances the `goal`?';
export const LAST_STEP_QUESTION = 'Did the most recent action in `recent_actions` achieve what it set out to do?';

/**
 * How option descriptions reach the model. A SelfHostedDecisionModel puts them
 * on the wire as choice criteria (as v3 did). SDK-native providers carry
 * `choices` as ids only, so the descriptions are folded into the question text.
 * Recorded per episode as `option_rendering`.
 */
export type OptionRendering = 'criteria' | 'question_text';

export function buildQuestions(acts: MacroAction[], withLastStep: boolean, rendering: OptionRendering): Record<string, BenchQuestion> {
  const ids = acts.map((a) => a.id) as [string, ...string[]];
  const descriptions = Object.fromEntries(acts.map((a) => [a.id, a.description]));
  const next: BenchQuestion = rendering === 'criteria'
    ? { question: INSTRUCTION, choices: ids, uncertainty: false, descriptions }
    : { question: `${INSTRUCTION}\nOptions:\n${acts.map((a) => `- ${a.id}: ${a.description}`).join('\n')}`, choices: ids, uncertainty: false };
  const q: Record<string, BenchQuestion> = { next_action: next };
  if (withLastStep) q.last_step_succeeded = { question: LAST_STEP_QUESTION, choices: 'boolean', uncertainty: false };
  return q;
}

/** The JSON text the model receives — indented, as strands-decider renders a dict state. */
export const serializeState = (state: Record<string, unknown>) => JSON.stringify(state, null, 2);

export interface Decision {
  step: number;
  action: string;
  confidence: number | null;
  top: Array<[string, number]>;
  modelId: string | null;
  latencyMs: number;
  serverLatencyMs: number | null;
  usage: { inputTokens?: number; outputTokens?: number };
  options: number;
  stateChars: number;
  outcome: string;
  /** Transport only: every tool call returned. Never shown to the model. */
  transportOk: boolean;
  error: string | null;
  verdict: string;
  check: string;
  reason: string;
  lastStepP: number | null;
  probabilitySum: number | null;
  /** True when this call ended because the episode-owned wall signal fired. */
  timedOut: boolean;
  request: { state: Record<string, unknown>; options: Record<string, string> };
}

export interface PolicyBrainOptions {
  profile: ContextProfile;
  rendering?: OptionRendering;
  now?: () => number;
  wait?: (ms: number) => Promise<void>;
}

/** Observe → candidates → one ask (next action + last-step judgment) → execute → verify. */
export class PolicyBrain {
  readonly steps: Step[] = [];
  readonly decisions: Decision[] = [];
  budget: Record<string, unknown> = {};
  private blockS: number[] = [];
  readonly rendering: OptionRendering;
  private readonly now: () => number;
  private readonly wait: (ms: number) => Promise<void>;

  constructor(readonly model: DecisionModel, readonly body: Body, readonly goal: string, readonly o: PolicyBrainOptions) {
    this.rendering = o.rendering ?? (model instanceof SelfHostedDecisionModel ? 'criteria' : 'question_text');
    this.now = o.now ?? (() => performance.now() / 1000);
    this.wait = o.wait ?? sleep;
  }

  async step(cancelSignal?: AbortSignal): Promise<Decision> {
    const tStart = this.now();
    const ob = await observe(this.body, this.o.profile);
    const acts = candidates(ob);
    const state = decisionState(this.goal, ob, this.steps, this.budget, this.o.profile);
    const questions = buildQuestions(acts, this.steps.length > 0, this.rendering);
    const request = { state, options: Object.fromEntries(acts.map((a) => [a.id, a.description])) };
    const n = this.decisions.length + 1;
    const stateChars = pyJson(state).length;
    const base = { step: n, options: acts.length, stateChars, request, outcome: '', transportOk: true, probabilitySum: null, timedOut: false } as const;
    const t0 = performance.now();
    let resp;
    try {
      resp = await this.model.ask(serializeState(state), questions as Record<string, Question>, { cancelSignal });
    } catch (err) {
      const timedOut = cancelSignal?.aborted === true;
      const msg = `${err instanceof Error ? err.name : 'Error'}: ${err instanceof Error ? err.message : String(err)}`.slice(0, 300);
      const d: Decision = {
        ...base, timedOut, action: timedOut ? 'timeout' : 'error', confidence: null, top: [], modelId: null, latencyMs: performance.now() - t0, serverLatencyMs: null,
        usage: {}, transportOk: false, error: msg, verdict: 'no_effect',
        check: timedOut ? 'the episode wall deadline cancelled the decision; no action ran' : 'the decision call raised; no action ran',
        reason: timedOut ? 'episode wall deadline reached' : msg, lastStepP: null,
      };
      this.decisions.push(d);
      return d;
    }
    const latency = performance.now() - t0;
    const raw = resp.answers.next_action;
    const probs = (resp.metadata.probabilities?.next_action ?? {}) as Record<string, number>;
    const top = Object.entries(probs).sort((a, b) => b[1] - a[1]).slice(0, 3).map(([k, p]) => [k, +p.toFixed(4)] as [string, number]);
    const lastP = resp.metadata.probabilities?.last_step_succeeded?.true;
    const extras = this.model instanceof SelfHostedDecisionModel ? this.model.last : {};
    const d: Decision = {
      ...base,
      action: raw instanceof Uncertain ? 'uncertain' : String(raw),
      confidence: resp.metadata.confidence?.next_action ?? null,
      top,
      modelId: resp.metadata.modelId ?? null,
      latencyMs: +latency.toFixed(1),
      serverLatencyMs: extras.serverLatencyMs ?? null,
      usage: resp.usage ? { inputTokens: resp.usage.inputTokens, outputTokens: resp.usage.outputTokens } : {},
      error: null,
      verdict: 'no_effect',
      check: '',
      reason: '',
      lastStepP: lastP === undefined ? null : +lastP.toFixed(4),
      probabilitySum: extras.probabilitySums?.next_action ?? null,
    };
    const act = acts.find((a) => a.id === d.action);
    let modelOutcome = '';
    if (!act) {
      d.transportOk = false;
      d.error = `model chose ${JSON.stringify(d.action)}, not one of the offered ids`;
      [d.verdict, d.check, d.reason] = ['no_effect', 'no such option; nothing ran', d.error];
    } else {
      const plan = act.meta.plan ?? null;
      const before = await takeSnapshot(this.body, HOSTILE, plan, this.now());
      this.body.setBusy(true);
      let outs: ToolOutcome[];
      try { outs = await runAction(act, this.body, this.wait); } finally { this.body.setBusy(false); }
      const after = await takeSnapshot(this.body, HOSTILE, plan, this.now());
      d.transportOk = outs.every((o) => o.ok);
      const full = outs.map((o) => o.text).join(' | ');
      d.outcome = outs.map((o) => short(o, 160)).join(' | ').slice(0, 400);
      modelOutcome = stepOutcome(outs.map((o) => o.text));
      if (plan) {
        const g = after.solid.size - before.solid.size;
        if (g > 0) this.blockS.push((after.t - before.t) / g);
      }
      const spb = this.blockS.length ? this.blockS.reduce((a, b) => a + b, 0) / this.blockS.length : null;
      const vd = checkAction(act.id, act.meta, before, after, full, spb);
      [d.verdict, d.check, d.reason] = [vd.verdict, vd.check, vd.reason];
      if (!d.transportOk && d.verdict === 'done') d.check += ' (a tool call raised, but the world shows the effect)';
    }
    const beforeFacts = snapshotFacts(ob);
    const afterFacts = snapshotFacts({ state: this.body.state(), standingOn: ob.standingOn, events: [] });
    const change = delta(beforeFacts, afterFacts, this.now() - tStart);
    // Events raised DURING this action belong to this step.
    const during = formatEvents(this.body.drainEvents(), this.o.profile.eventChars, this.o.profile.maxEvents);
    this.steps.push({ step: n, action: d.action, verdict: d.verdict, check: d.check, outcome: modelOutcome || cleanOutcome(d.error ?? ''), before: beforeFacts, reason: d.reason, change, events: during });
    this.decisions.push(d);
    return d;
  }
}
