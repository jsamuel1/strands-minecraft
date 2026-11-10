/**
 * The fixed task suite — the same five goals, budgets and checks as v3
 * (brain/src/mcbench/tasks.py on feat/bench-v3), so rows are comparable.
 *
 * Every goal is VERIFIED FROM BODY STATE by code, never from the model's
 * say-so. `check(state) → [done, progress]` reads the harness's world snapshot.
 */

export interface BodyState {
  inv: Record<string, number>;
  timeOfDay: number | null;
  counters: { deaths: number };
  /** Harness marks (world truth the harness measured, never the model). */
  _placedTable?: boolean;
  _passedNight?: boolean;
  /** Previous sampled day clock, used to prove a night→day wrap even when one long turn spans it. */
  _lastTimeOfDay?: number;
}

export interface Task {
  id: string;
  goal: string;
  check: (s: BodyState) => [boolean, number];
  /** Decisions (policy arms) / turns (agent arms). UNCAPPED_STEPS = none. */
  maxSteps: number;
  maxWallS: number;
  /** RCON commands that shape the world before the bot acts. Never gives items. */
  setup: readonly string[];
}

/** A step cap this large means "no step limit": the wall-clock cap binds. Shown to the model as null. */
export const UNCAPPED_STEPS = 10_000;

const sum = (s: BodyState, pred: (n: string) => boolean) =>
  Object.entries(s.inv ?? {}).reduce((t, [n, c]) => t + (pred(n) ? c : 0), 0);

export const countLogs = (s: BodyState) => sum(s, (n) => n.endsWith('_log'));
export const countCobble = (s: BodyState) => sum(s, (n) => n === 'cobblestone' || n === 'cobbled_deepslate');
export const countIron = (s: BodyState) => sum(s, (n) => ['raw_iron', 'iron_ore', 'deepslate_iron_ore', 'iron_ingot'].includes(n));

const ratio = (n: number, of: number): [boolean, number] => [n >= of, +Math.min(1, n / of).toFixed(3)];

function woodTools(s: BodyState): [boolean, number] {
  const inv = s.inv ?? {};
  // A PLACED table counts: crafting the pickaxe needs the table in the world, not in the bag.
  const table = (inv.crafting_table ?? 0) > 0 || !!s._placedTable;
  const pick = (inv.wooden_pickaxe ?? 0) > 0 || (inv.stone_pickaxe ?? 0) > 0;
  return [table && pick, +((table ? 0.5 : Math.min(0.4, countLogs(s) / 10)) + (pick ? 0.5 : 0)).toFixed(3)];
}

function survive(s: BodyState): [boolean, number] {
  const deaths = s.counters?.deaths ?? 0;
  const t = s.timeOfDay ?? 0;
  const done = deaths === 0 && !!s._passedNight;
  return [done, done ? 1 : deaths ? 0 : +Math.min(1, Math.max(0, t - 13000) / 10000).toFixed(3)];
}

export const SUITE: readonly Task[] = [
  { id: 'logs16', goal: 'Collect 16 logs (any wood type).', check: (s) => ratio(countLogs(s), 16), maxSteps: 20, maxWallS: 420, setup: [] },
  { id: 'wood_tools', goal: 'Craft a crafting table and a wooden pickaxe.', check: woodTools, maxSteps: 24, maxWallS: 480, setup: [] },
  { id: 'cobble8', goal: 'Get a pickaxe and mine 8 cobblestone.', check: (s) => ratio(countCobble(s), 8), maxSteps: 32, maxWallS: 600, setup: [] },
  { id: 'iron1', goal: 'Obtain at least one iron ore (needs a stone pickaxe).', check: (s) => ratio(countIron(s), 1), maxSteps: 48, maxWallS: 900, setup: [] },
  // Wall-clock binds, never steps: night is 11,500 ticks (575 s) from 12500 (v3 rationale).
  { id: 'night', goal: 'Survive the night without dying.', check: survive, maxSteps: UNCAPPED_STEPS, maxWallS: 660, setup: ['time set 12500', 'difficulty easy'] },
];

export const TASKS: Readonly<Record<string, Task>> = Object.freeze(Object.fromEntries(SUITE.map((t) => [t.id, t])));

/** Harness marks from world truth: placed table, night passed. Mutates `marks`; pure over its inputs. */
export function applyMarks(
  s: Omit<BodyState, '_placedTable' | '_passedNight' | '_lastTimeOfDay'>,
  marks: { tablesAtStart?: number; placedTable?: boolean; sawNight?: boolean; passedNight?: boolean; lastTimeOfDay?: number },
  tablesNear: number,
): BodyState {
  if (marks.tablesAtStart === undefined) marks.tablesAtStart = tablesNear;
  else if (!marks.placedTable && tablesNear > marks.tablesAtStart) marks.placedTable = true;
  const t = s.timeOfDay ?? 0;
  const wrapped = marks.lastTimeOfDay !== undefined && marks.lastTimeOfDay >= 12000 && t < 1000;
  if (t >= 13000 || wrapped) marks.sawNight = true;
  if (wrapped || (marks.sawNight && t >= 0 && t < 1000)) marks.passedNight = true;
  marks.lastTimeOfDay = t;
  return { ...s, _placedTable: !!marks.placedTable, _passedNight: !!marks.passedNight, _lastTimeOfDay: marks.lastTimeOfDay };
}
