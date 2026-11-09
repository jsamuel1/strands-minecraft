/**
 * 🏁 `npm run bench -- <cmd>` — the SDK-native benchmark CLI.
 *
 *   run    --arms decider --tasks logs16,night --n 3 [--out bench-results/sdk-native/<run>] [--port-base 25600]
 *   report <run-dir> [<run-dir> …]
 *   episode (internal) one episode in THIS process; the parent spawns it per episode.
 *
 * Arms (policy arms use a native SDK DecisionModel; agent arms an SDK Agent):
 *   decider       strands-decider server at STRANDS_DECIDER_URL (default http://127.0.0.1:8099)
 *   jev           TypeSafeDecisionModel (TYPESAFE_API_KEY), hosted Jev
 *   llm-policy    LLMDecisionModel over the env's Bedrock/OpenAI/Anthropic model, same alphabet
 *   agent         the env's model as a tool-calling Agent over the body's tools
 *
 * Results default to bench-results/sdk-native/ — never next to the v3 results.
 */
import { spawnSync } from 'node:child_process';
import { appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { TASKS } from './tasks.js';
import { createDecisionModel, resolveDecisionSpec } from './decider.js';
import { HARNESS } from './episode.js';
import { PROFILES, policyCfg } from './policy.js';
import { BENCH_BODY_ENV, type Arm } from './runner.js';
import { worldCfg } from './world.js';
import { cfg } from '../config.js';

export const RESULTS_ROOT = 'bench-results/sdk-native';
export const ARMS = ['decider', 'jev', 'llm-policy', 'agent'] as const;

/** Directories this harness must never write into: the v3 results are immutable. */
export const PROTECTED_RESULTS = /(^|\/)bench-results(?:$|\/(?!sdk-native(?:\/|$)))/;

export function parseArgs(argv: string[]): { cmd: string; flags: Record<string, string>; rest: string[] } {
  const [cmd = 'help', ...tail] = argv;
  const flags: Record<string, string> = {};
  const rest: string[] = [];
  for (let i = 0; i < tail.length; i++) {
    if (!tail[i].startsWith('--')) { rest.push(tail[i]); continue; }
    const key = tail[i].slice(2);
    const value = tail[i + 1];
    if (!value || value.startsWith('--')) throw new Error(`--${key} needs a value`);
    flags[key] = value;
    i++;
  }
  return { cmd, flags, rest };
}

/** The output dir, validated: absolute, and under bench-results/sdk-native (or outside bench-results entirely). */
export function resolveOut(out: string | undefined, runId: string, cwd = process.cwd()): string {
  const dir = resolve(cwd, out || join(RESULTS_ROOT, runId));
  if (PROTECTED_RESULTS.test(dir.replace(/\\/g, '/'))) {
    throw new Error(`refusing to write ${dir}: results from this harness go under ${RESULTS_ROOT}/ (v3 results are immutable)`);
  }
  return dir;
}

/** A run owns the directory exclusively once any path exists there. */
export function createFreshRunOutput(dir: string): void {
  try {
    mkdirSync(dir, { recursive: false });
  } catch (err) {
    const code = (err as NodeJS.ErrnoException).code;
    if (code === 'EEXIST') throw new Error(`${dir} exists — use a new run id/output; scored runs are append-immutable`, { cause: err });
    throw err;
  }
}

async function llm() {
  const { createModel, resolveModelSpec } = await import('../model.js');
  return createModel(resolveModelSpec());
}

export async function armFor(name: string): Promise<Arm> {
  if (name === 'decider' || name === 'jev') {
    const spec = resolveDecisionSpec({ ...process.env, BENCH_DECIDER_PROVIDER: name === 'jev' ? 'typesafe' : 'decider' });
    return { kind: 'policy', arm: name, label: spec.label, profile: name === 'jev' ? PROFILES.jev : PROFILES.decider, model: () => createDecisionModel(spec) };
  }
  if (name === 'llm-policy') {
    const m = await llm();
    const spec = resolveDecisionSpec({ ...process.env, BENCH_DECIDER_PROVIDER: 'llm' });
    return { kind: 'policy', arm: name, label: spec.label, profile: PROFILES.jev, model: () => createDecisionModel(spec, { llm: () => m }) };
  }
  if (name === 'agent') {
    const { resolveModelSpec } = await import('../model.js');
    const s = resolveModelSpec();
    return { kind: 'agent', arm: name, label: `${s.provider}:${s.modelId}`, model: llm };
  }
  throw new Error(`unknown arm ${JSON.stringify(name)} — use one of ${ARMS.join(', ')}`);
}

export function buildRunManifest(p: { runId: string; arms: string[]; tasks: string[]; n: number; startedAt: string; argv: string[] }) {
  const body = {
    env: { ...BENCH_BODY_ENV },
    safety_reflexes: true,
    sentinel: true,
    config: {
      sentinel: { ...cfg.sentinel },
      reflex: { ...cfg.reflex },
      combat: { ...cfg.combat, flyers: [...cfg.combat.flyers], neverPunch: [...cfg.combat.neverPunch] },
    },
  };
  return {
    harness: HARNESS,
    run_id: p.runId,
    arms: p.arms,
    tasks: p.tasks,
    n: p.n,
    started_at: p.startedAt,
    argv: p.argv,
    world: { image: worldCfg.image, version: worldCfg.version, seed: worldCfg.seed, difficulty: 'easy', mode: 'survival', tmpfs: '/data:size=3g' },
    task_budgets: Object.fromEntries(p.tasks.map((id) => [id, { steps: TASKS[id].maxSteps, wall_s: TASKS[id].maxWallS }])),
    policy: {
      ...policyCfg,
      profiles: Object.fromEntries(p.arms.filter((a) => a !== 'agent').map((a) => [a, { ...(a === 'decider' ? PROFILES.decider : PROFILES.jev) }])),
    },
    body,
  };
}

async function episodeCmd(flags: Record<string, string>) {
  const { runEpisode, appendJsonl } = await import('./runner.js');
  const task = TASKS[flags.task];
  if (!task) throw new Error(`unknown task ${flags.task}`);
  const base = Number(flags['port-base'] || 25600);
  const out = resolveOut(flags.out, flags['run-id']);
  const ep = await runEpisode({ runId: flags['run-id'], arm: await armFor(flags.arm), task, n: Number(flags.n), ports: [base, base + 1], logDir: join(out, 'logs') });
  appendJsonl(join(out, 'episodes.jsonl'), ep);
  console.log(`${ep.arm} ${ep.task} #${ep.episode}: ${ep.success ? 'success' : ep.error ? `infra: ${ep.error}` : `fail (${ep.cap_hit})`} steps=${ep.steps} loop=${ep.loop_s}s decide=${ep.decide_s}s`);
  // The body's rails keep interval timers (sentinel radar, reflex tick) that the game bot never
  // needs to end. The row is on disk (appendFileSync); this process exists for one episode.
  return 'exit';
}

function runCmd(flags: Record<string, string>) {
  const arms = (flags.arms || 'decider').split(',').filter(Boolean);
  const tasks = (flags.tasks || Object.keys(TASKS).join(',')).split(',').filter(Boolean);
  for (const t of tasks) if (!TASKS[t]) throw new Error(`unknown task ${t}`);
  for (const a of arms) if (!(ARMS as readonly string[]).includes(a)) throw new Error(`unknown arm ${a}`);
  const n = Number(flags.n || 1);
  if (!Number.isInteger(n) || n < 1) throw new Error(`--n must be a positive integer, got ${JSON.stringify(flags.n)}`);
  const runId = flags['run-id'] || `${new Date().toISOString().slice(0, 10)}-${arms.join('+')}`;
  const out = resolveOut(flags.out, runId);
  createFreshRunOutput(out);
  const manifest = buildRunManifest({
    runId,
    arms,
    tasks,
    n,
    startedAt: new Date().toISOString(),
    argv: process.argv.slice(2),
  });
  writeFileSync(join(out, 'run.json'), `${JSON.stringify(manifest, null, 2)}\n`);
  const self = fileURLToPath(import.meta.url);
  // Interleave arms within each (task, episode) so drift in host load hits every arm alike.
  for (const task of tasks) for (let i = 1; i <= n; i++) for (const arm of arms) {
    const r = spawnSync(process.execPath, ['--import', 'tsx', self, 'episode', '--arm', arm, '--task', task, '--n', String(i), '--out', out, '--run-id', runId, '--port-base', flags['port-base'] || '25600'], {
      stdio: 'inherit',
      // A backstop, not a budget: the task's own wall cap binds inside the episode; this only
      // reaps a child wedged in world bootstrap or teardown (world start allows 240 s).
      timeout: (TASKS[task].maxWallS + 600) * 1000,
      killSignal: 'SIGKILL',
    });
    if (r.status !== 0) {
      console.error(`episode ${arm}/${task}/${i} exited ${r.status ?? r.signal}`);
      // A child that died before writing its row is still an episode: record it as infra, never drop it.
      const rows = existsSync(join(out, 'episodes.jsonl')) ? readFileSync(join(out, 'episodes.jsonl'), 'utf8') : '';
      const wrote = rows.split('\n').some((l) => { try { const x = JSON.parse(l); return x.arm === arm && x.task === task && x.episode === i; } catch { return false; } });
      if (!wrote) {
        appendFileSync(join(out, 'episodes.jsonl'), `${JSON.stringify({ run_id: runId, arm, task, episode: i, harness: 'sdk-native-v1', success: false, error: `episode process exited ${r.status ?? r.signal} without a row`, started_at: new Date().toISOString() })}\n`);
      }
    }
  }
  console.log(`results: ${out}/episodes.jsonl — npm run bench -- report ${out}`);
}

async function reportCmd(rest: string[]) {
  const { loadRun, table } = await import('./report.js');
  console.log(table(rest.flatMap((d) => loadRun(d))));
}

export async function main(argv = process.argv.slice(2)) {
  const { cmd, flags, rest } = parseArgs(argv);
  if (cmd === 'run') return runCmd(flags);
  if (cmd === 'episode') return episodeCmd(flags);
  if (cmd === 'report') return reportCmd(rest);
  console.log('usage: npm run bench -- run --arms decider --tasks logs16 --n 1 | report <dir>…  (docs/bench/sdk-native.md)');
}

if (process.argv[1] && /bench[\\/]cli\.(ts|js)$/.test(process.argv[1])) {
  main().then((r) => { if (r === 'exit') process.exit(0); }).catch((err) => { console.error(err instanceof Error ? err.message : err); process.exit(1); });
}
