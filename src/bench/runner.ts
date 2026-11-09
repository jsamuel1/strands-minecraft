/**
 * 🏁 Episode runner: fresh world → fresh body → one arm → one task → one JSONL row.
 *
 * Isolation (v3's): a new server container per episode with its world on tmpfs
 * and a fixed seed; a fresh MEMORY_DIR; one bot per server; idle reflexes that
 * move or collect off for every arm. Infrastructure failures are recorded in
 * `error`, never hidden and never counted as a model failure.
 */
import { appendFileSync, copyFileSync, mkdirSync, mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { createRequire } from 'node:module';
import type { DecisionModel, Model } from '@strands-agents/sdk';
import type { Task } from './tasks.js';
import { newEpisode, type Episode } from './episode.js';
import { buildAgent, runAgent, runPolicy, type Marks, goalState } from './loops.js';
import { PROFILES, type ContextProfile } from './policy.js';
import { worldCfg, World } from './world.js';

export const SDK_VERSION: string = (() => {
  try { return (createRequire(import.meta.url)('@strands-agents/sdk/package.json') as { version: string }).version; }
  catch { return JSON.parse(readFileSync(new URL('../../node_modules/@strands-agents/sdk/package.json', import.meta.url), 'utf8')).version; }
})();

/** Short HEAD, suffixed `-dirty` when tracked files differ from it: a row must never claim code it did not run. */
export function gitCommit(): string | null {
  try {
    const sha = execFileSync('git', ['rev-parse', '--short', 'HEAD'], { encoding: 'utf8' }).trim();
    if (!sha) return null;
    const dirty = execFileSync('git', ['status', '--porcelain', '--untracked-files=no', '--', 'src', 'package.json', 'package-lock.json'], { encoding: 'utf8' }).trim();
    return dirty ? `${sha}-dirty` : sha;
  } catch { return null; }
}

export interface PolicyArm { kind: 'policy'; arm: string; label: string; model: () => DecisionModel; profile: ContextProfile }
export interface AgentArm { kind: 'agent'; arm: string; label: string; model: () => Model | Promise<Model> }
export type Arm = PolicyArm | AgentArm;

export interface EpisodeOptions { runId: string; arm: Arm; task: Task; n: number; ports: [number, number]; logDir?: string }

export async function runEpisode(o: EpisodeOptions): Promise<Episode> {
  const ep = newEpisode({
    run_id: o.runId, arm: o.arm.arm, kind: o.arm.kind, model_label: o.arm.label, task: o.task.id, episode: o.n,
    seed: worldCfg.seed, mc_version: worldCfg.version, brain_commit: gitCommit(), strands_sdk: SDK_VERSION,
  });
  const t0 = performance.now();
  const world = new World(o.ports[0], o.ports[1]);
  const mem = mkdtempSync(join(tmpdir(), 'mcbench-mem-'));
  // Set before the body's modules load: the memory store reads MEMORY_DIR at import time, and
  // legs/sentinel keep module state. So cli.ts runs ONE episode per child process and the body
  // modules are imported only below, after this.
  Object.assign(process.env, {
    MEMORY_DIR: mem, REFLEX_MODES_OFF: process.env.REFLEX_MODES_OFF ?? 'item_magnet,elbow_room,idle_staring',
    THINKER_DISABLED: 'true', VOICE_DISABLED: 'true', MC_HOST: '127.0.0.1', MC_PORT: String(o.ports[0]),
    MC_USERNAME: 'StrandsBot', MC_VERSION: worldCfg.version,
  });
  let rig: import('./rig.js').Rig | undefined;
  try {
    await world.start();
    ep.mc_image = await World.imageDigest();
    for (const cmd of o.task.setup) await world.rcon(cmd);
    const { createLiveBody } = await import('../body.js');
    const { Rig } = await import('./rig.js');
    const live = await createLiveBody({ host: '127.0.0.1', port: o.ports[0], username: 'StrandsBot', version: worldCfg.version });
    rig = new Rig(live);
    const marks: Marks = {};
    await goalState(rig, marks); // baseline: tables in reach before the brain acts
    ep.setup_s = +((performance.now() - t0) / 1000).toFixed(1);
    if (o.arm.kind === 'policy') {
      await runPolicy(ep, o.arm.model(), o.arm.profile, o.task, rig, marks);
    } else {
      const { SYSTEM_PROMPT } = await import('../agent.js');
      const { worldDigest } = await import('../digest.js');
      const bot = live.bot;
      const agent = buildAgent(await o.arm.model(), rig.tools, SYSTEM_PROMPT.replace('{username}', bot.username));
      const digestBody = Object.assign(rig, { digest: () => worldDigest(bot, { threats: () => rig!.sentinel?.threats() ?? [] }) });
      await runAgent(ep, agent, o.task, digestBody, marks);
      ep.actions = Object.entries(rig.toolCounts).map(([k, v]) => `${k}x${v}`);
      if (!ep.answering_models.length) ep.answering_models = [o.arm.label];
    }
    const final = rig.state();
    ep.final_inventory = { ...final.inv };
    ep.tool_calls = final.counters.toolCalls;
    ep.tool_errors = final.counters.toolErrors;
    ep.deaths = final.counters.deaths;
  } catch (err) {
    ep.error = `${err instanceof Error ? err.name : 'Error'}: ${err instanceof Error ? err.message : String(err)}`.slice(0, 500);
  } finally {
    try { rig?.stop(); } catch { /* closing */ }
    if (o.logDir) {
      try { mkdirSync(o.logDir, { recursive: true }); copyFileSync(join(mem, 'memory.json'), join(o.logDir, `${o.task.id}.${o.arm.arm}.${o.n}.memory.json`)); } catch { /* no memory written */ }
    }
    await world.stop();
    rmSync(mem, { recursive: true, force: true });
  }
  ep.wall_s = +((performance.now() - t0) / 1000).toFixed(1);
  // Local servers cost nothing measurable here; hosted token pricing is out of scope for this harness.
  ep.cost_usd = null;
  return ep;
}

export function appendJsonl(path: string, ep: Episode): void {
  mkdirSync(dirname(path), { recursive: true });
  appendFileSync(path, `${JSON.stringify(ep)}\n`);
}

export { PROFILES };
