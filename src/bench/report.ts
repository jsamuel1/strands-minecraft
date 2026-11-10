/**
 * The human results table — v3's `mcbench report`, ported: per arm × task,
 * median (min–max) per episode, budgets starred.
 *
 * Columns: n, success, fail, decisions, decide s, elapsed s, decide %.
 * - decisions: model calls that chose what to do (policy: one ask per step; agent: model calls).
 * - decide s: exact time waiting on the model (`decide_s`).
 * - elapsed s: brain loop time, first step to goal or budget (`loop_s`).
 * - `*` on a max: at least one episode in the row hit that budget (decisions → step cap, elapsed → wall cap).
 * - fail counts goal-not-reached; infrastructure failures show as `+N infra`.
 */
import { readFileSync } from 'node:fs';
import { basename, join, resolve } from 'node:path';
import { SUITE, UNCAPPED_STEPS } from './tasks.js';

type Row = Record<string, any> & { _run: string };

export function loadRun(dir: string, label?: string): Row[] {
  return readFileSync(join(dir, 'episodes.jsonl'), 'utf8').split('\n').filter((l) => l.trim())
    .map((l) => ({ ...JSON.parse(l), _run: label ?? basename(resolve(dir)) }));
}

const median = (xs: number[]) => {
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};

const f = (x: number, digits: number) => x.toFixed(digits);
const cell = (xs: number[], digits: number, starred: boolean) =>
  `${f(median(xs), digits)} (${f(Math.min(...xs), digits)}–${f(Math.max(...xs), digits)}${starred ? '*' : ''})`;

export function table(rows: Row[]): string {
  const groups = new Map<string, Row[]>();
  for (const r of rows) {
    const k = `${r._run}\u0000${r.arm}\u0000${r.task}`;
    groups.set(k, [...(groups.get(k) ?? []), r]);
  }
  const out = ['| arm | run | task | n | success | fail | decisions | decide s | elapsed s | decide % |', '|---|---|---|---|---|---|---|---|---|---|'];
  const arms = [...new Set(rows.map((r) => `${r._run}\u0000${r.arm}`))];
  for (const ra of arms) {
    const [run, arm] = ra.split('\u0000');
    for (const t of SUITE) {
      const g = groups.get(`${run}\u0000${arm}\u0000${t.id}`);
      if (!g) continue;
      const ok = g.filter((r) => !r.error);
      const infra = g.length - ok.length;
      const fail = `${ok.filter((r) => !r.success).length}${infra ? `+${infra} infra` : ''}`;
      const caps = ok.filter((r) => !r.success).map((r) => r.cap_hit);
      if (!ok.length) {
        out.push(`| ${arm} | ${run} | ${t.id} | ${g.length} | 0 | ${fail} | – | – | – | – |`);
        continue;
      }
      const dec = ok.map((r) => Number(r.decide_s ?? 0));
      const ela = ok.map((r) => Number(r.loop_s ?? 0));
      const share = ela.reduce((a, b) => a + b, 0) ? (100 * dec.reduce((a, b) => a + b, 0)) / ela.reduce((a, b) => a + b, 0) : 0;
      out.push(`| ${arm} | ${run} | ${t.id} | ${g.length} | ${ok.filter((r) => r.success).length} | ${fail} | `
        + `${cell(ok.map((r) => Number(r.model_calls)), 0, caps.includes('steps'))} | ${cell(dec, 1, false)} | `
        + `${cell(ela, 0, caps.includes('wall'))} | ${share.toFixed(1)}% |`);
    }
  }
  const budgets = SUITE.map((t) => `${t.id} ${t.maxSteps < UNCAPPED_STEPS ? t.maxSteps : '∞'}/${t.maxWallS}s`).join(', ');
  const harnesses = [...new Set(rows.map((r) => `${r.harness}@${r.brain_commit ?? '?'} sdk ${r.strands_sdk ?? r.strands_commit ?? '?'}`))].join('; ');
  out.push('', 'Cells: median (min–max) per episode. `*` on a max: an episode hit that budget.', `Budgets (step cap/wall-clock cap): ${budgets}.`, `Harness: ${harnesses}.`);
  return out.join('\n');
}
