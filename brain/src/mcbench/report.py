"""The human results table: per arm × task, median (min–max) per episode, budgets starred.

    mcbench report RUN_DIR [RUN_DIR …] [--label DIR=LABEL …] [--arms DIR=arm,arm …]

Columns: n, success, fail, decisions, decide s, elapsed s, decide %.

- decisions: model calls that chose what to do next (policy: one Choice per step; agent: model calls).
- decide s: time waiting on the model. v3 records it exactly for both arm kinds (``decide_s``); older
  rows without it are estimated (``~``) as model_calls × p50.
- elapsed s: brain loop time, from first step to the goal or the budget (``loop_s``). Older rows
  without it use time_to_goal_s, or wall_s minus the median setup of successful episodes (``≈``).
- ``*`` on a max: at least one episode in the row ran out of that task budget (decisions → step cap,
  elapsed → wall-clock cap).
- fail counts goal-not-reached; infrastructure failures are shown separately as ``+N infra``.
"""

from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from pathlib import Path
from typing import Any

from .tasks import BY_ID, UNCAPPED_STEPS

TASK_ORDER = [t for t in BY_ID]


def _load(path: str, label: str | None = None, arms: set[str] | None = None) -> list[dict[str, Any]]:
    """One run directory's episodes, tagged with its run label and filtered to ``arms``."""
    rows = [json.loads(line) for line in (Path(path) / "episodes.jsonl").read_text().splitlines() if line.strip()]
    tag = label or Path(path).name
    return [dict(r, _run=tag) for r in rows if arms is None or r["arm"] in arms]


#: Step caps in force BEFORE v3 recorded cap_hit, where they differ from today's suite. A row is
#: judged by the budget it actually ran under, never by a later one (v2 night was 150; v3 is uncapped).
LEGACY_STEP_CAPS: dict[str, int] = {"night": 150}


def _cap(r: dict[str, Any]) -> str | None:
    if r["success"] or r.get("error"):
        return None
    if r.get("cap_hit"):
        return str(r["cap_hit"])
    cap = LEGACY_STEP_CAPS.get(r["task"], BY_ID[r["task"]].max_steps)
    return "steps" if r["steps"] >= cap else "wall"


def _cell(xs: list[float], fmt: str, starred: bool) -> str:
    return f"{st.median(xs):{fmt}} ({min(xs):{fmt}}–{max(xs):{fmt}}{'*' if starred else ''})"


def table(rows: list[dict[str, Any]]) -> str:
    setup = [r["wall_s"] - r["time_to_goal_s"] for r in rows if r["success"] and r.get("time_to_goal_s") is not None]
    boot = st.median(setup) if setup else 0.0
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(r["_run"], r["arm"], r["task"])].append(r)
    out = [
        "| arm | run | task | n | success | fail | decisions | decide s | elapsed s | decide % |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    arms = list(dict.fromkeys((r["_run"], r["arm"]) for r in rows))
    for run, arm in arms:
        for task in TASK_ORDER:
            g = groups.get((run, arm, task))
            if not g:
                continue
            dec, exact_d = [], True
            for r in g:
                if r.get("decide_s") is not None:
                    dec.append(float(r["decide_s"]))
                elif r["kind"] == "policy":
                    dec.append(sum(d["ms"] for d in r["decisions"]) / 1000)
                else:
                    dec.append(r["model_calls"] * (r["latency_p50_ms"] or 0) / 1000)
                    exact_d = False
            ela, exact_e = [], True
            for r in g:
                if r.get("loop_s") is not None:
                    ela.append(float(r["loop_s"]))
                elif r["success"]:
                    ela.append(float(r["time_to_goal_s"]))
                else:
                    ela.append(max(0.0, r["wall_s"] - boot))
                    exact_e = False
            caps = [_cap(r) for r in g]
            infra = sum(bool(r.get("error")) for r in g)
            fail = str(sum(not r["success"] and not r.get("error") for r in g)) + (f"+{infra} infra" if infra else "")
            dm, em = ("" if exact_d else "~"), ("" if exact_e else "≈")
            share = 100 * sum(dec) / sum(ela) if sum(ela) else 0.0
            out.append(
                f"| {arm} | {run} | {task} | {len(g)} | {sum(r['success'] for r in g)} | {fail} | "
                f"{_cell([float(r['model_calls']) for r in g], '.0f', 'steps' in caps)} | "
                f"{dm}{_cell(dec, '.1f', False)} | {em}{_cell(ela, '.0f', 'wall' in caps)} | {dm}{share:.1f}% |"
            )
    budgets = ", ".join(f"{t.id} {t.max_steps if t.max_steps < UNCAPPED_STEPS else '∞'}/{t.max_wall_s}s" for t in BY_ID.values())
    out += [
        "",
        "Cells: median (min–max) per episode. `~` decide time estimated as calls × p50 (rows without exact timing).",
        f"`≈` elapsed estimated as wall − {boot:.0f} s setup (rows without loop_s). `*` on a max: an episode hit that budget.",
        f"Budgets (step cap/wall-clock cap): {budgets}.",
    ]
    return "\n".join(out)


def report(dirs: list[str], labels: dict[str, str] | None = None, arms: dict[str, set[str]] | None = None) -> str:
    rows: list[dict[str, Any]] = []
    for d in dirs:
        rows += _load(d, (labels or {}).get(d), (arms or {}).get(d))
    return table(rows)
