"""JSONL → the results table (Markdown). Every number comes from the episodes file."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _med(xs: list[float]) -> str:
    return f"{statistics.median(xs):.0f}" if xs else "–"


def table(rows: list[dict[str, Any]]) -> str:
    by: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by[(r["arm"], r["task"])].append(r)
    arms = list(dict.fromkeys(r["arm"] for r in rows))
    tasks = list(dict.fromkeys(r["task"] for r in rows))
    out = [
        "| arm | model (as answered) | task | n | success | progress | time-to-goal s (median) | steps | model calls | tool calls | deaths | tokens in/out | $ total | p50 / p95 ms | server p50 ms | errors | infra fails |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for arm in arms:
        for task in tasks:
            eps = by.get((arm, task), [])
            if not eps:
                continue
            ok = [e for e in eps if not e.get("error")]
            succ = sum(1 for e in ok if e["success"])
            ttg = [e["time_to_goal_s"] for e in ok if e.get("time_to_goal_s") is not None]
            costs = [e["cost_usd"] for e in ok if e.get("cost_usd") is not None]
            p50 = [e["latency_p50_ms"] for e in ok if e.get("latency_p50_ms") is not None]
            p95 = [e["latency_p95_ms"] for e in ok if e.get("latency_p95_ms") is not None]
            srv = [e["server_latency_p50_ms"] for e in ok if e.get("server_latency_p50_ms") is not None]
            errs: dict[str, int] = defaultdict(int)
            for e in ok:
                for k, v in (e.get("provider_errors") or {}).items():
                    errs[k] += v
            answered = sorted({m for e in ok for m in e.get("answering_models") or []}) or [eps[0]["model_label"]]
            cells = [
                arm,
                ", ".join(answered),
                task,
                str(len(ok)),
                f"{succ}/{len(ok)}",
                f"{statistics.mean(e['progress'] for e in ok):.2f}" if ok else "–",
                _med(ttg),
                _med([e["steps"] for e in ok]),
                _med([e["model_calls"] for e in ok]),
                _med([e["tool_calls"] for e in ok]),
                str(sum(e["deaths"] for e in ok)),
                f"{sum(e['tokens_in'] for e in ok):,} / {sum(e['tokens_out'] for e in ok):,}",
                f"{sum(costs):.4f}" if costs else "unknown",
                f"{_med(p50)} / {_med(p95)}",
                _med(srv),
                ", ".join(f"{k}×{v}" for k, v in errs.items()) or "0",
                str(len(eps) - len(ok)),
            ]
            out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)
