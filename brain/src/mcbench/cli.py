"""``mcbench`` — run arms × tasks × episodes, one fresh world each, and write JSONL + a table.

    mcbench run --arms jev,kev,clm,sonnet5,terra --tasks logs16,wood_tools --episodes 3 --out ../bench-results/2026-09-26
    mcbench summarize ../bench-results/2026-09-26/episodes.jsonl

Arms (``kind`` is ``policy`` = System One Choice over code-built macro-actions, ``agent`` = LLM
tool calling over the 54 MCP tools):
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time
from pathlib import Path

from mcbrain.models import BrainSpec, resolve_brain_spec

from .report import report
from .runner import append_jsonl, run_episode
from .summary import load, table
from .tasks import BY_ID

ARMS: dict[str, tuple[str, dict[str, str]]] = {
    "jev": ("policy", {"STRANDS_MODEL_PROVIDER": "typesafe", "STRANDS_MODEL_ID": "jev-latest"}),
    # Kev-4B on SageMaker (infra/kev-sagemaker). The v1 run (2026-09-26-v1) served it from this box's CPU.
    "kev": ("policy", {"STRANDS_MODEL_PROVIDER": "kev-sagemaker", "STRANDS_MODEL_ID": "kev-latest"}),
    "kev-cpu": ("policy", {"STRANDS_MODEL_PROVIDER": "kev", "STRANDS_MODEL_ID": "kev-latest"}),
    "clm": ("policy", {"STRANDS_MODEL_PROVIDER": "clm-sagemaker", "STRANDS_MODEL_ID": "clm-latest"}),
    # strands-decider (Hobson-2B v19, github.com/strands-labs/strands-decider) on SageMaker (infra/hobson-sagemaker).
    "hobson": ("policy", {"STRANDS_MODEL_PROVIDER": "hobson-sagemaker", "STRANDS_MODEL_ID": "hobson-latest"}),
    "sonnet5": ("agent", {"STRANDS_MODEL_PROVIDER": "bedrock", "STRANDS_MODEL_ID": "global.anthropic.claude-sonnet-5"}),
    "terra": ("agent", {"STRANDS_MODEL_PROVIDER": "mantle", "STRANDS_MODEL_ID": "openai.gpt-5.6-terra"}),
    # Same-alphabet controls: the LLMs as the policy, over the SAME macro-actions the decision models get.
    "sonnet5-policy": ("policy", {"STRANDS_MODEL_PROVIDER": "bedrock", "STRANDS_MODEL_ID": "global.anthropic.claude-sonnet-5"}),
    "terra-policy": ("policy", {"STRANDS_MODEL_PROVIDER": "mantle", "STRANDS_MODEL_ID": "openai.gpt-5.6-terra"}),
}
__doc__ = (__doc__ or "") + "\n".join(
    f"    {k:15s} {kind:6s} {env['STRANDS_MODEL_PROVIDER']}:{env['STRANDS_MODEL_ID']}" for k, (kind, env) in ARMS.items()
)


def spec_for(arm: str) -> tuple[str, BrainSpec]:
    kind, env = ARMS[arm]
    return kind, resolve_brain_spec({**os.environ, **env})


async def _run(args: argparse.Namespace) -> int:
    out = Path(args.out)
    jsonl = out / "episodes.jsonl"
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    base = int(args.port_base)
    k = 0
    for task_id in args.tasks.split(","):
        task = BY_ID[task_id]
        for n in range(1, args.episodes + 1):
            for arm in args.arms.split(","):
                kind, spec = spec_for(arm)
                ports = (base + 3 * k, base + 3 * k + 1, base + 3 * k + 2)
                k += 1
                print(f"▶ {arm} ({spec.label}) · {task.id} · episode {n}", flush=True)
                ep = await run_episode(run_id, arm, kind, spec, task, n, ports, logs=out / "logs")
                append_jsonl(jsonl, ep)
                status = "✅" if ep.success else ("💥 " + (ep.error or "")[:120] if ep.error else "❌")
                print(
                    f"  {status} progress={ep.progress} steps={ep.steps} wall={ep.wall_s}s p50={ep.latency_p50_ms}ms cost={ep.cost_usd} actions={' '.join(ep.actions[-8:])}",
                    flush=True,
                )
    (out / "summary.md").write_text(table(load(jsonl)) + "\n")
    (out / "stats.md").write_text(report([str(out)]) + "\n")
    print((out / "summary.md").read_text())
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--arms", default="jev,kev,clm,sonnet5,terra")
    r.add_argument("--tasks", default="logs16,wood_tools,cobble8")
    r.add_argument("--episodes", type=int, default=1)
    r.add_argument("--out", default="../bench-results/latest")
    r.add_argument("--port-base", default="26000")
    s = sub.add_parser("summarize")
    s.add_argument("jsonl")
    rp = sub.add_parser("report", help="median (min–max) table across one or more run directories")
    rp.add_argument("dirs", nargs="+")
    rp.add_argument("--label", action="append", default=[], help="DIR=LABEL")
    rp.add_argument("--arms", action="append", default=[], help="DIR=arm,arm (only these arms from DIR)")
    args = p.parse_args()
    if args.cmd == "report":
        labels = dict(x.split("=", 1) for x in args.label)
        arms = {d: set(a.split(",")) for d, a in (x.split("=", 1) for x in args.arms)}
        print(report(args.dirs, labels, arms))
        return 0
    if args.cmd == "summarize":
        print(table(load(Path(args.jsonl))))
        return 0
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
