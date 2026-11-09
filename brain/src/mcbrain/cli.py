"""``strands-minecraft-brain`` — run the Python brain against a running sidecar.

    STRANDS_MODEL_PROVIDER=bedrock  strands-minecraft-brain            # REPL, Sonnet 5
    STRANDS_MODEL_PROVIDER=typesafe strands-minecraft-brain --goal "collect 16 logs"   # Jev as policy
    STRANDS_MODEL_PROVIDER=kev      strands-minecraft-brain --goal "…" --steps 40
    STRANDS_MODEL_PROVIDER=clm-sagemaker strands-minecraft-brain --goal "…"

An LLM provider runs the full tool-calling brain (REPL on stdin plus the event rails). A
decision provider (typesafe | kev | kev-sagemaker | clm | clm-sagemaker) runs the System One policy toward
``--goal`` for ``--steps`` decisions.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from .models import create_decision_model, create_llm, describe, resolve_brain_spec
from .policy import PolicyBrain
from .sidecar import Body


async def _policy(args: argparse.Namespace) -> int:
    spec = resolve_brain_spec()
    print(describe(spec))
    body = Body(args.body)
    brain = PolicyBrain(create_decision_model(spec), body, args.goal)
    try:
        for _ in range(args.steps):
            d = await brain.step()
            print(
                f"#{d.step} {d.action} conf={d.confidence} {d.latency_ms:.0f}ms {'ok' if d.ok else 'FAIL'} {d.outcome or d.error or ''}"[
                    :300
                ]
            )
    finally:
        body.close()
    return 0


async def _llm(args: argparse.Namespace) -> int:
    from .brain import Brain

    spec = resolve_brain_spec()
    print(describe(spec))
    body = Body(args.body)
    brain = Brain(create_llm(spec), body)
    brain.on_answer = lambda src, a: print(f"🤖 [{src}] {a}")
    stop = asyncio.Event()
    pump = asyncio.create_task(brain.pump(stop))
    brain.thinker.start()
    if args.goal:
        await brain.run(f"Start a journey for this goal and pursue it: {args.goal}")
    loop = asyncio.get_running_loop()
    try:
        while True:
            line = await loop.run_in_executor(None, sys.stdin.readline)
            if not line or line.strip() in ("exit", "quit"):
                break
            if line.strip():
                asyncio.create_task(brain.run(line.strip()))
    finally:
        stop.set()
        brain.thinker.stop()
        pump.cancel()
        body.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--body", default="StrandsBot", help="body name on the sidecar (MC_USERNAME)")
    parser.add_argument("--goal", default="", help="goal for the policy (decision models) or a first journey (LLMs)")
    parser.add_argument("--steps", type=int, default=30, help="decisions to take (decision models)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    spec = resolve_brain_spec()
    return asyncio.run(_policy(args) if spec.is_decision else _llm(args))


if __name__ == "__main__":
    raise SystemExit(main())
