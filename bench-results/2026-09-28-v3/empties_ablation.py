"""Step-1 ablation: does dropping empty containers from the state change each policy model's pick?

Replays the exact recorded step-1 requests of the v3 run (same options, same instruction) against
the live models. Read-only: no world, no body.
"""

import asyncio
import json
import os

from strands.experimental.decisions import Choice

from mcbrain.models import create_decision_model, resolve_brain_spec
from mcbrain.policy import INSTRUCTIONS

RUN = "../bench-results/2026-09-28-v3/episodes.jsonl"
ARMS = {
    "jev": {"STRANDS_MODEL_PROVIDER": "typesafe"},
    "kev": {"STRANDS_MODEL_PROVIDER": "kev-sagemaker"},
    "clm": {"STRANDS_MODEL_PROVIDER": "clm-sagemaker"},
}


def drop_empties(s: dict) -> dict:
    return {k: v for k, v in s.items() if not (isinstance(v, (list, dict)) and not v)}


async def main() -> None:
    rows = [json.loads(line) for line in open(RUN)]
    for arm, env in ARMS.items():
        model = create_decision_model(resolve_brain_spec({**os.environ, **env}))
        for task in ("logs16", "wood_tools", "cobble8", "iron1"):
            for ep in (1, 2, 3):
                d = next(r for r in rows if r["arm"] == arm and r["task"] == task and r["episode"] == ep)["decisions"][0]
                state, opts = d["request"]["state"], d["request"]["options"]
                out = []
                for s in (state, drop_empties(state)):
                    a = (await model.ask(s, {"next_action": Choice(INSTRUCTIONS, opts)})).answers["next_action"]
                    out.append(f"{a.choice}({a.probabilities.get('wait', 0):.2f})")
                print(f"{arm} {task} {ep}: as-run {out[0]:24s} empties-dropped {out[1]}")


asyncio.run(main())
