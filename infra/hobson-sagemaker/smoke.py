"""Smoke the Hobson endpoint: model card, typed request, and the policy's real step-1 request shape."""

import asyncio
import json
import os
import time

import boto3
from strands.experimental.decisions import Choice, Score, YesNo

from mcbrain.models import SageMakerSystemOneModel
from mcbrain.policy import INSTRUCTIONS

PROFILE, REGION, ENDPOINT = os.environ.get("AWS_PROFILE"), os.environ.get("HOBSON_REGION", "us-east-1"), "hobson-2b-v19"
sess = boto3.Session(profile_name=PROFILE, region_name=REGION)
v = sess.client("sagemaker").describe_endpoint(EndpointName=ENDPOINT)["ProductionVariants"][0]
print("placed on:", [p["InstanceType"] for p in v.get("InstancePools") or [] if p.get("CurrentInstanceCount")])
rt = sess.client("sagemaker-runtime")
card = json.loads(rt.invoke_endpoint(EndpointName=ENDPOINT, ContentType="application/json", Body=b"{}", CustomAttributes="path=/v1/models")["Body"].read())
print("model card:", card["models"][0])


async def main() -> None:
    m = SageMakerSystemOneModel(endpoint_name=ENDPOINT, region=REGION, profile=PROFILE, model_id="hobson-latest")
    q = {
        "department": Choice("Which team should handle this?", {"billing": "Charges, payments", "technical": "Bugs", "sales": "Pricing"}),
        "urgent": YesNo("Does this convey urgency?"),
        "frustration": Score("How frustrated is the writer?", ["Calm", "Frustrated", "Very angry"]),
    }
    r = await m.ask("Help! My payouts have been failing for 3 days.", q)
    print("typed:", {k: getattr(a, "choice", None) or getattr(a, "probability", None) or getattr(a, "score", None) for k, a in r.answers.items()})
    rows = [json.loads(line) for line in open("../bench-results/2026-09-28-v3/episodes.jsonl")]
    d = next(x for x in rows if x["arm"] == "jev" and x["task"] == "wood_tools" and x["episode"] == 1)["decisions"][0]
    state = {k: v for k, v in d["request"]["state"].items() if not (isinstance(v, (list, dict)) and not v)}
    lat, srv = [], []
    for _ in range(6):
        t = time.perf_counter()
        r = await m.ask(state, {"next_action": Choice(INSTRUCTIONS, d["request"]["options"])})
        lat.append((time.perf_counter() - t) * 1000)
        srv.append(m.last_server_latency_ms)
    a = r.answers["next_action"]
    print(f"policy step 1: {a.choice} conf={a.confidence:.2f} p(wait)={a.probabilities.get('wait', 0):.2f}")
    print("client ms", [round(x) for x in lat], "server ms", srv, "input tokens", r.usage)


asyncio.run(main())
