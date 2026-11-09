"""Smoke-test the Kev-4B endpoint: model card, a typed systemone request, and the policy's real request shape."""

import asyncio
import json
import os
import time

import boto3
from strands.experimental.decisions import Choice, Score, YesNo

from mcbrain.models import SageMakerSystemOneModel
from mcbrain.policy import INSTRUCTIONS, Observation, candidates, decision_state

PROFILE, REGION, ENDPOINT = os.environ.get("AWS_PROFILE"), "us-west-2", "kev-4b"
sess = boto3.Session(profile_name=PROFILE, region_name=REGION)
v = sess.client("sagemaker").describe_endpoint(EndpointName=ENDPOINT)["ProductionVariants"][0]
print("placed on:", [p["InstanceType"] for p in v.get("InstancePools") or [] if p.get("CurrentInstanceCount")] or v.get("InstanceType"))
rt = sess.client("sagemaker-runtime")
card = json.loads(rt.invoke_endpoint(EndpointName=ENDPOINT, ContentType="application/json", Body=b"{}", CustomAttributes="path=/v1/models")["Body"].read())
m0 = card["models"][0]
print("model card:", {k: m0.get(k) for k in ("name", "run", "base", "device", "dtype", "temperature", "backend")})


async def main() -> None:
    m = SageMakerSystemOneModel(endpoint_name=ENDPOINT, region=REGION, profile=PROFILE, model_id="kev-latest")
    q = {
        "department": Choice("Which team should handle this?", {"returns": "Exchanges, refunds", "shipping": "Delivery, delays", "billing": "Charges, payments"}),
        "escalate": YesNo("Does this need urgent human attention?"),
        "frustration": Score("How frustrated is the customer?", ["Calm", "Frustrated", "Very angry"]),
    }
    r = await m.ask("My invoice was charged twice and nobody answers the phone!", q)
    print("systemone:", {k: (getattr(a, "choice", None) or getattr(a, "probability", None) or getattr(a, "score", None)) for k, a in r.answers.items()}, "model", r.model_id, f"client {m.last_client_latency_ms:.0f} ms")
    tree = [{"position": {"x": 30, "y": 64, "z": 75}, "distance": 5.2, "veinSize": 6, "exposed": True, "block": "oak_log"}]
    ob = Observation(state={"pos": {"x": 26.5, "y": 64, "z": 71.5}, "health": 20, "food": 20, "inv": {}, "isDay": True}, trees=tree)
    acts = candidates(ob)
    lat = []
    for _ in range(5):
        t = time.perf_counter()
        r = await m.ask(decision_state("Collect 16 logs (any wood type).", ob, []), {"next_action": Choice(INSTRUCTIONS, {a.id: a.description for a in acts})})
        lat.append((time.perf_counter() - t) * 1000)
    a = r.answers["next_action"]
    print(f"policy: {a.choice} conf={a.confidence:.2f}; client ms {[round(x) for x in lat]}")


asyncio.run(main())
