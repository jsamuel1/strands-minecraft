"""Smoke-test the CLM endpoint: models, systemone (the README example), and rank.

    python smoke.py [--endpoint clm-v01-8b] [--profile $AWS_PROFILE] [--region us-west-2]
"""
import argparse
import json
import os
import time

import boto3

ap = argparse.ArgumentParser()
ap.add_argument("--endpoint", default="clm-v01-8b")
ap.add_argument("--profile", default=os.environ.get("AWS_PROFILE"))
ap.add_argument("--region", default="us-west-2")
a = ap.parse_args()
rt = boto3.Session(profile_name=a.profile, region_name=a.region).client("sagemaker-runtime")


def call(body):
    t0 = time.perf_counter()
    r = rt.invoke_endpoint(EndpointName=a.endpoint, ContentType="application/json", Body=json.dumps(body))
    out = json.loads(r["Body"].read())
    out["client_latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return out


print("models:", json.dumps(call({"path": "/v1/models"})))
print("systemone:", json.dumps(call({
    "state": "Customer: my invoice was charged twice and nobody answers the phone!",
    "model": "clm-latest",
    "questions": {
        "urgency": {"type": "noul", "instructions": "Is this urgent?"},
        "department": {"type": "choice", "instructions": "Which team should handle this?",
                       "criteria": {"billing": "Charges, invoices, refunds", "technical": "Bugs and outages"}},
        "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                        "criteria": ["Calm", "Frustrated", "Very angry"]},
    },
}), indent=1))
print("rank (minecraft):", json.dumps(call({
    "context": "Health 6/20, night, a zombie is 3 blocks away, holding a wooden sword, a cave entrance is to the north.",
    "question": "What should the bot do next?",
    "answers": ["attack the zombie", "flee into the cave", "eat bread", "place a block pillar and wait for day"],
}), indent=1))
