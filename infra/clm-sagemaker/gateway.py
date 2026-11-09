"""SageMaker contract for clm-serve.

GET  /ping         200 only when clm-serve reports ok and its embedder is healthy.
POST /invocations  body with "questions" -> /v1/systemone (TypeSafe wire format)
                   body with "answers"   -> /v1/rank
                   {"path": "/v1/models"} -> GET /v1/models
The response is clm-serve's JSON, unchanged. The server-side latency header
X-CLM-Latency-Ms is echoed into the body as "server_latency_ms", because
SageMaker's InvokeEndpoint drops custom response headers.
"""
from __future__ import annotations

import json

import httpx
from fastapi import FastAPI, Request, Response

CLM = "http://127.0.0.1:8700"
app = FastAPI()
client = httpx.AsyncClient(base_url=CLM, timeout=120.0)


@app.get("/ping")
async def ping() -> Response:
    try:
        r = await client.get("/health", timeout=5.0)
        h = r.json()
        ok = r.status_code == 200 and h.get("ok") and h.get("embedder")
    except Exception:  # noqa: BLE001 (clm-serve is not up yet)
        ok = False
    return Response(status_code=200 if ok else 503)


@app.post("/invocations")
async def invocations(request: Request) -> Response:
    try:
        body = await request.json()
    except Exception as e:  # noqa: BLE001
        return Response(json.dumps({"error": f"body is not JSON: {e}"}), 422, media_type="application/json")
    if isinstance(body, dict) and body.get("path") == "/v1/models":
        r = await client.get("/v1/models")
    elif isinstance(body, dict) and "answers" in body:
        r = await client.post("/v1/rank", json=body)
    else:
        r = await client.post("/v1/systemone", json=body)
    try:
        out = r.json()
    except ValueError:
        return Response(r.content, r.status_code, media_type="application/json")
    lat = r.headers.get("X-CLM-Latency-Ms")
    if isinstance(out, dict) and lat is not None:
        out["server_latency_ms"] = float(lat)
    return Response(json.dumps(out), r.status_code, media_type="application/json")
