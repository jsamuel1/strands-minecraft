"""SageMaker front for hobson: GET /ping, POST /invocations → hobson's own /v1/systemone engine.

The checkpoint's hobson_config.json names ``Qwen/Qwen3.5-2B-Base``; with HF_HUB_OFFLINE the loader would
fail to find it, so the config is rewritten (in a copy) to point at the S3-mounted base snapshot. Nothing
else about the model changes. ``latency_ms`` (hobson's own server-side timing) is echoed as
``server_latency_ms`` because InvokeEndpoint drops custom response headers.
"""

from __future__ import annotations

import json
import os
import shutil
import time

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

MODEL = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
CKPT = os.path.join(MODEL, "checkpoint")
BASE = os.path.join(MODEL, "base")
LOCAL = "/tmp/hobson-checkpoint"


def local_checkpoint() -> str:
    """A writable copy of the checkpoint whose config points at the mounted base weights."""
    shutil.copytree(CKPT, LOCAL, dirs_exist_ok=True)
    cfg_path = os.path.join(LOCAL, "hobson_config.json")
    cfg = json.load(open(cfg_path))
    cfg["base_model"] = BASE
    json.dump(cfg, open(cfg_path, "w"), indent=2)
    return LOCAL


from hobson.infer import EngineConfig, SystemOneEngine  # noqa: E402
from hobson.modeling import HobsonModel  # noqa: E402
from hobson.schema import SystemOneRequest  # noqa: E402

t0 = time.perf_counter()
model = HobsonModel.load(local_checkpoint())
engine = SystemOneEngine(model, EngineConfig(device="cuda", use_prefix_cache=True, model_name="hobson-2b-v19"))
LOAD_S = round(time.perf_counter() - t0, 1)
print(f"hobson loaded in {LOAD_S}s (base {BASE}, ref {os.environ.get('HOBSON_REF')})", flush=True)

app = FastAPI()


@app.get("/ping")
def ping() -> dict:
    return {"status": "ok", "model": "hobson-2b-v19", "load_s": LOAD_S}


@app.post("/invocations")
async def invocations(request: Request) -> JSONResponse:
    body = await request.json()
    attrs = request.headers.get("X-Amzn-SageMaker-Custom-Attributes", "")
    if "path=/v1/models" in attrs:
        c = engine.model.config
        return JSONResponse(
            {
                "models": [
                    {
                        "name": "hobson-2b-v19",
                        "base": c.base_model,
                        "max_length": c.max_length,
                        "num_slots": c.num_slots,
                        "temperature_by_kind": getattr(c, "temperature_by_kind", None),
                        "device": engine.cfg.device,
                        "ref": os.environ.get("HOBSON_REF"),
                    }
                ]
            }
        )
    started = time.perf_counter()
    try:
        response = engine.evaluate(SystemOneRequest.model_validate(body))
    except ValueError as exc:
        return JSONResponse({"detail": str(exc)}, status_code=422)
    payload = response.model_dump()
    payload["server_latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
    return JSONResponse(payload)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, workers=1)
