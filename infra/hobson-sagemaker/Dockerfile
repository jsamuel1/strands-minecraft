# strands-decider / Hobson-2B v19 (github.com/strands-labs/strands-decider) as a SageMaker real-time endpoint (BYOC).
#
# One process: hobson's own FastAPI app (create_app, POST /v1/systemone) wrapped by a thin SageMaker front
# (GET /ping, POST /invocations). Weights are NOT baked in; SageMaker mounts ModelDataSource (S3 prefix):
#   /opt/ml/model/checkpoint/   hobson-2b-v19 package (lora/, slot_head.pt, hobson_config.json, tokenizer)
#   /opt/ml/model/base/         Qwen/Qwen3.5-2B-Base @ b1485b2f (the loader pins no revision; we do)
# The versions are the ones hobson's training/README.md pins for Linux + CUDA: torch 2.7.1 (cu126),
# transformers 5.17.0, peft 0.21.0, flash-linear-attention (its Triton kernels need CUDA; without them
# DeltaNet falls back to slow reference PyTorch).
FROM python:3.12-slim
ARG HOBSON_REF=4f1d7d0
# The repo is private; deploy.sh exports ~/src/hobson at HOBSON_REF (git archive) into the build
# context as hobson-src/, so no credential ever enters the image.
COPY hobson-src /opt/hobson
RUN apt-get update && apt-get install -y --no-install-recommends git build-essential && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126 \
 && pip install --no-cache-dir transformers==5.17.0 peft==0.21.0 flash-linear-attention \
 && pip install --no-cache-dir /opt/hobson
ENV HF_HUB_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 \
    TRITON_CACHE_DIR=/tmp/triton-cache HOBSON_REF=${HOBSON_REF}
COPY sm_serve.py /opt/program/sm_serve.py
WORKDIR /opt/program
EXPOSE 8080
ENTRYPOINT ["python", "/opt/program/sm_serve.py"]
