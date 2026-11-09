# CLM-8B (Contrastive-LM/CLM) as a SageMaker real-time endpoint (BYOC).
#
# Two processes plus a gateway, all inside one container:
#   vLLM   :8090  Qwen3-8B pooling encoder (last-token, prefix cache)
#   clm    :8700  clm-serve: the contrastive head plus the System One API
#   gateway :8080 SageMaker's contract: GET /ping, POST /invocations
#
# Weights are NOT baked in. SageMaker mounts ModelDataSource (an S3 prefix) at
# /opt/ml/model:
#   /opt/ml/model/qwen3-8b/        HF snapshot of Qwen/Qwen3-8B
#   /opt/ml/model/CLM_v0.1-8B.pt   reference head
FROM vllm/vllm-openai:v0.11.0

# CLM pinned to the commit this was validated against.
ARG CLM_REF=bb42c6c5bf914fd449bed2f6ca65be80602cb1f7
# --no-deps: vllm/torch come from the base image, and must not be re-resolved.
RUN pip install --no-cache-dir --no-deps "contrastive-lm @ git+https://github.com/Contrastive-LM/CLM@${CLM_REF}" \
 && pip install --no-cache-dir "fastapi>=0.100" "uvicorn>=0.23" "httpx>=0.27" "pyarrow>=14"

ENV HF_HUB_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1 \
    CLM_EMB_MAX_TOKENS=2048 \
    VLLM_GPU_UTIL=0.80

COPY gateway.py /opt/clm/gateway.py
COPY serve /usr/local/bin/serve
RUN chmod +x /usr/local/bin/serve

EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/serve"]
