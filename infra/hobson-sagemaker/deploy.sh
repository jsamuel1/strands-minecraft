#!/bin/bash
# strands-decider / Hobson-2B v19 on SageMaker in your AWS account (AWS_PROFILE_HOBSON / AWS_PROFILE), us-west-2.
#   ./deploy.sh build     build + push the BYOC image (hobson @ HOBSON_REF) to ECR hobson-sagemaker
#   ./deploy.sh stage     checkpoint + pinned base -> s3://…/hobson/hobson-2b-v19/{checkpoint,base}/ (idempotent)
#   ./deploy.sh deploy    create model + endpoint config + endpoint, wait for InService
#   ./deploy.sh delete    delete the endpoint (it bills while it exists)
#   ./deploy.sh status
#
# Current public source: github.com/strands-labs/strands-decider. This benchmark image used pre-publication
# source at HOBSON_REF; pass HOBSON_SRC as a checkout containing that historical commit. The published v19
# adapter is at huggingface.co/StrandsAgents/strands-decider-2B-hobson-v19.
#
# 2B bf16 (~5 GB) fits any >=24 GB Ampere+/Ada GPU (fla Triton + bf16; T4 g4dn is excluded). 2026-10-01: g5.xlarge alone hit
# InsufficientInstanceCapacity, so the pool adds the >=24 GB types that sit in no other endpoint pool and have quota.
# CreateEndpoint checks quota for EVERY InstancePools entry, so only types with free quota are listed.
# The image's torch is cu126; pin the al2023 driver-580 AMI (the default g5 AMI is CUDA 11.4).
set -euo pipefail
PROFILE=${AWS_PROFILE_HOBSON:-${AWS_PROFILE:-default}}
REGION=${HOBSON_REGION:-us-west-2}
ACCOUNT=${HOBSON_ACCOUNT:-$(command aws --profile "$PROFILE" sts get-caller-identity --query Account --output text)}
REPO=hobson-sagemaker
HOBSON_REF=${HOBSON_REF:-4f1d7d0}
TAG=${HOBSON_IMAGE_TAG:-v19-$HOBSON_REF}
IMAGE=$ACCOUNT.dkr.ecr.$REGION.amazonaws.com/$REPO:$TAG
MODEL_S3=${HOBSON_MODEL_S3:-s3://sagemaker-$REGION-$ACCOUNT/hobson/hobson-2b-v19/}
ROLE=arn:aws:iam::$ACCOUNT:role/service-role/AmazonSageMakerAdminIAMExecutionRole
POOLS=${HOBSON_INSTANCE_POOLS:-ml.g5.xlarge ml.g6.xlarge ml.g6e.2xlarge ml.g6.4xlarge}
ENDPOINT=${HOBSON_ENDPOINT:-hobson-2b-v19}
AMI=${HOBSON_AMI:-al2023-ami-sagemaker-inference-gpu-4-1}
CKPT=${HOBSON_CKPT:-$HOME/src/hobson-models/hobson-2b-v19}
BASE=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B-Base/snapshots/b1485b2fa6dfa1287294f269f5fb618e03d52d7c
NAME=$ENDPOINT-$(date -u +%Y%m%d%H%M%S)
HERE=$(cd "$(dirname "$0")" && pwd)
aws() { command aws --profile "$PROFILE" --region "$REGION" "$@"; }

case "${1:-}" in
build)
  : "${HOBSON_SRC:?set HOBSON_SRC to a checkout containing benchmark source commit $HOBSON_REF}"
  aws ecr describe-repositories --repository-names "$REPO" >/dev/null 2>&1 || aws ecr create-repository --repository-name "$REPO" >/dev/null
  aws ecr get-login-password | docker login --username AWS --password-stdin "$ACCOUNT.dkr.ecr.$REGION.amazonaws.com" >/dev/null
  src=$HOBSON_SRC
  rm -rf "$HERE/hobson-src" && mkdir -p "$HERE/hobson-src"
  git -C "$src" archive "$HOBSON_REF" | tar -x -C "$HERE/hobson-src"   # exact commit, no .git, no credentials
  docker build --build-arg HOBSON_REF="$HOBSON_REF" -t "$IMAGE" "$HERE"
  rm -rf "$HERE/hobson-src"
  docker push "$IMAGE" | tail -1
  ;;
stage)
  aws s3 sync "$CKPT/" "${MODEL_S3}checkpoint/" --only-show-errors
  aws s3 sync "$BASE/" "${MODEL_S3}base/" --only-show-errors   # sync follows the HF snapshot symlinks
  aws s3 ls "$MODEL_S3" --recursive --summarize | tail -2
  ;;
deploy)
  pools=$(p=1; for t in $POOLS; do printf '{"InstanceType":"%s","Priority":%d},' "$t" $p; p=$((p+1)); done | sed 's/,$//')
  aws sagemaker create-model --model-name "$NAME" --execution-role-arn "$ROLE" \
    --primary-container "{\"Image\":\"$IMAGE\",\"ModelDataSource\":{\"S3DataSource\":{\"S3Uri\":\"$MODEL_S3\",\"S3DataType\":\"S3Prefix\",\"CompressionType\":\"None\"}},\"Environment\":{\"HOBSON_REF\":\"$HOBSON_REF\"}}" >/dev/null
  aws sagemaker create-endpoint-config --endpoint-config-name "$NAME" --production-variants "[{\"VariantName\":\"AllTraffic\",\"ModelName\":\"$NAME\",\"InstancePools\":[$pools],\"InitialInstanceCount\":1,
    \"InferenceAmiVersion\":\"$AMI\",\"ModelDataDownloadTimeoutInSeconds\":1200,\"ContainerStartupHealthCheckTimeoutInSeconds\":1200}]" >/dev/null
  if aws sagemaker describe-endpoint --endpoint-name "$ENDPOINT" >/dev/null 2>&1; then
    aws sagemaker update-endpoint --endpoint-name "$ENDPOINT" --endpoint-config-name "$NAME" >/dev/null
  else
    aws sagemaker create-endpoint --endpoint-name "$ENDPOINT" --endpoint-config-name "$NAME" >/dev/null
  fi
  echo "endpoint $ENDPOINT -> config $NAME (pools: $POOLS); waiting for InService..."
  aws sagemaker wait endpoint-in-service --endpoint-name "$ENDPOINT" || true
  aws sagemaker describe-endpoint --endpoint-name "$ENDPOINT" --query '[EndpointName,EndpointStatus,FailureReason]' --output text
  ;;
delete)
  aws sagemaker delete-endpoint --endpoint-name "$ENDPOINT"
  echo "deleted endpoint $ENDPOINT (models, configs, S3 weights are kept; they are free or cents)"
  ;;
status)
  aws sagemaker describe-endpoint --endpoint-name "$ENDPOINT" --query '[EndpointName,EndpointStatus,FailureReason]' --output text
  ;;
*) echo "usage: $0 build|stage|deploy|delete|status" >&2; exit 2 ;;
esac
