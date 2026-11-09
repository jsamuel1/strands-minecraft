#!/bin/bash
# Kev-4B (jaredpalmer/kev-4b on Qwen/Qwen3.5-4B-Base) on SageMaker in your AWS account (AWS_PROFILE_KEV / AWS_PROFILE), us-west-2.
#   ./deploy.sh stage     weights -> s3://…/kev/kev-4b-139fdd94/{adapter,base}/ (idempotent)
#   ./deploy.sh deploy    create model + endpoint config + endpoint, wait for InService
#   ./deploy.sh delete    delete the endpoint (it bills while it exists)
#   ./deploy.sh status
#
# Image: set KEV_IMAGE_TAG to an image containing a model-agnostic Kev server: backbone from
# /opt/ml/model/base, adapter + head from /opt/ml/model/adapter, loaded bf16 onto one GPU with
# kev.serve's CUDA serving defaults (CUDA graphs + fused Qwen3.5 kernels). KEV_MODEL only labels /v1/models.
# 4B bf16 is ~9 GB resident, so any >=24 GB GPU fits; InstancePools falls back through the types that
# had capacity and quota in this account for CLM (2026-09-25).
set -euo pipefail
PROFILE=${AWS_PROFILE_KEV:-${AWS_PROFILE:-default}}
REGION=${KEV_REGION:-us-west-2}
ACCOUNT=${KEV_ACCOUNT:-$(command aws --profile "$PROFILE" sts get-caller-identity --query Account --output text)}
IMAGE=$ACCOUNT.dkr.ecr.$REGION.amazonaws.com/kev-sagemaker:${KEV_IMAGE_TAG:-27b-1b62aa2d}
MODEL_S3=${KEV_MODEL_S3:-s3://sagemaker-$REGION-$ACCOUNT/kev/kev-4b-139fdd94/}
ROLE=arn:aws:iam::$ACCOUNT:role/service-role/AmazonSageMakerAdminIAMExecutionRole
# Only types with free endpoint quota (2026-09-28): CreateEndpoint checks quota for EVERY pool entry, and
# the CLM endpoint's pool already reserves g6e.xlarge, g6.2xlarge and g5.2xlarge (quota 1, 1, 2 with Whisper on
# the other g5.2xlarge). g5.xlarge (A10G 24 GB, quota 4) and g5.4xlarge (A10G, quota 2) have room.
POOLS=${KEV_INSTANCE_POOLS:-ml.g5.xlarge ml.g5.4xlarge}
ENDPOINT=${KEV_ENDPOINT:-kev-4b}
# The image's torch is cu128. The default g5 host AMI has driver 470 / CUDA 11.4, and the container
# died with "NVIDIA driver on your system is too old (found version 11040)" (2026-09-28). The
# al2023 gpu-4-1 AMI has driver 580 / CUDA 13.0.
AMI=${KEV_AMI:-al2023-ami-sagemaker-inference-gpu-4-1}
HUB=${HF_HUB:-$HOME/.cache/huggingface/hub}
ADAPTER=$HUB/models--jaredpalmer--kev-4b/snapshots/139fdd94f1b6a6ad80cc15e08fcb99cac885a101
BASE=$HUB/models--Qwen--Qwen3.5-4B-Base/snapshots/1001bb4d826a52d1f399e183466143f4da7b741b
NAME=$ENDPOINT-$(date -u +%Y%m%d%H%M%S)
aws() { command aws --profile "$PROFILE" --region "$REGION" "$@"; }

case "${1:-}" in
stage)
  # sync follows the HF snapshot symlinks into real files
  aws s3 sync "$ADAPTER/" "${MODEL_S3}adapter/" --only-show-errors
  aws s3 sync "$BASE/" "${MODEL_S3}base/" --only-show-errors
  aws s3 ls --recursive --summarize "$MODEL_S3" | tail -2
  ;;
deploy)
  aws sagemaker create-model --model-name "$NAME" --execution-role-arn "$ROLE" --primary-container "$(cat <<EOF
{"Image":"$IMAGE","ModelDataSource":{"S3DataSource":{"S3Uri":"$MODEL_S3","S3DataType":"S3Prefix","CompressionType":"None"}},
 "Environment":{"KEV_MODEL":"jaredpalmer/kev-4b","KEV_RELEASE_DATE":"2026-09-24","KEV_SHARD":"0"}}
EOF
)" >/dev/null
  pools=$(p=1; for t in $POOLS; do printf '%s{"InstanceType":"%s","Priority":%d}' "$([ $p -gt 1 ] && echo ,)" "$t" $p; p=$((p+1)); done)
  aws sagemaker create-endpoint-config --endpoint-config-name "$NAME" --production-variants "$(cat <<EOF
[{"VariantName":"AllTraffic","ModelName":"$NAME","InstancePools":[$pools],"InitialInstanceCount":1,
  "InferenceAmiVersion":"$AMI",
  "ContainerStartupHealthCheckTimeoutInSeconds":1800,"ModelDataDownloadTimeoutInSeconds":1200}]
EOF
)" >/dev/null
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
*) echo "usage: $0 stage|deploy|delete|status" >&2; exit 2 ;;
esac
