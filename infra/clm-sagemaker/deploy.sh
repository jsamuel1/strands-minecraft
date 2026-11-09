#!/bin/bash
# Build, push and deploy CLM-8B to SageMaker in your AWS account (AWS_PROFILE_CLM / AWS_PROFILE), us-west-2.
#   ./deploy.sh build     build the image and push it to ECR
#   ./deploy.sh deploy    create the model, endpoint config and endpoint, then wait for InService
#   ./deploy.sh delete    delete the endpoint (it bills while it exists)
#   ./deploy.sh status
set -euo pipefail
PROFILE=${AWS_PROFILE_CLM:-${AWS_PROFILE:-default}}
REGION=${CLM_REGION:-us-west-2}
ACCOUNT=${CLM_ACCOUNT:-$(command aws --profile "$PROFILE" sts get-caller-identity --query Account --output text)}
REPO=clm-sagemaker
TAG=${CLM_TAG:-v0.1-8b-bb42c6c}
IMAGE=$ACCOUNT.dkr.ecr.$REGION.amazonaws.com/$REPO:$TAG
MODEL_S3=${CLM_MODEL_S3:-s3://sagemaker-$REGION-$ACCOUNT/clm/v0.1-8b/}
ROLE=arn:aws:iam::$ACCOUNT:role/service-role/AmazonSageMakerAdminIAMExecutionRole
# Priority-ordered fallback: us-west-2 has had InsufficientInstanceCapacity on
# g6e.xlarge and g5.2xlarge (2026-09-25). Every type has >=24 GB GPU, enough for
# Qwen3-8B bf16 pooling plus the head, and had endpoint quota in the benchmark account.
POOLS=${CLM_INSTANCE_POOLS:-ml.g6e.xlarge ml.g6.2xlarge ml.g5.2xlarge ml.g5.xlarge}
ENDPOINT=${CLM_ENDPOINT:-clm-v01-8b}
NAME=$ENDPOINT-$(date -u +%Y%m%d%H%M%S)
aws() { command aws --profile "$PROFILE" --region "$REGION" "$@"; }
cd "$(dirname "$0")"

case "${1:-}" in
build)
  aws ecr describe-repositories --repository-names $REPO >/dev/null 2>&1 || aws ecr create-repository --repository-name $REPO >/dev/null
  aws ecr get-login-password | docker login --username AWS --password-stdin "$ACCOUNT.dkr.ecr.$REGION.amazonaws.com"
  docker build -t "$IMAGE" .
  docker push "$IMAGE"
  ;;
deploy)
  aws sagemaker create-model --model-name "$NAME" --execution-role-arn "$ROLE" --primary-container "$(cat <<EOF
{"Image":"$IMAGE","ModelDataSource":{"S3DataSource":{"S3Uri":"$MODEL_S3","S3DataType":"S3Prefix","CompressionType":"None"}}}
EOF
)" >/dev/null
  pools=$(p=1; for t in $POOLS; do printf '%s{"InstanceType":"%s","Priority":%d}' "$([ $p -gt 1 ] && echo ,)" "$t" $p; p=$((p+1)); done)
  aws sagemaker create-endpoint-config --endpoint-config-name "$NAME" --production-variants "$(cat <<EOF
[{"VariantName":"AllTraffic","ModelName":"$NAME","InstancePools":[$pools],"InitialInstanceCount":1,
  "ContainerStartupHealthCheckTimeoutInSeconds":1200,"ModelDataDownloadTimeoutInSeconds":1200}]
EOF
)" >/dev/null
  if aws sagemaker describe-endpoint --endpoint-name "$ENDPOINT" >/dev/null 2>&1; then
    aws sagemaker update-endpoint --endpoint-name "$ENDPOINT" --endpoint-config-name "$NAME" >/dev/null
  else
    aws sagemaker create-endpoint --endpoint-name "$ENDPOINT" --endpoint-config-name "$NAME" >/dev/null
  fi
  echo "endpoint $ENDPOINT -> config $NAME (pools: $POOLS); waiting for InService..."
  aws sagemaker wait endpoint-in-service --endpoint-name "$ENDPOINT"
  aws sagemaker describe-endpoint --endpoint-name "$ENDPOINT" --query '[EndpointName,EndpointStatus,FailureReason]' --output text
  ;;
delete)
  aws sagemaker delete-endpoint --endpoint-name "$ENDPOINT"
  echo "deleted endpoint $ENDPOINT (models and configs are kept; they are free)"
  ;;
status)
  aws sagemaker describe-endpoint --endpoint-name "$ENDPOINT" --query '[EndpointName,EndpointStatus,FailureReason]' --output text
  ;;
*) echo "usage: $0 build|deploy|delete|status" >&2; exit 2 ;;
esac
