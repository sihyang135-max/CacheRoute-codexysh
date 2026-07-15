#!/usr/bin/env bash
# Run this script on the Docker host, not inside the CacheRoute container.
set -euo pipefail

: "${PROJECT_HOST:?Set PROJECT_HOST to the host path of cacheroute-rl-main}"
: "${MODEL_DIR:?Set MODEL_DIR to the model directory visible inside the container}"
: "${MODEL_NAME:?Set MODEL_NAME, e.g. llama3-8b}"

IMAGE="${IMAGE:-cacheroute:v0.1.7}"
CONTAINER="${CONTAINER:-cacheroute-rl}"
REDIS_CONTAINER="${REDIS_CONTAINER:-lmcache-redis}"
PROJECT_IN_CONTAINER="${PROJECT_IN_CONTAINER:-/workspace/llm-stack/CacheRoute}"

if ! docker inspect "$REDIS_CONTAINER" >/dev/null 2>&1; then
  docker run -d --name "$REDIS_CONTAINER" --network host redis:7 \
    redis-server --bind 0.0.0.0 --protected-mode no --save "" \
    --appendonly no --maxmemory 200gb --maxmemory-policy allkeys-lru
elif [ "$(docker inspect -f '{{.State.Running}}' "$REDIS_CONTAINER")" != "true" ]; then
  docker start "$REDIS_CONTAINER"
fi

if ! docker inspect "$CONTAINER" >/dev/null 2>&1; then
  docker run -d --gpus all --name "$CONTAINER" --network host --ipc=host \
    --shm-size=64g --ulimit memlock=-1 --ulimit stack=67108864 \
    --memory=0 --memory-swap=0 \
    -v /llm-stack:/workspace/llm-stack \
    -v "$PROJECT_HOST:$PROJECT_IN_CONTAINER" \
    "$IMAGE" sleep infinity
elif [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER")" != "true" ]; then
  docker start "$CONTAINER"
fi

docker exec -e MODEL_DIR="$MODEL_DIR" -e MODEL_NAME="$MODEL_NAME" \
  -e PROJECT="$PROJECT_IN_CONTAINER" -e PREWARM_COUNT="${PREWARM_COUNT:-0}" \
  -e INSTANCE_COUNT="${INSTANCE_COUNT:-4}" -e TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}" \
  -e INSTANCE_GPU_GROUPS="${INSTANCE_GPU_GROUPS:-}" \
  -e INSTANCE_TP_SIZES="${INSTANCE_TP_SIZES:-}" \
  -e PROXY_INSTANCE_STRATEGY="${PROXY_INSTANCE_STRATEGY:-linucb}" \
  -e PROXY_RL_ENABLED="${PROXY_RL_ENABLED:-1}" \
  -e PROXY_RL_ALPHA="${PROXY_RL_ALPHA:-0.4}" \
  -e PROXY_RL_LAMBDA="${PROXY_RL_LAMBDA:-1.0}" \
  -e PROXY_RL_WARMUP_REQUESTS="${PROXY_RL_WARMUP_REQUESTS:-30}" \
  -e PROXY_RL_REWARD_TTFT_SCALE_MS="${PROXY_RL_REWARD_TTFT_SCALE_MS:-1000.0}" \
  -e PROXY_RL_REWARD_CLIP="${PROXY_RL_REWARD_CLIP:-5.0}" \
  -e SCHEDULER_EMBEDDING_MODEL="${SCHEDULER_EMBEDDING_MODEL:-/workspace/llm-stack/models/intfloat/multilingual-e5-large-instruct}" \
  -e LMCACHE_CONFIG_FILE="${LMCACHE_CONFIG_FILE:-/workspace/llm-stack/config/lmcache_with_redis.yaml}" \
  "$CONTAINER" bash "$PROJECT_IN_CONTAINER/scripts/start_rl_4instance_in_container.sh"

echo "Started. Logs: $PROJECT_IN_CONTAINER/log/rl4/"
echo "Check: docker exec -it $CONTAINER bash -lc 'cat $PROJECT_IN_CONTAINER/log/rl4/status.txt'"
