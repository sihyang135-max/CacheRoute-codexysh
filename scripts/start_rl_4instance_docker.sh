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
EXPECTED_COMMIT="${EXPECTED_COMMIT:-}"

PROJECT_HOST="$(readlink -f "$PROJECT_HOST")"
if [ -z "$EXPECTED_COMMIT" ] && git -C "$PROJECT_HOST" rev-parse HEAD >/dev/null 2>&1; then
  EXPECTED_COMMIT="$(git -C "$PROJECT_HOST" rev-parse HEAD)"
fi

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

mounted_project="$(docker inspect -f '{{range .Mounts}}{{println .Source "|" .Destination}}{{end}}' "$CONTAINER" \
  | awk -F ' \\| ' -v target="$PROJECT_IN_CONTAINER" '$2 == target {print $1; exit}')"
if [ -z "$mounted_project" ]; then
  echo "[FAIL] container $CONTAINER does not mount $PROJECT_IN_CONTAINER" >&2
  exit 2
fi
mounted_project="$(readlink -f "$mounted_project")"
if [ "$mounted_project" != "$PROJECT_HOST" ]; then
  echo "[FAIL] container $CONTAINER project mount mismatch" >&2
  echo "expected=$PROJECT_HOST" >&2
  echo "actual=$mounted_project" >&2
  exit 2
fi

docker exec -e MODEL_DIR="$MODEL_DIR" -e MODEL_NAME="$MODEL_NAME" \
  -e PROJECT="$PROJECT_IN_CONTAINER" -e PREWARM_COUNT="${PREWARM_COUNT:-0}" \
  -e EXPECTED_COMMIT="$EXPECTED_COMMIT" -e REQUIRE_CLEAN_WORKTREE=1 \
  -e INSTANCE_COUNT="${INSTANCE_COUNT:-4}" -e TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}" \
  -e INSTANCE_GPU_GROUPS="${INSTANCE_GPU_GROUPS:-}" \
  -e INSTANCE_TP_SIZES="${INSTANCE_TP_SIZES:-}" \
  -e INSTANCE_PREFILL_CAPACITY_RATIOS="${INSTANCE_PREFILL_CAPACITY_RATIOS:-}" \
  -e PROXY_INSTANCE_STRATEGY="${PROXY_INSTANCE_STRATEGY:-linucb}" \
  -e PROXY_RL_ENABLED="${PROXY_RL_ENABLED:-1}" \
  -e PROXY_RL_ALPHA="${PROXY_RL_ALPHA:-0.4}" \
  -e PROXY_RL_LAMBDA="${PROXY_RL_LAMBDA:-1.0}" \
  -e PROXY_RL_WARMUP_REQUESTS="${PROXY_RL_WARMUP_REQUESTS:-30}" \
  -e PROXY_RL_REWARD_TTFT_SCALE_MS="${PROXY_RL_REWARD_TTFT_SCALE_MS:-1000.0}" \
  -e PROXY_RL_REWARD_CLIP="${PROXY_RL_REWARD_CLIP:-5.0}" \
  -e PROXY_RL_COMPUTE_COST_SCALE_MS="${PROXY_RL_COMPUTE_COST_SCALE_MS:-1000.0}" \
  -e PROXY_RL_KV_READY_COST_SCALE_MS="${PROXY_RL_KV_READY_COST_SCALE_MS:-1000.0}" \
  -e PROXY_RL_KV_RESIDENCY_SCOPE="${PROXY_RL_KV_RESIDENCY_SCOPE:-global}" \
  -e PROXY_RL_KV_LINK_SCOPE="${PROXY_RL_KV_LINK_SCOPE:-global}" \
  -e PROXY_RL_MODEL_LOAD_PATH="${PROXY_RL_MODEL_LOAD_PATH:-}" \
  -e PROXY_RL_MODEL_SAVE_PATH="${PROXY_RL_MODEL_SAVE_PATH:-}" \
  -e PROXY_RL_SOURCE_COMMIT="${PROXY_RL_SOURCE_COMMIT:-}" \
  -e PROXY_RL_FROZEN="${PROXY_RL_FROZEN:-0}" \
  -e PROXY_RL_PARAMETER_SNAPSHOT_PATH="${PROXY_RL_PARAMETER_SNAPSHOT_PATH:-}" \
  -e PROXY_RL_PARAMETER_SNAPSHOT_INTERVAL="${PROXY_RL_PARAMETER_SNAPSHOT_INTERVAL:-20}" \
  -e SCHEDULER_EMBEDDING_MODEL="${SCHEDULER_EMBEDDING_MODEL:-/workspace/llm-stack/models/intfloat/multilingual-e5-large-instruct}" \
  -e SCHEDULER_CUDA_VISIBLE_DEVICES="${SCHEDULER_CUDA_VISIBLE_DEVICES:-}" \
  -e KDN_EMBEDDING_MODEL="${KDN_EMBEDDING_MODEL:-${SCHEDULER_EMBEDDING_MODEL:-/workspace/llm-stack/models/intfloat/multilingual-e5-large-instruct}}" \
  -e KDN_CUDA_VISIBLE_DEVICES="${KDN_CUDA_VISIBLE_DEVICES:-}" \
  -e KDN_BACKFILL_EMBEDDINGS="${KDN_BACKFILL_EMBEDDINGS:-0}" \
  -e HEALTH_PASSES="${HEALTH_PASSES:-3}" \
  -e HEALTH_INTERVAL_S="${HEALTH_INTERVAL_S:-10}" \
  -e LMCACHE_CONFIG_FILE="${LMCACHE_CONFIG_FILE:-/workspace/llm-stack/config/lmcache_with_redis.yaml}" \
  "$CONTAINER" bash "$PROJECT_IN_CONTAINER/scripts/start_rl_4instance_in_container.sh"

echo "Started. Logs: $PROJECT_IN_CONTAINER/log/rl4/"
echo "Check: docker exec -it $CONTAINER bash -lc 'cat $PROJECT_IN_CONTAINER/log/rl4/status.txt'"
