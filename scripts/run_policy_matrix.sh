#!/usr/bin/env bash
# Run on the Docker host. Every measured cell restarts the serving stack,
# performs an explicit warm-up phase, and writes one JSONL file.
set -euo pipefail

: "${PROJECT_HOST:?Set PROJECT_HOST to the host CacheRoute checkout}"
: "${MODEL_DIR:?Set MODEL_DIR to the model directory visible in the container}"
: "${MODEL_NAME:?Set MODEL_NAME to the served model name}"

CONTAINER="${CONTAINER:-cacheroute-rl}"
REDIS_CONTAINER="${REDIS_CONTAINER:-lmcache-redis}"
PROJECT_IN_CONTAINER="${PROJECT_IN_CONTAINER:-/workspace/llm-stack/CacheRoute}"
BASE_URL="${BASE_URL:-http://127.0.0.1:7001}"
WORKLOAD_FILE="${WORKLOAD_FILE:-$PROJECT_IN_CONTAINER/client/taskset/workload_nq.json}"

STRATEGIES="${STRATEGIES:-round_robin,least_inflight,linucb}"
CONCURRENCIES="${CONCURRENCIES:-1,4,8,16}"
REPEATS="${REPEATS:-3}"
REQUESTS="${REQUESTS:-100}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-200}"
MAX_TOKENS="${MAX_TOKENS:-64}"
INJECTION_TYPE="${INJECTION_TYPE:-kvcache}"
CACHE_PREWARM_COUNT="${CACHE_PREWARM_COUNT:-all}"

LINUCB_ALPHA="${LINUCB_ALPHA:-0.05}"
LINUCB_LAMBDA="${LINUCB_LAMBDA:-1.0}"
REWARD_SCALE_MS="${REWARD_SCALE_MS:-1000.0}"
REWARD_CLIP="${REWARD_CLIP:-5.0}"

INSTANCE_COUNT="${INSTANCE_COUNT:-4}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
INSTANCE_GPU_GROUPS="${INSTANCE_GPU_GROUPS:-}"
INSTANCE_TP_SIZES="${INSTANCE_TP_SIZES:-}"

case "$REPEATS:$REQUESTS:$WARMUP_REQUESTS" in
  *[!0-9:]*) echo "[FAIL] REPEATS, REQUESTS, and WARMUP_REQUESTS must be integers"; exit 2 ;;
esac
if [ "$REPEATS" -lt 1 ] || [ "$REQUESTS" -lt 1 ] || [ "$WARMUP_REQUESTS" -lt 1 ]; then
  echo "[FAIL] repeat/request counts must be positive"
  exit 2
fi
if [ "$INJECTION_TYPE" != "kvcache" ] && [ "$INJECTION_TYPE" != "text" ]; then
  echo "[FAIL] INJECTION_TYPE must be kvcache or text"
  exit 2
fi

stamp="$(date +%Y%m%d_%H%M%S)"
run_dir="$PROJECT_IN_CONTAINER/log/policy_matrix/run-$stamp"
host_run_dir="$PROJECT_HOST/log/policy_matrix/run-$stamp"
mkdir -p "$host_run_dir"

start_stack() {
  local strategy="$1"
  local prewarm_count="$2"
  local rl_enabled=0
  if [ "$strategy" = "linucb" ]; then
    rl_enabled=1
  fi

  PROJECT_HOST="$PROJECT_HOST" \
  MODEL_DIR="$MODEL_DIR" \
  MODEL_NAME="$MODEL_NAME" \
  CONTAINER="$CONTAINER" \
  REDIS_CONTAINER="$REDIS_CONTAINER" \
  PROJECT_IN_CONTAINER="$PROJECT_IN_CONTAINER" \
  PREWARM_COUNT="$prewarm_count" \
  INSTANCE_COUNT="$INSTANCE_COUNT" \
  TENSOR_PARALLEL_SIZE="$TENSOR_PARALLEL_SIZE" \
  INSTANCE_GPU_GROUPS="$INSTANCE_GPU_GROUPS" \
  INSTANCE_TP_SIZES="$INSTANCE_TP_SIZES" \
  PROXY_INSTANCE_STRATEGY="$strategy" \
  PROXY_RL_ENABLED="$rl_enabled" \
  PROXY_RL_ALPHA="$LINUCB_ALPHA" \
  PROXY_RL_LAMBDA="$LINUCB_LAMBDA" \
  PROXY_RL_WARMUP_REQUESTS="$WARMUP_REQUESTS" \
  PROXY_RL_REWARD_TTFT_SCALE_MS="$REWARD_SCALE_MS" \
  PROXY_RL_REWARD_CLIP="$REWARD_CLIP" \
  bash "$PROJECT_HOST/scripts/start_rl_4instance_docker.sh"
}

run_client() {
  local requests="$1"
  local concurrency="$2"
  local seed="$3"
  local output="$4"

  docker exec "$CONTAINER" python3 "$PROJECT_IN_CONTAINER/client/perf_client.py" \
    --mode concurrent \
    --base-url "$BASE_URL" \
    --workload-file "$WORKLOAD_FILE" \
    --requests "$requests" \
    --concurrency "$concurrency" \
    --allow-duplicate \
    --seed "$seed" \
    --model "$MODEL_NAME" \
    --stream true \
    --rag true \
    --injection-type "$INJECTION_TYPE" \
    --max-tokens "$MAX_TOKENS" \
    --temperature 0 \
    --output-jsonl "$output"
}

IFS=',' read -r -a strategies <<< "$STRATEGIES"
IFS=',' read -r -a concurrencies <<< "$CONCURRENCIES"

for strategy in "${strategies[@]}"; do
  case "$strategy" in
    round_robin|least_inflight|linucb) ;;
    *) echo "[FAIL] unsupported strategy: $strategy"; exit 2 ;;
  esac
done
for concurrency in "${concurrencies[@]}"; do
  case "$concurrency" in
    ''|*[!0-9]*) echo "[FAIL] invalid concurrency: $concurrency"; exit 2 ;;
  esac
  if [ "$concurrency" -lt 1 ]; then
    echo "[FAIL] concurrency must be positive: $concurrency"
    exit 2
  fi
done

cat > "$host_run_dir/config.txt" <<EOF
strategies=$STRATEGIES
concurrencies=$CONCURRENCIES
repeats=$REPEATS
requests=$REQUESTS
warmup_requests=$WARMUP_REQUESTS
cache_prewarm_count=$CACHE_PREWARM_COUNT
injection_type=$INJECTION_TYPE
alpha=$LINUCB_ALPHA
lambda=$LINUCB_LAMBDA
reward_scale_ms=$REWARD_SCALE_MS
reward_clip=$REWARD_CLIP
instance_count=$INSTANCE_COUNT
tp_default=$TENSOR_PARALLEL_SIZE
gpu_groups=${INSTANCE_GPU_GROUPS:-auto}
tp_sizes=${INSTANCE_TP_SIZES:-auto}
EOF

printf 'pair\trepeat\tconcurrency\tstrategy\twarmup_seed\tmeasure_seed\n' \
  > "$host_run_dir/run-order.tsv"

if [ "$CACHE_PREWARM_COUNT" != "0" ]; then
  echo "===== Populate shared Redis before measured restarts ====="
  start_stack "round_robin" "$CACHE_PREWARM_COUNT"
fi

measurements=()
pair=0
strategy_count="${#strategies[@]}"
for concurrency in "${concurrencies[@]}"; do
  for repeat in $(seq 1 "$REPEATS"); do
    pair=$((pair + 1))
    seed=$((10000 + pair))
    strategy_offset=$(((pair - 1) % strategy_count))
    for strategy_step in $(seq 0 $((strategy_count - 1))); do
      strategy_index=$(((strategy_offset + strategy_step) % strategy_count))
      strategy="${strategies[$strategy_index]}"
      prefix="${strategy}-c${concurrency}-r${repeat}"
      warmup_file="$run_dir/${prefix}-warmup.jsonl"
      measure_file="$run_dir/${prefix}.jsonl"
      summary_file="$run_dir/${prefix}-summary.json"

      printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$pair" "$repeat" "$concurrency" "$strategy" \
        "$seed" "$((seed + 100000))" >> "$host_run_dir/run-order.tsv"

      echo "===== $prefix: restart ====="
      start_stack "$strategy" 0
      echo "===== $prefix: warm-up ($WARMUP_REQUESTS requests) ====="
      run_client "$WARMUP_REQUESTS" "$concurrency" "$seed" "$warmup_file"
      echo "===== $prefix: measure ($REQUESTS requests) ====="
      run_client "$REQUESTS" "$concurrency" "$((seed + 100000))" "$measure_file"

      docker exec "$CONTAINER" python3 \
        "$PROJECT_IN_CONTAINER/scripts/analyze_experiment_jsonl.py" \
        "$measure_file" --output "$summary_file"
      measurements+=("$measure_file")
    done
  done
done

docker exec "$CONTAINER" python3 \
  "$PROJECT_IN_CONTAINER/scripts/analyze_experiment_jsonl.py" \
  "${measurements[@]}" --output "$run_dir/all-summaries.json"

echo "[DONE] $host_run_dir"
