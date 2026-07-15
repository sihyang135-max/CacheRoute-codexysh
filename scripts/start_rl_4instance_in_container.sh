#!/usr/bin/env bash
# Called by start_rl_4instance_docker.sh inside the dedicated CacheRoute container.
set -euo pipefail

: "${PROJECT:?PROJECT is required}"
: "${MODEL_DIR:?MODEL_DIR is required}"
: "${MODEL_NAME:?MODEL_NAME is required}"

INSTANCE_COUNT="${INSTANCE_COUNT:-4}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
INSTANCE_GPU_GROUPS="${INSTANCE_GPU_GROUPS:-}"
INSTANCE_TP_SIZES="${INSTANCE_TP_SIZES:-}"
PROXY_INSTANCE_STRATEGY="${PROXY_INSTANCE_STRATEGY:-linucb}"
PROXY_RL_ENABLED="${PROXY_RL_ENABLED:-1}"
PROXY_RL_ALPHA="${PROXY_RL_ALPHA:-0.4}"
PROXY_RL_LAMBDA="${PROXY_RL_LAMBDA:-1.0}"
PROXY_RL_WARMUP_REQUESTS="${PROXY_RL_WARMUP_REQUESTS:-30}"
PROXY_RL_REWARD_TTFT_SCALE_MS="${PROXY_RL_REWARD_TTFT_SCALE_MS:-1000.0}"
PROXY_RL_REWARD_CLIP="${PROXY_RL_REWARD_CLIP:-5.0}"

case "$INSTANCE_COUNT" in
  ''|*[!0-9]*) echo "[FAIL] INSTANCE_COUNT must be a positive integer"; exit 2 ;;
esac
case "$TENSOR_PARALLEL_SIZE" in
  ''|*[!0-9]*) echo "[FAIL] TENSOR_PARALLEL_SIZE must be a positive integer"; exit 2 ;;
esac
if [ "$INSTANCE_COUNT" -lt 1 ] || [ "$TENSOR_PARALLEL_SIZE" -lt 1 ]; then
  echo "[FAIL] INSTANCE_COUNT and TENSOR_PARALLEL_SIZE must be >= 1"
  exit 2
fi
case "${PREWARM_COUNT:-0}" in
  all|0) ;;
  ''|*[!0-9]*) echo "[FAIL] PREWARM_COUNT must be 0, 'all', or a positive integer"; exit 2 ;;
esac

declare -a gpu_groups=()
declare -a tp_sizes=()
declare -A used_gpus=()
if [ -n "$INSTANCE_GPU_GROUPS" ] || [ -n "$INSTANCE_TP_SIZES" ]; then
  if [ -z "$INSTANCE_GPU_GROUPS" ] || [ -z "$INSTANCE_TP_SIZES" ]; then
    echo "[FAIL] INSTANCE_GPU_GROUPS and INSTANCE_TP_SIZES must be set together"
    exit 2
  fi
  IFS=';' read -r -a gpu_groups <<< "$INSTANCE_GPU_GROUPS"
  IFS=',' read -r -a tp_sizes <<< "$INSTANCE_TP_SIZES"
  if [ "${#gpu_groups[@]}" -ne "$INSTANCE_COUNT" ] || [ "${#tp_sizes[@]}" -ne "$INSTANCE_COUNT" ]; then
    echo "[FAIL] custom GPU/TP topology must contain INSTANCE_COUNT entries"
    exit 2
  fi
  for idx in $(seq 0 $((INSTANCE_COUNT - 1))); do
    case "${tp_sizes[$idx]}" in
      ''|*[!0-9]*) echo "[FAIL] invalid TP size: ${tp_sizes[$idx]}"; exit 2 ;;
    esac
    if [ "${tp_sizes[$idx]}" -lt 1 ] || [ -z "${gpu_groups[$idx]}" ]; then
      echo "[FAIL] GPU group and TP size must be non-empty and positive"
      exit 2
    fi
    IFS=',' read -r -a group_gpu_ids <<< "${gpu_groups[$idx]}"
    if [ "${#group_gpu_ids[@]}" -ne "${tp_sizes[$idx]}" ]; then
      echo "[FAIL] instance $idx has ${#group_gpu_ids[@]} GPUs but TP=${tp_sizes[$idx]}"
      exit 2
    fi
    for gpu_id in "${group_gpu_ids[@]}"; do
      case "$gpu_id" in
        ''|*[!0-9]*) echo "[FAIL] invalid GPU id: $gpu_id"; exit 2 ;;
      esac
      if [ -n "${used_gpus[$gpu_id]:-}" ]; then
        echo "[FAIL] GPU $gpu_id is assigned to more than one instance"
        exit 2
      fi
      used_gpus[$gpu_id]=1
    done
  done
fi

export PYTHONPATH="$PROJECT"
export PYTHONHASHSEED=0
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export LMCACHE_CONFIG_FILE="${LMCACHE_CONFIG_FILE:-/workspace/llm-stack/config/lmcache_with_redis.yaml}"
export SCHEDULER_MODEL_PATH="$MODEL_DIR"
export SCHEDULER_MODEL_NAME="$MODEL_NAME"
export SCHEDULER_TOKENIZER_PATH="$MODEL_DIR"
export SCHEDULER_EMBEDDING_MODEL="${SCHEDULER_EMBEDDING_MODEL:-/workspace/llm-stack/models/intfloat/multilingual-e5-large-instruct}"

LOG_DIR="$PROJECT/log/rl4"
mkdir -p "$LOG_DIR"

stop_old() {
  pkill -f 'vllm.entrypoints.openai.api_server' || true
  pkill -f 'demo_instance.py' || true
  pkill -f 'demo_proxy.py' || true
  pkill -f 'demo_kdn.py' || true
  pkill -f 'demo_scheduler.py' || true
  pkill -f 'demo_scheduler_rl.py' || true
  sleep 2
}

wait_http() {
  local url="$1"; local name="$2"; local limit="${3:-120}"
  for _ in $(seq 1 "$limit"); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      echo "[OK] $name" | tee -a "$LOG_DIR/status.txt"
      return 0
    fi
    sleep 1
  done
  echo "[FAIL] $name: $url" | tee -a "$LOG_DIR/status.txt"
  return 1
}

start_bg() {
  local log="$1"; shift
  nohup "$@" > "$log" 2>&1 &
}

stop_old
: > "$LOG_DIR/status.txt"

# Default uses equal TP sizes. For TP4/TP2/TP1/TP1, set
# INSTANCE_GPU_GROUPS='0,1,2,3;4,5;6;7' and INSTANCE_TP_SIZES='4,2,1,1'.
for idx in $(seq 0 $((INSTANCE_COUNT - 1))); do
  if [ "${#gpu_groups[@]}" -gt 0 ]; then
    gpu_ids="${gpu_groups[$idx]}"
    tp_size="${tp_sizes[$idx]}"
  else
    gpu_start=$((idx * TENSOR_PARALLEL_SIZE))
    gpu_end=$((gpu_start + TENSOR_PARALLEL_SIZE - 1))
    gpu_ids="$(seq -s, "$gpu_start" "$gpu_end")"
    tp_size="$TENSOR_PARALLEL_SIZE"
  fi
  vllm_port=$((18000 + idx))
  CUDA_VISIBLE_DEVICES="$gpu_ids" start_bg "$LOG_DIR/vllm-${idx}.log" \
    python3 -m vllm.entrypoints.openai.api_server \
      --model "$MODEL_DIR" --served-model-name "$MODEL_NAME" \
      --host 127.0.0.1 --port "$vllm_port" \
      --tensor-parallel-size "$tp_size" --gpu-memory-utilization 0.82 \
      --max-model-len 4096 --max-num-seqs 8 --max-num-batched-tokens 8192 \
      --kv-offloading-backend lmcache --kv-offloading-size 32 \
      --disable-hybrid-kv-cache-manager --kv-cache-metrics
done

for idx in $(seq 0 $((INSTANCE_COUNT - 1))); do
  wait_http "http://127.0.0.1:$((18000 + idx))/v1/models" "vLLM-${idx}" 300
  if ! curl -s "http://127.0.0.1:$((18000 + idx))/metrics" | grep -Eq 'kv_cache_usage|gpu_cache_usage'; then
    echo "[WARN] vLLM-${idx}: KVCache Prometheus metric was not found" | tee -a "$LOG_DIR/status.txt"
  fi
done

cd "$PROJECT/test"
start_bg "$LOG_DIR/scheduler.log" python3 "$PROJECT/scripts/demo_scheduler_rl.py"
wait_http http://127.0.0.1:7001/debug/status Scheduler 120

start_bg "$LOG_DIR/kdn.log" python3 demo_kdn.py
wait_http http://127.0.0.1:9101/v1/topology/ping KDN 120

export PROXY_INSTANCE_STRATEGY
export PROXY_RL_ENABLED
export PROXY_RL_ALPHA
export PROXY_RL_LAMBDA
export PROXY_RL_WARMUP_REQUESTS
export PROXY_RL_REWARD_TTFT_SCALE_MS
export PROXY_RL_REWARD_CLIP
export PROXY_RL_PROMETHEUS_INTERVAL_S=1.0
export PROXY_RL_PROMETHEUS_TIMEOUT_S=0.2
export PROXY_RL_PROMETHEUS_STALE_S=3.0
export PROXY_RL_KV_USAGE_LIMIT=0.90
start_bg "$LOG_DIR/proxy.log" python3 demo_proxy.py --strategy "$PROXY_INSTANCE_STRATEGY" --injection-strategy default
wait_http http://127.0.0.1:8002/healthz Proxy 120

for idx in $(seq 0 $((INSTANCE_COUNT - 1))); do
  vllm_port=$((18000 + idx)); instance_port=$((19001 + idx)); cp_port=$((19101 + idx))
  INSTANCE_ID="inst-${idx}" INSTANCE_CP_PORT="$cp_port" \
  PROXY_CP_URL=http://127.0.0.1:8002 \
  VLLM_BASE_URL="http://127.0.0.1:${vllm_port}" \
  VLLM_METRICS_URL="http://127.0.0.1:${vllm_port}/metrics" \
  INSTANCE_TOPOLOGY_KDN_TARGETS=127.0.0.1:9101 \
  start_bg "$LOG_DIR/instance-${idx}.log" python3 "$PROJECT/test/demo_instance.py" \
    --host 127.0.0.1 --port "$instance_port" --kdn-targets 127.0.0.1:9101
done

sleep 5
curl -fsS http://127.0.0.1:8002/v1/instance/list > "$LOG_DIR/instances.json"
registered_count="$(grep -o 'inst-[0-9]\+' "$LOG_DIR/instances.json" | sort -u | wc -l)"
if [ "$registered_count" -ne "$INSTANCE_COUNT" ]; then
  echo "[FAIL] Expected $INSTANCE_COUNT Instances, found $registered_count; inspect $LOG_DIR/instance-*.log" | tee -a "$LOG_DIR/status.txt"
  exit 1
fi
echo "[OK] $INSTANCE_COUNT Instances registered" | tee -a "$LOG_DIR/status.txt"
echo "[CONFIG] strategy=$PROXY_INSTANCE_STRATEGY rl_enabled=$PROXY_RL_ENABLED alpha=$PROXY_RL_ALPHA lambda=$PROXY_RL_LAMBDA warmup=$PROXY_RL_WARMUP_REQUESTS reward_scale_ms=$PROXY_RL_REWARD_TTFT_SCALE_MS reward_clip=$PROXY_RL_REWARD_CLIP tp_default=$TENSOR_PARALLEL_SIZE gpu_groups=${INSTANCE_GPU_GROUPS:-auto} tp_sizes=${INSTANCE_TP_SIZES:-auto}" | tee -a "$LOG_DIR/status.txt"

# Optional, deliberately explicit: KV building may take a long time.
if [ "${PREWARM_COUNT:-0}" != "0" ]; then
  cd "$PROJECT/kdn_server/util"
  python3 batch_register_kdn.py \
    --manifest knowledge_manifest_nq.json --count "$PREWARM_COUNT" \
    --base-url http://127.0.0.1:9101 \
    --api-url http://127.0.0.1:18000/v1/chat/completions \
    --model "$MODEL_NAME" --redis-host 127.0.0.1 \
    --result-json "$LOG_DIR/kdn_prewarm.json"
fi

echo "[DONE] $PROXY_INSTANCE_STRATEGY $INSTANCE_COUNT-Instance environment is ready" | tee -a "$LOG_DIR/status.txt"
