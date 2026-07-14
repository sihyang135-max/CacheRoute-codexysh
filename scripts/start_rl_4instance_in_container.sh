#!/usr/bin/env bash
# Called by start_rl_4instance_docker.sh inside the dedicated CacheRoute container.
set -euo pipefail

: "${PROJECT:?PROJECT is required}"
: "${MODEL_DIR:?MODEL_DIR is required}"
: "${MODEL_NAME:?MODEL_NAME is required}"

export PYTHONPATH="$PROJECT"
export PYTHONHASHSEED=0
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export LMCACHE_CONFIG_FILE="${LMCACHE_CONFIG_FILE:-/workspace/llm-stack/config/lmcache_with_redis.yaml}"
export SCHEDULER_MODEL_PATH="$MODEL_DIR"
export SCHEDULER_MODEL_NAME="$MODEL_NAME"
export SCHEDULER_TOKENIZER_PATH="$MODEL_DIR"

LOG_DIR="$PROJECT/log/rl4"
mkdir -p "$LOG_DIR"

stop_old() {
  pkill -f 'vllm.entrypoints.openai.api_server' || true
  pkill -f 'demo_instance.py' || true
  pkill -f 'demo_proxy.py' || true
  pkill -f 'demo_kdn.py' || true
  pkill -f 'demo_scheduler.py' || true
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

# Four TP=2 Instances use all eight 5090 GPUs. MODEL_DIR must contain a model
# that fits on two GPUs (or be launched with the appropriate quantization flags).
for idx in 0 1 2 3; do
  gpu0=$((idx * 2)); gpu1=$((gpu0 + 1)); vllm_port=$((18000 + idx))
  CUDA_VISIBLE_DEVICES="$gpu0,$gpu1" start_bg "$LOG_DIR/vllm-${idx}.log" \
    python3 -m vllm.entrypoints.openai.api_server \
      --model "$MODEL_DIR" --served-model-name "$MODEL_NAME" \
      --host 127.0.0.1 --port "$vllm_port" \
      --tensor-parallel-size 2 --gpu-memory-utilization 0.82 \
      --max-model-len 4096 --max-num-seqs 8 --max-num-batched-tokens 8192 \
      --kv-offloading-backend lmcache --kv-offloading-size 32 \
      --disable-hybrid-kv-cache-manager --kv-cache-metrics
done

for idx in 0 1 2 3; do
  wait_http "http://127.0.0.1:$((18000 + idx))/v1/models" "vLLM-${idx}" 300
  if ! curl -s "http://127.0.0.1:$((18000 + idx))/metrics" | grep -Eq 'gpu.*cache.*usage|gpu_cache_usage'; then
    echo "[WARN] vLLM-${idx}: KVCache Prometheus metric was not found" | tee -a "$LOG_DIR/status.txt"
  fi
done

cd "$PROJECT/test"
start_bg "$LOG_DIR/scheduler.log" python3 "$PROJECT/scripts/demo_scheduler_rl.py"
wait_http http://127.0.0.1:7001/debug/status Scheduler 120

start_bg "$LOG_DIR/kdn.log" python3 demo_kdn.py
wait_http http://127.0.0.1:9101/v1/topology/ping KDN 120

export PROXY_INSTANCE_STRATEGY=linucb
export PROXY_RL_ENABLED=1
export PROXY_RL_ALPHA=0.4
export PROXY_RL_LAMBDA=1.0
export PROXY_RL_WARMUP_REQUESTS=30
export PROXY_RL_PROMETHEUS_INTERVAL_S=1.0
export PROXY_RL_PROMETHEUS_TIMEOUT_S=0.2
export PROXY_RL_PROMETHEUS_STALE_S=3.0
export PROXY_RL_KV_USAGE_LIMIT=0.90
start_bg "$LOG_DIR/proxy.log" python3 demo_proxy.py --strategy linucb --injection-strategy default
wait_http http://127.0.0.1:8002/healthz Proxy 120

for idx in 0 1 2 3; do
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
grep -o 'inst-[0-3]' "$LOG_DIR/instances.json" | sort -u | wc -l | grep -qx 4 || {
  echo "[FAIL] Four Instances did not register; inspect $LOG_DIR/instance-*.log" | tee -a "$LOG_DIR/status.txt"
  exit 1
}
echo "[OK] Four Instances registered" | tee -a "$LOG_DIR/status.txt"

# Optional, deliberately explicit: KV building may take a long time.
if [ "${PREWARM_COUNT:-0}" -gt 0 ]; then
  cd "$PROJECT/kdn_server/util"
  python3 batch_register_kdn.py \
    --manifest knowledge_manifest_nq.json --count "$PREWARM_COUNT" \
    --base-url http://127.0.0.1:9101 \
    --api-url http://127.0.0.1:18000/v1/chat/completions \
    --model "$MODEL_NAME" --redis-host 127.0.0.1 \
    --result-json "$LOG_DIR/kdn_prewarm.json"
fi

echo "[DONE] LinUCB 4-Instance environment is ready" | tee -a "$LOG_DIR/status.txt"
