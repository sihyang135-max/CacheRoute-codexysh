#!/usr/bin/env bash
# Issue #4 phase-1.0 signal smoke only. This does not produce performance claims.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${EXPECTED_COMMIT:?Set EXPECTED_COMMIT to the exact checked-out smoke commit}"
: "${MODEL_DIR:?Set MODEL_DIR to the model directory visible in the container}"
: "${MODEL_NAME:?Set MODEL_NAME to the served model name}"
: "${CALIBRATION_AGGREGATE:?Set CALIBRATION_AGGREGATE to the verified three-round aggregate JSON}"

PROJECT_HOST="$(readlink -f "${PROJECT_HOST:-$ROOT}")"
if [ "$PROJECT_HOST" != "$(readlink -f "$ROOT")" ]; then
  echo "[FAIL] PROJECT_HOST must be the checkout containing this script" >&2
  exit 2
fi

CONTAINER="${CONTAINER:-cr0720-rl}"
REDIS_CONTAINER="${REDIS_CONTAINER:-lmcache-redis}"
PROJECT_IN_CONTAINER="${PROJECT_IN_CONTAINER:-/workspace/llm-stack/CacheRoute}"
BASE_URL="${BASE_URL:-http://127.0.0.1:7001}"
WORKLOAD_FILE="${WORKLOAD_FILE:-$PROJECT_IN_CONTAINER/client/taskset/workload_nq.json}"
REQUESTS="${REQUESTS:-80}"
CONCURRENCY="${CONCURRENCY:-1}"
SEED="${SEED:-20260722}"
MAX_TOKENS="${MAX_TOKENS:-1}"
INJECTION_TYPE="${INJECTION_TYPE:-kvcache}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-30}"
ENGINE_WARMUP_REQUESTS_PER_INSTANCE="${ENGINE_WARMUP_REQUESTS_PER_INSTANCE:-2}"
LINUCB_ALPHA="${LINUCB_ALPHA:-0.4}"
LINUCB_LAMBDA="${LINUCB_LAMBDA:-1.0}"
REWARD_SCALE_MS="${REWARD_SCALE_MS:-1000.0}"
REWARD_CLIP="${REWARD_CLIP:-5.0}"
EXPECTED_CALIBRATION_COMMIT="${EXPECTED_CALIBRATION_COMMIT:-5ec94fede57d5153fcb5094fcdae28736b003a10}"

case "$REQUESTS:$CONCURRENCY:$SEED:$WARMUP_REQUESTS:$ENGINE_WARMUP_REQUESTS_PER_INSTANCE" in
  *[!0-9:]*) echo "[FAIL] request, concurrency, seed, and warmup values must be integers" >&2; exit 2 ;;
esac
if [ "$REQUESTS" -lt 50 ] || [ "$REQUESTS" -gt 100 ]; then
  echo "[FAIL] signal smoke requires 50-100 requests per policy" >&2
  exit 2
fi
if [ "$CONCURRENCY" -ne 1 ]; then
  echo "[FAIL] this phase-1.0 smoke is fixed at concurrency=1" >&2
  exit 2
fi
if [ "$REQUESTS" -le "$WARMUP_REQUESTS" ]; then
  echo "[FAIL] request count must exceed LinUCB warmup" >&2
  exit 2
fi
if [ "$INJECTION_TYPE" != "kvcache" ] && [ "$INJECTION_TYPE" != "text" ]; then
  echo "[FAIL] INJECTION_TYPE must be kvcache or text" >&2
  exit 2
fi

calibration_info="$({
  python3 - "$CALIBRATION_AGGREGATE" "$EXPECTED_CALIBRATION_COMMIT" <<'PY'
import json
import sys

path, expected_commit = sys.argv[1:]
data = json.load(open(path, encoding="utf-8"))
if data.get("status") != "passed":
    raise SystemExit("calibration aggregate status is not passed")
if data.get("git_commit") != expected_commit:
    raise SystemExit(
        f"calibration commit mismatch: expected={expected_commit} actual={data.get('git_commit')}"
    )
ratios = data.get("diagnostic_speed_factors_csv")
if not isinstance(ratios, str) or len(ratios.split(",")) != 4:
    raise SystemExit("calibration aggregate has no four-value diagnostic speed-factor CSV")
print(ratios)
PY
} 2>&1)" || {
  echo "[FAIL] invalid calibration aggregate: $calibration_info" >&2
  exit 2
}
INSTANCE_PREFILL_CAPACITY_RATIOS="$calibration_info"

actual_commit="$(git -C "$PROJECT_HOST" rev-parse HEAD)"
if [ "$actual_commit" != "$EXPECTED_COMMIT" ]; then
  echo "[FAIL] commit mismatch: expected=$EXPECTED_COMMIT actual=$actual_commit" >&2
  exit 2
fi
if [ -n "$(git -C "$PROJECT_HOST" status --porcelain)" ]; then
  echo "[FAIL] server worktree must be clean before the experiment" >&2
  git -C "$PROJECT_HOST" status --short >&2
  exit 2
fi
EXPECTED_COMMIT="$EXPECTED_COMMIT" REQUIRE_CLEAN_WORKTREE=1 \
  PROJECT_ROOT="$PROJECT_HOST" bash "$PROJECT_HOST/scripts/verify_source_sync.sh"

run_id="${RUN_ID:-signal-smoke-$(date -u +%Y%m%dT%H%M%SZ)}"
host_run_dir="$PROJECT_HOST/log/rl4-signal-smoke/$run_id"
container_run_dir="$PROJECT_IN_CONTAINER/log/rl4-signal-smoke/$run_id"
if [ -e "$host_run_dir" ]; then
  echo "[FAIL] result directory already exists: $host_run_dir" >&2
  exit 2
fi
mkdir -p "$host_run_dir"
cp "$CALIBRATION_AGGREGATE" "$host_run_dir/calibration-aggregate.json"

finalize() {
  local code="$1"
  trap - EXIT
  printf 'exit_code=%s\nfinished_at_utc=%s\n' "$code" "$(date -u +%FT%TZ)" \
    > "$host_run_dir/run-status.txt"
  (
    cd "$host_run_dir"
    find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum
  ) > "$host_run_dir/SHA256SUMS"
  local archive="$host_run_dir.tar.gz"
  tar -C "$(dirname "$host_run_dir")" -czf "$archive" "$(basename "$host_run_dir")"
  sha256sum "$archive" > "$archive.sha256"
  echo "[RESULT] directory=$host_run_dir"
  echo "[RESULT] archive=$archive"
  exit "$code"
}
trap 'finalize $?' EXIT

cat > "$host_run_dir/config.txt" <<EOF
experiment_kind=rl4_rr_linucb_signal_smoke
claim_boundary=signal validation only; no performance conclusion
git_commit=$EXPECTED_COMMIT
calibration_commit=$EXPECTED_CALIBRATION_COMMIT
calibration_input_summary_sha256=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["input_summary_sha256"])' "$CALIBRATION_AGGREGATE")
model_name=$MODEL_NAME
model_dir=$MODEL_DIR
container=$CONTAINER
redis_container=$REDIS_CONTAINER
workload_file=$WORKLOAD_FILE
strategies=round_robin,linucb
requests_per_strategy=$REQUESTS
concurrency=$CONCURRENCY
seed=$SEED
max_tokens=$MAX_TOKENS
injection_type=$INJECTION_TYPE
engine_warmup_requests_per_instance=$ENGINE_WARMUP_REQUESTS_PER_INSTANCE
linucb_warmup_requests=$WARMUP_REQUESTS
linucb_alpha=$LINUCB_ALPHA
linucb_lambda=$LINUCB_LAMBDA
reward_scale_ms=$REWARD_SCALE_MS
reward_clip=$REWARD_CLIP
instance_gpu_groups=0,1,2,3;4,5;6;7
instance_tp_sizes=4,2,1,1
instance_prefill_capacity_ratios=$INSTANCE_PREFILL_CAPACITY_RATIOS
kv_residency_scope=global
kv_link_scope=global
EOF

{
  echo "captured_at_utc=$(date -u +%FT%TZ)"
  echo "git_commit=$actual_commit"
  echo "git_status=clean"
  echo "host_uname=$(uname -a)"
  docker version --format 'docker_client={{.Client.Version}} docker_server={{.Server.Version}}'
  nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total \
    --format=csv,noheader
} > "$host_run_dir/environment.txt"

start_stack() {
  local strategy="$1"
  local rl_enabled=0
  if [ "$strategy" = "linucb" ]; then
    rl_enabled=1
  fi
  EXPECTED_COMMIT="$EXPECTED_COMMIT" \
  PROJECT_HOST="$PROJECT_HOST" MODEL_DIR="$MODEL_DIR" MODEL_NAME="$MODEL_NAME" \
  CONTAINER="$CONTAINER" REDIS_CONTAINER="$REDIS_CONTAINER" \
  PROJECT_IN_CONTAINER="$PROJECT_IN_CONTAINER" PREWARM_COUNT=0 \
  INSTANCE_PREFILL_CAPACITY_RATIOS="$INSTANCE_PREFILL_CAPACITY_RATIOS" \
  PROXY_INSTANCE_STRATEGY="$strategy" PROXY_RL_ENABLED="$rl_enabled" \
  PROXY_RL_ALPHA="$LINUCB_ALPHA" PROXY_RL_LAMBDA="$LINUCB_LAMBDA" \
  PROXY_RL_WARMUP_REQUESTS="$WARMUP_REQUESTS" \
  PROXY_RL_REWARD_TTFT_SCALE_MS="$REWARD_SCALE_MS" \
  PROXY_RL_REWARD_CLIP="$REWARD_CLIP" \
  PROXY_RL_KV_RESIDENCY_SCOPE=global PROXY_RL_KV_LINK_SCOPE=global \
    bash "$PROJECT_HOST/scripts/start_rl4_signal_smoke.sh"
}

warmup_engines() {
  local strategy="$1"
  docker exec -e PYTHONPATH="$PROJECT_IN_CONTAINER" -w "$PROJECT_IN_CONTAINER" \
    "$CONTAINER" python3 "$PROJECT_IN_CONTAINER/scripts/warmup_vllm_instances.py" \
    --model "$MODEL_NAME" --instance-count 4 \
    --requests-per-instance "$ENGINE_WARMUP_REQUESTS_PER_INSTANCE" \
    --output "$container_run_dir/${strategy}-engine-warmup.json"
}

run_client() {
  local strategy="$1"
  docker exec -e PYTHONPATH="$PROJECT_IN_CONTAINER" -w "$PROJECT_IN_CONTAINER" \
    "$CONTAINER" python3 "$PROJECT_IN_CONTAINER/client/perf_client.py" \
    --mode concurrent --base-url "$BASE_URL" --workload-file "$WORKLOAD_FILE" \
    --requests "$REQUESTS" --concurrency "$CONCURRENCY" --allow-duplicate \
    --seed "$SEED" --model "$MODEL_NAME" --stream true --rag true \
    --injection-type "$INJECTION_TYPE" --max-tokens "$MAX_TOKENS" \
    --temperature 0 --output-jsonl "$container_run_dir/${strategy}-raw.jsonl"
}

capture_runtime_fingerprint() {
  docker inspect -f 'container_image={{.Config.Image}} image_id={{.Image}}' "$CONTAINER"
  docker inspect -f 'redis_image={{.Config.Image}} redis_image_id={{.Image}}' "$REDIS_CONTAINER"
  docker exec -e MODEL_DIR="$MODEL_DIR" "$CONTAINER" bash -lc '
    python3 - <<"PY"
import importlib.metadata
for name in ("torch", "vllm", "lmcache"):
    try:
        print(f"package_{name}={importlib.metadata.version(name)}")
    except importlib.metadata.PackageNotFoundError:
        print(f"package_{name}=missing")
PY
    find "$MODEL_DIR" -maxdepth 1 -type f -printf "%f\t%s\n" | sort
    find "$MODEL_DIR" -maxdepth 1 -type f \( -name "*.json" -o -name "tokenizer.model" \) -print0 \
      | sort -z | xargs -0 -r sha256sum
  '
}

printf 'order\tstrategy\tseed\trequests\tconcurrency\n1\tround_robin\t%s\t%s\t%s\n2\tlinucb\t%s\t%s\t%s\n' \
  "$SEED" "$REQUESTS" "$CONCURRENCY" "$SEED" "$REQUESTS" "$CONCURRENCY" \
  > "$host_run_dir/run-order.tsv"

for strategy in round_robin linucb; do
  echo "===== $strategy: clean restart ====="
  start_stack "$strategy"
  cp "$PROJECT_HOST/log/rl4/status.txt" "$host_run_dir/${strategy}-startup-status.txt"
  cp "$PROJECT_HOST/log/rl4/instances.json" "$host_run_dir/${strategy}-instances.json"
  if [ "$strategy" = "round_robin" ]; then
    capture_runtime_fingerprint >> "$host_run_dir/environment.txt"
  fi
  echo "===== $strategy: direct engine warmup ====="
  warmup_engines "$strategy"
  echo "===== $strategy: $REQUESTS-request signal smoke ====="
  run_client "$strategy"
  docker exec "$CONTAINER" bash \
    "$PROJECT_IN_CONTAINER/scripts/check_rl_4instance_health.sh" \
    > "$host_run_dir/${strategy}-post-health.txt"
  python3 "$PROJECT_HOST/scripts/analyze_experiment_jsonl.py" \
    "$host_run_dir/${strategy}-raw.jsonl" \
    --output "$host_run_dir/${strategy}-summary.json"
done

python3 "$PROJECT_HOST/scripts/validate_rl4_signal_smoke.py" \
  --rr-jsonl "$host_run_dir/round_robin-raw.jsonl" \
  --linucb-jsonl "$host_run_dir/linucb-raw.jsonl" \
  --expected-speed-factors "$INSTANCE_PREFILL_CAPACITY_RATIOS" \
  --warmup-requests "$WARMUP_REQUESTS" \
  --expected-commit "$EXPECTED_COMMIT" \
  --config "$host_run_dir/config.txt" \
  --environment "$host_run_dir/environment.txt" \
  --calibration "$host_run_dir/calibration-aggregate.json" \
  --output "$host_run_dir/validation.json"

echo "[DONE] signal smoke acceptance passed"
