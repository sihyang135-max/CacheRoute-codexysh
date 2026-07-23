#!/usr/bin/env bash
# Train a commit-compatible LinUCB model, then run paired RR vs frozen-LinUCB.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${EXPECTED_COMMIT:?Set the exact 40-character experiment commit}"
: "${MODEL_DIR:?Set the model directory visible in the container}"
: "${MODEL_NAME:?Set the served model name}"
: "${CALIBRATION_AGGREGATE:?Set the verified calibration aggregate JSON}"

PROJECT_HOST="$(readlink -f "${PROJECT_HOST:-$ROOT}")"
PROJECT_IN_CONTAINER="${PROJECT_IN_CONTAINER:-/workspace/llm-stack/CacheRoute}"
CONTAINER="${CONTAINER:-cr0720-rl}"
REDIS_CONTAINER="${REDIS_CONTAINER:-lmcache-redis}"
BASE_URL="${BASE_URL:-http://127.0.0.1:7001}"
WORKLOAD_FILE="${WORKLOAD_FILE:-$PROJECT_IN_CONTAINER/client/taskset/workload_nq.json}"
TRAINING_REQUESTS="${TRAINING_REQUESTS:-240}"
MEASURE_REQUESTS="${MEASURE_REQUESTS:-200}"
REPEATS="${REPEATS:-5}"
CONCURRENCY="${CONCURRENCY:-1}"
SEED="${SEED:-20260723}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-30}"
SNAPSHOT_INTERVAL="${SNAPSHOT_INTERVAL:-20}"
LINUCB_ALPHA="${LINUCB_ALPHA:-0.4}"
LINUCB_LAMBDA="${LINUCB_LAMBDA:-1.0}"
REWARD_SCALE_MS="${REWARD_SCALE_MS:-1000.0}"
REWARD_CLIP="${REWARD_CLIP:-5.0}"
ENGINE_WARMUP_REQUESTS_PER_INSTANCE="${ENGINE_WARMUP_REQUESTS_PER_INSTANCE:-2}"
CACHE_PREWARM_COUNT="${CACHE_PREWARM_COUNT:-all}"
MAX_TOKENS="${MAX_TOKENS:-1}"

fail() {
  echo "[FAIL] $*" >&2
  exit 2
}

case "$EXPECTED_COMMIT" in
  [0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]) ;;
  *) fail "EXPECTED_COMMIT must be a lowercase 40-character SHA" ;;
esac
case "$TRAINING_REQUESTS:$MEASURE_REQUESTS:$REPEATS:$CONCURRENCY:$ENGINE_WARMUP_REQUESTS_PER_INSTANCE" in
  *[!0-9:]*) fail "request, repeat, concurrency, and warm-up counts must be integers" ;;
esac
[ "$TRAINING_REQUESTS" -eq 240 ] || fail "TRAINING_REQUESTS is fixed at 240"
[ "$MEASURE_REQUESTS" -ge 200 ] || fail "MEASURE_REQUESTS must be at least 200"
[ "$REPEATS" -ge 5 ] || fail "REPEATS must be at least 5"
[ "$CONCURRENCY" -eq 1 ] || fail "CONCURRENCY is fixed at 1 for the first paired experiment"
[ "$PROJECT_HOST" = "$(readlink -f "$ROOT")" ] || fail "PROJECT_HOST must be this checkout"

actual_commit="$(git -C "$PROJECT_HOST" rev-parse HEAD)"
[ "$actual_commit" = "$EXPECTED_COMMIT" ] || fail "commit mismatch expected=$EXPECTED_COMMIT actual=$actual_commit"
[ -z "$(git -C "$PROJECT_HOST" status --porcelain)" ] || fail "worktree must be clean"
EXPECTED_COMMIT="$EXPECTED_COMMIT" REQUIRE_CLEAN_WORKTREE=1 PROJECT_ROOT="$PROJECT_HOST" \
  bash "$PROJECT_HOST/scripts/verify_source_sync.sh"

speed_factors="$(python3 - "$CALIBRATION_AGGREGATE" <<'PY'
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
if data.get("status") != "passed":
    raise SystemExit("calibration status is not passed")
value = data.get("diagnostic_speed_factors_csv")
if not isinstance(value, str) or len(value.split(",")) != 4:
    raise SystemExit("missing four speed factors")
print(value)
PY
)"

run_id="${RUN_ID:-rr-frozen-paired-$(date -u +%Y%m%dT%H%M%SZ)}"
host_run_dir="$PROJECT_HOST/log/rr-frozen-linucb/$run_id"
container_run_dir="$PROJECT_IN_CONTAINER/log/rr-frozen-linucb/$run_id"
[ ! -e "$host_run_dir" ] || fail "result directory exists: $host_run_dir"
mkdir -p "$host_run_dir"/{model-before,model-after,service-logs}
cp "$CALIBRATION_AGGREGATE" "$host_run_dir/calibration-aggregate.json"

container_model="$container_run_dir/linucb-model.json"
container_snapshots="$container_run_dir/training-parameter-snapshots.jsonl"
host_model="$host_run_dir/linucb-model.json"
host_snapshots="$host_run_dir/training-parameter-snapshots.jsonl"

finalize() {
  code="$1"
  trap - EXIT
  printf '{"exit_code":%s,"finished_at_utc":"%s"}\n' "$code" "$(date -u +%FT%TZ)" > "$host_run_dir/run-status.json"
  (cd "$host_run_dir" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum) > "$host_run_dir/SHA256SUMS"
  archive="$host_run_dir.tar.gz"
  tar -C "$(dirname "$host_run_dir")" -czf "$archive" "$(basename "$host_run_dir")"
  sha256sum "$archive" > "$archive.sha256"
  echo "[RESULT] directory=$host_run_dir"
  echo "[RESULT] archive=$archive"
  exit "$code"
}
trap 'finalize $?' EXIT

python3 - "$host_run_dir/config.json" <<PY
import json
json.dump({
  "experiment_kind": "rr_vs_frozen_linucb_paired",
  "claim_boundary": "paired concurrency-1 TTFT comparison under global KV scope; no KV-locality claim",
  "git_commit": "$EXPECTED_COMMIT",
  "training_requests": $TRAINING_REQUESTS,
  "measure_requests": $MEASURE_REQUESTS,
  "repeats": $REPEATS,
  "concurrency": $CONCURRENCY,
  "seed": $SEED,
  "warmup_requests": $WARMUP_REQUESTS,
  "engine_warmup_requests_per_instance": $ENGINE_WARMUP_REQUESTS_PER_INSTANCE,
  "cache_prewarm_count": "$CACHE_PREWARM_COUNT",
  "linucb_alpha": $LINUCB_ALPHA,
  "linucb_lambda": $LINUCB_LAMBDA,
  "reward_scale_ms": $REWARD_SCALE_MS,
  "reward_clip": $REWARD_CLIP,
  "speed_factors": "$speed_factors",
  "kv_residency_scope": "global",
  "kv_link_scope": "global",
  "workload_file": "$WORKLOAD_FILE",
  "model_name": "$MODEL_NAME",
  "max_tokens": $MAX_TOKENS
}, open("$host_run_dir/config.json", "w", encoding="utf-8"), indent=2)
PY
python3 - "$host_run_dir/environment.json" <<'PY'
import json, platform, subprocess, sys
def run(command):
    result = subprocess.run(command, text=True, capture_output=True)
    return {"exit_code": result.returncode, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
json.dump({
    "captured_at_utc": run(["date", "-u", "+%FT%TZ"])["stdout"],
    "platform": platform.platform(),
    "python": platform.python_version(),
    "docker": run(["docker", "version"]),
    "nvidia_smi": run(["nvidia-smi"]),
}, open(sys.argv[1], "w", encoding="utf-8"), indent=2)
PY

start_stack() {
  mode="$1"
  strategy="$2"
  prewarm_count="$3"
  rl_enabled=0
  load_path=""
  save_path=""
  frozen=0
  snapshot_path=""
  snapshot_interval=0
  if [ "$mode" = "fresh-training" ]; then
    rl_enabled=1
    save_path="$container_model"
    snapshot_path="$container_snapshots"
    snapshot_interval="$SNAPSHOT_INTERVAL"
  elif [ "$mode" = "loaded-frozen" ]; then
    rl_enabled=1
    load_path="$container_model"
    save_path="$container_model"
    frozen=1
  fi

  EXPECTED_COMMIT="$EXPECTED_COMMIT" PROJECT_HOST="$PROJECT_HOST" PROJECT_IN_CONTAINER="$PROJECT_IN_CONTAINER" \
  MODEL_DIR="$MODEL_DIR" MODEL_NAME="$MODEL_NAME" CONTAINER="$CONTAINER" REDIS_CONTAINER="$REDIS_CONTAINER" \
  PREWARM_COUNT="$prewarm_count" INSTANCE_PREFILL_CAPACITY_RATIOS="$speed_factors" \
  PROXY_INSTANCE_STRATEGY="$strategy" PROXY_RL_ENABLED="$rl_enabled" \
  PROXY_RL_ALPHA="$LINUCB_ALPHA" PROXY_RL_LAMBDA="$LINUCB_LAMBDA" \
  PROXY_RL_WARMUP_REQUESTS="$WARMUP_REQUESTS" \
  PROXY_RL_REWARD_TTFT_SCALE_MS="$REWARD_SCALE_MS" PROXY_RL_REWARD_CLIP="$REWARD_CLIP" \
  PROXY_RL_MODEL_LOAD_PATH="$load_path" PROXY_RL_MODEL_SAVE_PATH="$save_path" \
  PROXY_RL_SOURCE_COMMIT="$EXPECTED_COMMIT" PROXY_RL_FROZEN="$frozen" \
  PROXY_RL_PARAMETER_SNAPSHOT_PATH="$snapshot_path" \
  PROXY_RL_PARAMETER_SNAPSHOT_INTERVAL="$snapshot_interval" \
  KDN_TEXT_DB_DIR="$container_run_dir/kdn-text-db" \
  KDN_KV_DB_DIR="$container_run_dir/kdn-kv-db" \
  PROXY_RL_KV_RESIDENCY_SCOPE=global PROXY_RL_KV_LINK_SCOPE=global \
  bash "$PROJECT_HOST/scripts/start_rl4_signal_smoke.sh"
}

warmup_engines() {
  output="$1"
  [ "$ENGINE_WARMUP_REQUESTS_PER_INSTANCE" -ne 0 ] || return 0
  docker exec -e PYTHONPATH="$PROJECT_IN_CONTAINER" -w "$PROJECT_IN_CONTAINER" "$CONTAINER" python3 \
    "$PROJECT_IN_CONTAINER/scripts/warmup_vllm_instances.py" --model "$MODEL_NAME" --instance-count 4 \
    --requests-per-instance "$ENGINE_WARMUP_REQUESTS_PER_INSTANCE" --output "$output"
}

run_client() {
  count="$1"
  seed="$2"
  output="$3"
  docker exec -e PYTHONPATH="$PROJECT_IN_CONTAINER" -w "$PROJECT_IN_CONTAINER" "$CONTAINER" python3 \
    "$PROJECT_IN_CONTAINER/client/perf_client.py" --mode concurrent --base-url "$BASE_URL" \
    --workload-file "$WORKLOAD_FILE" --requests "$count" --concurrency 1 --allow-duplicate \
    --seed "$seed" --model "$MODEL_NAME" --stream true --rag true --injection-type kvcache \
    --max-tokens "$MAX_TOKENS" --temperature 0 --output-jsonl "$output"
}

echo "[PHASE] train a model bound to $EXPECTED_COMMIT"
start_stack fresh-training linucb "$CACHE_PREWARM_COUNT"
warmup_engines "$container_run_dir/training-engine-warmup.json"
run_client "$TRAINING_REQUESTS" "$SEED" "$container_run_dir/training-raw.jsonl"
test -s "$host_model"
test -s "$host_snapshots"
cp "$host_model" "$host_run_dir/model-before/linucb-model.json"
docker logs "$CONTAINER" > "$host_run_dir/service-logs/training.log" 2>&1

printf 'pair\trepeat\torder\tstrategy\tseed\traw_file\tsummary_file\n' > "$host_run_dir/run-order.tsv"
measurements=()
for repeat in $(seq 1 "$REPEATS"); do
  pair="$repeat"
  pair_seed=$((SEED + 100000 + pair))
  if [ $((pair % 2)) -eq 1 ]; then
    order="rr-first"
    policies="round_robin linucb"
  else
    order="linucb-first"
    policies="linucb round_robin"
  fi
  for strategy in $policies; do
    if [ "$strategy" = "linucb" ]; then
      mode="loaded-frozen"
      label="frozen-linucb"
    else
      mode="round-robin"
      label="round-robin"
    fi
    prefix="pair-${pair}-${label}"
    raw_file="$container_run_dir/${prefix}.jsonl"
    summary_file="$container_run_dir/${prefix}-summary.json"
    echo "[PHASE] pair=$pair order=$order strategy=$strategy seed=$pair_seed"
    start_stack "$mode" "$strategy" 0
    warmup_engines "$container_run_dir/${prefix}-engine-warmup.json"
    run_client "$MEASURE_REQUESTS" "$pair_seed" "$raw_file"
    python3 "$PROJECT_HOST/scripts/analyze_experiment_jsonl.py" \
      "$host_run_dir/${prefix}.jsonl" --output "$host_run_dir/${prefix}-summary.json"
    docker logs "$CONTAINER" > "$host_run_dir/service-logs/${prefix}.log" 2>&1
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$pair" "$repeat" "$order" "$strategy" "$pair_seed" \
      "$host_run_dir/${prefix}.jsonl" "$host_run_dir/${prefix}-summary.json" >> "$host_run_dir/run-order.tsv"
    measurements+=("$host_run_dir/${prefix}.jsonl")
    if [ "$strategy" = "linucb" ]; then
      cmp "$host_model" "$host_run_dir/model-before/linucb-model.json"
    fi
  done
done

cp "$host_model" "$host_run_dir/model-after/linucb-model.json"
python3 "$PROJECT_HOST/scripts/analyze_experiment_jsonl.py" \
  "${measurements[@]}" --output "$host_run_dir/all-run-summaries.json"
python3 "$PROJECT_HOST/scripts/validate_rr_frozen_linucb_paired.py" \
  --config "$host_run_dir/config.json" --run-order "$host_run_dir/run-order.tsv" \
  --model-before "$host_run_dir/model-before/linucb-model.json" \
  --model-after "$host_run_dir/model-after/linucb-model.json" \
  --expected-commit "$EXPECTED_COMMIT" --output "$host_run_dir/validation-and-paired-summary.json"
test -z "$(git -C "$PROJECT_HOST" status --porcelain)"
echo "[DONE] RR vs frozen LinUCB paired experiment passed its facility checks"
