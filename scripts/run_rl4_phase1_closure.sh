#!/usr/bin/env bash
# Issue #8 bounded train/load/freeze facility acceptance. No performance claim.
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
FROZEN_REQUESTS="${FROZEN_REQUESTS:-80}"
CONCURRENCY="${CONCURRENCY:-1}"
SEED="${SEED:-20260723}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-30}"
SNAPSHOT_INTERVAL="${SNAPSHOT_INTERVAL:-20}"
LINUCB_ALPHA="${LINUCB_ALPHA:-0.4}"
LINUCB_LAMBDA="${LINUCB_LAMBDA:-1.0}"
REWARD_SCALE_MS="${REWARD_SCALE_MS:-1000.0}"
REWARD_CLIP="${REWARD_CLIP:-5.0}"
ENGINE_WARMUP_REQUESTS_PER_INSTANCE="${ENGINE_WARMUP_REQUESTS_PER_INSTANCE:-2}"

if [ "$PROJECT_HOST" != "$(readlink -f "$ROOT")" ]; then echo "[FAIL] PROJECT_HOST must be this checkout" >&2; exit 2; fi
if [ "$TRAINING_REQUESTS" -ne 240 ] || [ "$FROZEN_REQUESTS" -ne 80 ] || [ "$CONCURRENCY" -ne 1 ]; then
  echo "[FAIL] Issue #8 acceptance is fixed at training=240 frozen=80 concurrency=1" >&2; exit 2
fi
actual_commit="$(git -C "$PROJECT_HOST" rev-parse HEAD)"
if [ "$actual_commit" != "$EXPECTED_COMMIT" ]; then echo "[FAIL] commit mismatch expected=$EXPECTED_COMMIT actual=$actual_commit" >&2; exit 2; fi
if [ -n "$(git -C "$PROJECT_HOST" status --porcelain)" ]; then echo "[FAIL] worktree is not clean" >&2; exit 2; fi
EXPECTED_COMMIT="$EXPECTED_COMMIT" REQUIRE_CLEAN_WORKTREE=1 PROJECT_ROOT="$PROJECT_HOST" bash "$PROJECT_HOST/scripts/verify_source_sync.sh"

speed_factors="$(python3 - "$CALIBRATION_AGGREGATE" <<'PY'
import json, sys
d=json.load(open(sys.argv[1], encoding="utf-8"))
if d.get("status") != "passed": raise SystemExit("calibration status is not passed")
v=d.get("diagnostic_speed_factors_csv")
if not isinstance(v, str) or len(v.split(",")) != 4: raise SystemExit("missing four speed factors")
print(v)
PY
)"

run_id="${RUN_ID:-phase1-closure-$(date -u +%Y%m%dT%H%M%SZ)}"
host_run_dir="$PROJECT_HOST/log/rl4-phase1-closure/$run_id"
container_run_dir="$PROJECT_IN_CONTAINER/log/rl4-phase1-closure/$run_id"
if [ -e "$host_run_dir" ]; then echo "[FAIL] result directory exists: $host_run_dir" >&2; exit 2; fi
mkdir -p "$host_run_dir"/{model-before,model-after,service-logs,figures}
cp "$CALIBRATION_AGGREGATE" "$host_run_dir/calibration-aggregate.json"
container_model="$container_run_dir/linucb-model.json"
container_snapshots="$container_run_dir/parameter-snapshots.jsonl"
host_model="$host_run_dir/linucb-model.json"
host_snapshots="$host_run_dir/parameter-snapshots.jsonl"

finalize() {
  code="$1"; trap - EXIT
  printf '{"exit_code":%s,"finished_at_utc":"%s"}\n' "$code" "$(date -u +%FT%TZ)" > "$host_run_dir/run-status.json"
  (cd "$host_run_dir" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum) > "$host_run_dir/SHA256SUMS"
  archive="$host_run_dir.tar.gz"; tar -C "$(dirname "$host_run_dir")" -czf "$archive" "$(basename "$host_run_dir")"; sha256sum "$archive" > "$archive.sha256"
  echo "[RESULT] directory=$host_run_dir"; echo "[RESULT] archive=$archive"; exit "$code"
}
trap 'finalize $?' EXIT

python3 - "$host_run_dir/config.json" <<PY
import json
json.dump({"experiment_kind":"rl4_phase1_closure","claim_boundary":"facility acceptance only; no performance conclusion","git_commit":"$EXPECTED_COMMIT","training_requests":240,"frozen_requests":80,"concurrency":1,"seed":$SEED,"warmup_requests":$WARMUP_REQUESTS,"parameter_snapshot_interval":$SNAPSHOT_INTERVAL,"linucb_alpha":$LINUCB_ALPHA,"linucb_lambda":$LINUCB_LAMBDA,"reward_scale_ms":$REWARD_SCALE_MS,"reward_clip":$REWARD_CLIP,"speed_factors":"$speed_factors","kv_residency_scope":"global","kv_link_scope":"global","workload_file":"$WORKLOAD_FILE","model_name":"$MODEL_NAME"},open("$host_run_dir/config.json","w",encoding="utf-8"),indent=2)
PY
python3 - "$host_run_dir/environment.json" <<'PY'
import json, platform, subprocess, sys
def run(cmd):
    p=subprocess.run(cmd,text=True,capture_output=True); return {"exit_code":p.returncode,"stdout":p.stdout.strip(),"stderr":p.stderr.strip()}
json.dump({"captured_at_utc":run(["date","-u","+%FT%TZ"])["stdout"],"platform":platform.platform(),"python":platform.python_version(),"docker":run(["docker","version"]),"nvidia_smi":run(["nvidia-smi"])},open(sys.argv[1],"w",encoding="utf-8"),indent=2)
PY

start_stack() {
  mode="$1"; load_path=""; frozen=0
  if [ "$mode" != "fresh-training" ]; then load_path="$container_model"; fi
  if [ "$mode" = "loaded-frozen" ]; then frozen=1; fi
  EXPECTED_COMMIT="$EXPECTED_COMMIT" PROJECT_HOST="$PROJECT_HOST" PROJECT_IN_CONTAINER="$PROJECT_IN_CONTAINER" \
  MODEL_DIR="$MODEL_DIR" MODEL_NAME="$MODEL_NAME" CONTAINER="$CONTAINER" REDIS_CONTAINER="$REDIS_CONTAINER" PREWARM_COUNT=0 \
  INSTANCE_PREFILL_CAPACITY_RATIOS="$speed_factors" PROXY_INSTANCE_STRATEGY=linucb PROXY_RL_ENABLED=1 \
  PROXY_RL_ALPHA="$LINUCB_ALPHA" PROXY_RL_LAMBDA="$LINUCB_LAMBDA" PROXY_RL_WARMUP_REQUESTS="$WARMUP_REQUESTS" \
  PROXY_RL_REWARD_TTFT_SCALE_MS="$REWARD_SCALE_MS" PROXY_RL_REWARD_CLIP="$REWARD_CLIP" \
  PROXY_RL_MODEL_LOAD_PATH="$load_path" PROXY_RL_MODEL_SAVE_PATH="$container_model" PROXY_RL_SOURCE_COMMIT="$EXPECTED_COMMIT" PROXY_RL_FROZEN="$frozen" \
  PROXY_RL_PARAMETER_SNAPSHOT_PATH="$container_snapshots" PROXY_RL_PARAMETER_SNAPSHOT_INTERVAL="$SNAPSHOT_INTERVAL" \
  KDN_TEXT_DB_DIR="$container_run_dir/kdn-text-db" \
  PROXY_RL_KV_RESIDENCY_SCOPE=global PROXY_RL_KV_LINK_SCOPE=global bash "$PROJECT_HOST/scripts/start_rl4_signal_smoke.sh"
  cp "$PROJECT_HOST/log/rl4/status.txt" "$host_run_dir/${mode}-startup-status.txt"
}
warmup_engines() {
  mode="$1"; docker exec -e PYTHONPATH="$PROJECT_IN_CONTAINER" -w "$PROJECT_IN_CONTAINER" "$CONTAINER" python3 \
    "$PROJECT_IN_CONTAINER/scripts/warmup_vllm_instances.py" --model "$MODEL_NAME" --instance-count 4 \
    --requests-per-instance "$ENGINE_WARMUP_REQUESTS_PER_INSTANCE" --output "$container_run_dir/${mode}-engine-warmup.json"
}
run_client() {
  count="$1"; seed="$2"; output="$3"
  docker exec -e PYTHONPATH="$PROJECT_IN_CONTAINER" -w "$PROJECT_IN_CONTAINER" "$CONTAINER" python3 \
    "$PROJECT_IN_CONTAINER/client/perf_client.py" --mode concurrent --base-url "$BASE_URL" --workload-file "$WORKLOAD_FILE" \
    --requests "$count" --concurrency 1 --allow-duplicate --seed "$seed" --model "$MODEL_NAME" --stream true --rag true \
    --injection-type kvcache --max-tokens 1 --temperature 0 --output-jsonl "$output"
}

echo "[PHASE] fresh-training"
start_stack fresh-training; warmup_engines fresh-training; run_client 240 "$SEED" "$container_run_dir/training-raw.jsonl"
test -s "$host_model"; test -s "$host_snapshots"; cp "$host_model" "$host_run_dir/model-before/linucb-model.json"
docker logs "$CONTAINER" > "$host_run_dir/service-logs/fresh-training.log" 2>&1

echo "[PHASE] loaded-training startup/load gate"
start_stack loaded-training; warmup_engines loaded-training; docker logs "$CONTAINER" > "$host_run_dir/service-logs/loaded-training.log" 2>&1
cmp "$host_model" "$host_run_dir/model-before/linucb-model.json"

echo "[PHASE] loaded-frozen"
start_stack loaded-frozen; warmup_engines loaded-frozen; run_client 80 "$((SEED + 1))" "$container_run_dir/frozen-raw.jsonl"
docker logs "$CONTAINER" > "$host_run_dir/service-logs/loaded-frozen.log" 2>&1
cp "$host_model" "$host_run_dir/model-after/linucb-model.json"

python3 "$PROJECT_HOST/scripts/analyze_linucb_convergence.py" --requests-jsonl "$host_run_dir/training-raw.jsonl" --output "$host_run_dir/convergence-summary.json"
python3 "$PROJECT_HOST/scripts/validate_rl4_phase1_closure.py" --training-jsonl "$host_run_dir/training-raw.jsonl" --frozen-jsonl "$host_run_dir/frozen-raw.jsonl" \
  --snapshots-jsonl "$host_snapshots" --model-before "$host_run_dir/model-before/linucb-model.json" --model-after "$host_run_dir/model-after/linucb-model.json" \
  --analysis "$host_run_dir/convergence-summary.json" --config "$host_run_dir/config.json" --environment "$host_run_dir/environment.json" \
  --expected-commit "$EXPECTED_COMMIT" --output "$host_run_dir/validation-summary.json"
test -z "$(git -C "$PROJECT_HOST" status --porcelain)"
echo "[DONE] Issue #8 bounded closure passed"
