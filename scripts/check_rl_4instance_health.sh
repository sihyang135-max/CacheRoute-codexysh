#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/workspace/llm-stack/CacheRoute}"
INSTANCE_COUNT="${INSTANCE_COUNT:-4}"
HEALTH_PASSES="${HEALTH_PASSES:-3}"
HEALTH_INTERVAL_S="${HEALTH_INTERVAL_S:-10}"
REQUIRE_KDN_EMBEDDINGS="${REQUIRE_KDN_EMBEDDINGS:-0}"
KNOWLEDGE_READY_TIMEOUT_S="${KNOWLEDGE_READY_TIMEOUT_S:-180}"
PID_DIR="$PROJECT/log/rl4/pids"
LOG_DIR="$PROJECT/log/rl4"

failures=()

check_pid() {
  local name="$1" pid_file="$PID_DIR/$1.pid" pid
  if [ ! -s "$pid_file" ]; then
    failures+=("$name pid file missing")
    return
  fi
  pid="$(cat "$pid_file")"
  if ! kill -0 "$pid" 2>/dev/null; then
    failures+=("$name process $pid is not alive")
  fi
}

check_http() {
  local name="$1" url="$2"
  if ! curl -fsS --max-time 5 "$url" >/dev/null 2>&1; then
    failures+=("$name endpoint failed: $url")
  fi
}

check_registered_instances() {
  local payload count
  if ! payload="$(curl -fsS --max-time 5 http://127.0.0.1:8002/v1/instance/list)"; then
    failures+=("proxy instance list unavailable")
    return
  fi
  count="$(python3 -c 'import json,sys; print(len(json.load(sys.stdin)))' <<<"$payload")"
  if [ "$count" -ne "$INSTANCE_COUNT" ]; then
    failures+=("registered instances=$count expected=$INSTANCE_COUNT")
  fi
}

knowledge_is_ready() {
  LAST_KDN_STATUS=""
  LAST_SCHEDULER_STATUS=""
  if ! LAST_KDN_STATUS="$(curl -fsS --max-time 10 -X POST \
      http://127.0.0.1:9101/knowledge/pool_status \
      -H 'Content-Type: application/json' -d '{"sample_limit":0}')"; then
    return 1
  fi
  if ! python3 -c '
import json,sys
x=json.load(sys.stdin)
total=int(x.get("total_blocks") or 0)
ready=int(x.get("embedding_ready_blocks") or 0)
raise SystemExit(0 if total > 0 and ready == total else 1)
' <<<"$LAST_KDN_STATUS"; then
    return 1
  fi

  if ! LAST_SCHEDULER_STATUS="$(curl -fsS --max-time 10 http://127.0.0.1:7001/debug/status)"; then
    return 1
  fi
  if ! python3 -c '
import json,sys
x=json.load(sys.stdin)
ok=(x.get("knowledge_loaded") is True and int(x.get("entries") or 0) > 0 and
    int(x.get("faiss_total") or 0) == int(x.get("entries") or 0) and
    x.get("kdn_last_refresh_ok") is True)
raise SystemExit(0 if ok else 1)
' <<<"$LAST_SCHEDULER_STATUS"; then
    return 1
  fi
  return 0
}

check_knowledge() {
  if ! knowledge_is_ready; then
    failures+=("scheduler knowledge index is not ready")
  fi
}

dump_failures() {
  printf '[FAIL] %s\n' "${failures[@]}" >&2
  echo "=== recent service logs ===" >&2
  for log in "$LOG_DIR"/vllm-*.log "$LOG_DIR"/scheduler.log \
      "$LOG_DIR"/kdn.log "$LOG_DIR"/proxy.log "$LOG_DIR"/instance-*.log; do
    [ -f "$log" ] || continue
    echo "--- $log ---" >&2
    tail -n 30 "$log" >&2 || true
  done
}

if [ "$REQUIRE_KDN_EMBEDDINGS" = "1" ]; then
  knowledge_ready=0
  for _ in $(seq 1 "$KNOWLEDGE_READY_TIMEOUT_S"); do
    if knowledge_is_ready; then
      knowledge_ready=1
      break
    fi
    sleep 1
  done
  if [ "$knowledge_ready" -ne 1 ]; then
    echo "[FAIL] knowledge index did not become ready within ${KNOWLEDGE_READY_TIMEOUT_S}s" >&2
    echo "KDN status: ${LAST_KDN_STATUS:-unavailable}" >&2
    echo "Scheduler status: ${LAST_SCHEDULER_STATUS:-unavailable}" >&2
    exit 1
  fi
  echo "[OK] KDN embeddings and Scheduler FAISS index are ready"
fi

for pass in $(seq 1 "$HEALTH_PASSES"); do
  failures=()
  for idx in $(seq 0 $((INSTANCE_COUNT - 1))); do
    check_pid "vllm-$idx"
    check_http "vLLM-$idx" "http://127.0.0.1:$((18000 + idx))/v1/models"
  done
  check_pid scheduler
  check_pid kdn
  check_pid proxy
  check_http Scheduler http://127.0.0.1:7001/debug/status
  check_http Scheduler-CP http://127.0.0.1:7002/healthz
  check_http KDN http://127.0.0.1:9101/v1/topology/ping
  check_http Proxy http://127.0.0.1:8002/healthz
  for idx in $(seq 0 $((INSTANCE_COUNT - 1))); do
    check_pid "instance-$idx"
    check_http "Instance-$idx-CP" "http://127.0.0.1:$((19101 + idx))/healthz"
  done
  check_registered_instances
  if [ "$REQUIRE_KDN_EMBEDDINGS" = "1" ]; then
    check_knowledge
  fi
  if [ "${#failures[@]}" -ne 0 ]; then
    dump_failures
    exit 1
  fi
  echo "[OK] health pass $pass/$HEALTH_PASSES"
  if [ "$pass" -lt "$HEALTH_PASSES" ]; then
    sleep "$HEALTH_INTERVAL_S"
  fi
done

echo "[OK] serving stack remained healthy across $HEALTH_PASSES checks"
