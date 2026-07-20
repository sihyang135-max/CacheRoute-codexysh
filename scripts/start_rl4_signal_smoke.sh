#!/usr/bin/env bash
# Stable environment gate for Issue #4 phase 1.0. This is not a formal
# performance-measurement configuration because Scheduler/KDN embeddings run
# on CPU to avoid sharing vLLM GPUs.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export PROJECT_HOST="$ROOT"
export PROJECT_IN_CONTAINER="${PROJECT_IN_CONTAINER:-/workspace/llm-stack/CacheRoute}"
export MODEL_DIR="${MODEL_DIR:-/workspace/llm-stack/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}"
export MODEL_NAME="${MODEL_NAME:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}"
export CONTAINER="${CONTAINER:-cr0720-rl}"
export REDIS_CONTAINER="${REDIS_CONTAINER:-lmcache-redis}"

export PREWARM_COUNT=0
export INSTANCE_COUNT=4
export INSTANCE_GPU_GROUPS='0,1,2,3;4,5;6;7'
export INSTANCE_TP_SIZES='4,2,1,1'
export INSTANCE_PREFILL_CAPACITY_RATIOS='4,2,1,1'

export PROXY_INSTANCE_STRATEGY=linucb
export PROXY_RL_ENABLED=1
export PROXY_RL_ALPHA=0.4
export PROXY_RL_LAMBDA=1.0
export PROXY_RL_WARMUP_REQUESTS=30

export SCHEDULER_CUDA_VISIBLE_DEVICES=''
export KDN_CUDA_VISIBLE_DEVICES=''
export KDN_BACKFILL_EMBEDDINGS=1
export HEALTH_PASSES=3
export HEALTH_INTERVAL_S=10

exec bash "$ROOT/scripts/start_rl_4instance_docker.sh"
