# 5090×8 上运行二级 LinUCB 实验

## 前提

二级调度至少需要两个独立的 vLLM Instance。若将 70B 模型以 TP=8 运行在全部 8 张卡上，系统只有一个 Instance，无法验证 Instance 选择策略。

建议的实验部署是 8 个单卡 Instance（例如 8B/14B 模型），或 4 个 TP=2 Instance（模型需能放入两卡显存）。每个 Instance 需要独立的 vLLM 端口、Instance 端口、控制面端口、`INSTANCE_ID` 和 `VLLM_METRICS_URL`。

## 启动约定

以下例子使用同一主机的 8 个单卡 vLLM 实例：

| GPU | vLLM | Instance | Instance 控制面 |
| --- | --- | --- | --- |
| 0..7 | 8000..8007 | 9001..9008 | 9102..9109 |

首先启动 Scheduler、KDN 和启用 LinUCB 的 Proxy：

```bash
export PYTHONPATH=/workspace/llm-stack/cacheroute-rl-main
cd /workspace/llm-stack/cacheroute-rl-main/test
python3 demo_scheduler.py --cacheroute
python3 demo_proxy.py --strategy linucb --injection-strategy default
```

然后在每个 GPU 上启动一个 vLLM；务必开启指标端点（vLLM 默认在服务端口暴露 `/metrics`）：

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" --served-model-name llama3-8b \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 1 --gpu-memory-utilization 0.82 \
  --max-model-len 4096 --kv-offloading-backend lmcache \
  --kv-offloading-size 32 --disable-hybrid-kv-cache-manager --kv-cache-metrics
```

为该 vLLM 启动对应 Instance：

```bash
export PROXY_CP_URL=http://127.0.0.1:8002
export INSTANCE_ID=inst-gpu0
export VLLM_BASE_URL=http://127.0.0.1:8000
export VLLM_METRICS_URL=http://127.0.0.1:8000/metrics
python3 test/demo_instance.py --host 127.0.0.1 --port 9001
```

其余 7 个实例以相同方式启动，替换 GPU 编号、vLLM 端口、`INSTANCE_ID` 和 Instance 端口即可。

## 必要环境变量

```bash
export PROXY_INSTANCE_STRATEGY=linucb
export PROXY_RL_ALPHA=0.4
export PROXY_RL_WARMUP_REQUESTS=30
export PROXY_RL_PROMETHEUS_INTERVAL_S=1.0
export PROXY_RL_PROMETHEUS_STALE_S=3.0
export PROXY_RL_KV_USAGE_LIMIT=0.90
```

确认一个 vLLM 指标可用：

```bash
curl -s http://127.0.0.1:8000/metrics | grep -E 'gpu.*cache.*usage|gpu_cache_usage'
```

当前实现识别 `vllm:gpu_cache_usage_perc`、`vllm_gpu_cache_usage_perc` 和 `vllm_gpu_kv_cache_usage_perc`。若实际 vLLM 指标名不同，请在 `proxy/metrics/prometheus_cache.py` 的 `_KV_NAMES` 中加入该名称。

## 网络实验说明

单机 8 卡时，KDN 到各 Instance 的物理链路通常相同。若要验证网络感知特征，需要在多机部署，或通过容器网络/`tc netem` 为不同 Instance 制造可控的带宽和时延差异。
