# LinUCB 四实例 Docker 实验

## 前置条件

- 服务器有 8 张 5090；
- 模型能以 TP=2 运行。未量化 70B 通常不能装入 2 张 5090，应改用较小模型或自行添加量化参数；
- 主机已准备 CacheRoute Docker 镜像、Redis Docker 镜像和项目目录；
- 已完成 LMCache 的 `token_database.py`、`memory_management.py` 补丁替换。

## 一键启动

在 Docker 宿主机执行：

```bash
cd <项目目录>
chmod +x scripts/start_rl_4instance_*.sh

export PROJECT_HOST=$(pwd)
export MODEL_DIR=/workspace/llm-stack/models/<模型目录>
export MODEL_NAME=<模型服务名>
export PREWARM_COUNT=1       # 首次实验建议先预热 1 个知识块；设为 0 则跳过
bash scripts/start_rl_4instance_docker.sh
```

脚本会自动：启动 Redis；创建或启动 Docker 容器；启动 4 个 TP=2 vLLM、Scheduler、KDN、LinUCB Proxy 和 4 个 Instance；检查端口与 Instance 注册；可选预热 KV。

## 验证

```bash
docker exec -it cacheroute-rl bash -lc 'cat /workspace/llm-stack/CacheRoute/log/rl4/status.txt'
docker exec -it cacheroute-rl bash -lc 'curl -s http://127.0.0.1:8002/v1/instance/list'
```

随后发送流式 KVCache 请求：

```bash
curl -N http://127.0.0.1:7001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"<MODEL_NAME>","messages":[{"role":"user","content":"What is CacheRoute?"}],"max_tokens":64,"stream":true,"RAG":true,"Injection_type":"kvcache"}'
```

前 30 个有效请求为 Least Inflight 冷启动；之后才会使用 LinUCB。响应末尾的 `cacheroute_meta` 中出现 `rl_updated=1`，表示一次在线更新成功。

## 基线

每轮对比前重新运行脚本，并把 `--strategy linucb` 改为 `round_robin` 或 `least_inflight`；保持模型、知识库、负载和随机种子不变。
