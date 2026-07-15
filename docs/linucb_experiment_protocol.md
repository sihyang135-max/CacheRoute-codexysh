# LinUCB 二级调度实验协议

## 1. 当前判断

旧实验只能证明 LinUCB 更新和共享 Redis KV 复用曾经发生，不能证明 LinUCB 优于基线：

- 4 个实例共享同一个 Redis，预热后运行时 `payload_bytes=0`，实际没有实例间 KV 传输差异。
- 4 个实例位于同一台机器并使用回环链路，网络特征近似常量。
- 旧配置只预热 1 个知识块，99 个问题没有形成完整知识工作集。
- 旧 reward 使用本机默认 `15000 ms` SLO 归一化，300 ms 与 900 ms 的 reward 仅差 0.04，低于探索项量级。
- 旧 TTFT 可能把 role-only SSE 块当成首 token；吞吐量包含失败请求。
- Least Inflight 读取的心跳 `inflight` 可能为 null，基线会退化成固定选择第一个实例。

因此先做正确性门禁，再做环境、状态和参数消融。任何门禁失败时不得进入论文性能表。

## 2. 三端一致性门禁

正式实验只使用 Git 提交态，禁止直接在服务器上修改后继续跑实验。

```bash
cd /llm-stack/cr0624
EXPECTED_COMMIT=<local_commit> bash scripts/verify_source_sync.sh
```

必须记录：Git commit、源码 SHA-256、Docker image ID、模型目录、GPU 型号、驱动、CUDA、PyTorch、vLLM、LMCache、Redis 配置和完整启动环境变量。

## 3. 正确性冒烟测试

启动参数至少显式指定：

```bash
INSTANCE_COUNT=4
TENSOR_PARALLEL_SIZE=1
PREWARM_COUNT=all
PROXY_INSTANCE_STRATEGY=linucb
PROXY_RL_ENABLED=1
PROXY_RL_ALPHA=0.4
PROXY_RL_LAMBDA=1.0
PROXY_RL_WARMUP_REQUESTS=30
PROXY_RL_REWARD_TTFT_SCALE_MS=1000
PROXY_RL_REWARD_CLIP=5
```

容器内发送 120 个低速请求：

```bash
cd /workspace/llm-stack/CacheRoute
PYTHONPATH=. python3 client/perf_client.py \
  --mode rps --rps 0.5 --requests 120 --allow-duplicate --seed 20260715 \
  --base-url http://127.0.0.1:7001 \
  --workload-file client/taskset/workload_nq.json \
  --model deepseek-r1-distill-qwen-7b \
  --stream true --rag true --injection-type kvcache --max-tokens 64 \
  --monitor-gpu --gpu-ids 0,1,2,3 \
  --output-jsonl log/experiments/smoke-linucb.jsonl
```

通过标准：

- 成功率 100%，`missing_meta_requests=0`，`trace_warning_requests=0`。
- 120 个请求均有真实 `client_ttft_ms`、选中实例和控制端口。
- 前 30 个有效更新为 warmup，之后存在 `selection_phase=linucb`。
- `rl_updated_requests=120`，候选分数和特征完整，不发生静默 fallback。
- 四个实例均被注册且请求分布不是由空负载字段导致的固定首实例。
- LMCache 日志能确认 store/retrieve；共享 Redis 场景允许运行时传输字节为 0，但必须在结果中明确标注。

汇总命令：

```bash
python3 scripts/analyze_experiment_jsonl.py \
  log/experiments/smoke-linucb.jsonl --discard-first 30
```

## 4. 第一阶段：同构共享缓存基线

目的不是预设 LinUCB 获胜，而是验证在只有动态队列变化时是否不劣于合理基线。

策略包括 `round_robin`、修正后的 `least_inflight` 和 `linucb`。先用 RPS 梯度寻找饱和点，例如 0.5、1、2、4 req/s；每点 300 个请求。正式比较选择低载、中载和接近饱和三个点，每个策略、负载点运行至少 5 个独立重复，种子和运行顺序配对并随机化。

主指标：

- 成功请求的 mean client TTFT。
- 成功请求吞吐量 req/s。
- 输出 token 吞吐量 token/s。
- 成功率。

辅助指标为 median/P95 TTFT、实例负载分布和 GPU 利用率。LinUCB 与所有基线均丢弃相同数量的前 30 个请求后报告稳态结果，同时单独报告包含 warmup 的端到端结果。

## 5. 第二阶段：知识密集型异构环境

共享 Redis 环境不能支撑“知识感知调度优越性”的论文结论。下一阶段必须建立每实例独立 KV 存储或独立 Redis 端口，使一次知识传输只温热被选实例，并让不同实例到 KDN 的带宽/排队差异真实作用于请求 TTFT。

在该环境中依次做：

1. 热点偏斜：均匀、Zipf 轻偏斜、Zipf 重偏斜。
2. 知识块大小：短、中、长分桶及混合分布。
3. 计算负载：稳定低载、稳定高载、突发负载和相位切换。
4. 网络条件：同构、稳定异构、带宽突降和恢复。
5. 缓存容量：充足与受限，观察迁移、命中和淘汰。

若算法状态中没有“知识在各实例的驻留/命中可能性”，则 LinUCB 无法利用独立缓存带来的主要差异。该项应作为代码检查和状态消融的重点，不能只通过调整 alpha 解决。

## 6. 参数与状态消融

在固定中载和异构场景下依次改变单个因素：

- `alpha`: 0、0.05、0.1、0.2、0.4。
- `lambda`: 0.1、1、10。
- warmup: 0、10、30、100。
- reward scale: 500、1000、2000 ms。
- 状态组：仅队列；队列+请求长度；队列+网络；队列+缓存驻留；全部状态。
- reward：直接 TTFT、裁剪 TTFT，以及后续可能的 TTFT+失败惩罚。

调参数据与最终报告数据必须使用不同的运行种子；不能在测试集上选出最好参数后仍把同一批结果作为论文主表。

## 7. 进入论文阶段的判据

只有同时满足以下条件，才开始写性能结论：

- 三端源码和环境指纹可复现。
- 真实 KV 路径、真实首 token 和成功吞吐均可审计。
- LinUCB 在至少一个与论文背景一致的异构知识工作负载上，对强基线的 mean TTFT 或吞吐有稳定优势，并有置信区间。
- 在同构或不适用场景中不夸大收益，明确算法适用边界。
- 消融能解释收益来自哪些状态、reward 和在线适应机制，而不是偶然参数。
