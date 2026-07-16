# LinUCB 二级调度实验协议

## 1. 当前结论边界

旧实验只能证明 LinUCB 更新和共享 Redis KV 复用曾经发生，不能证明 LinUCB 优于基线：

- 四个实例共享同一个 Redis。预热后 `keys_injected=0`、`payload_bytes=0` 通常表示 KV 已驻留，不表示 KV 路径未启用。
- 共享 Redis 没有实例级知识位置差异，不能用于证明“知识感知放置/路由”的优势。
- 旧 reward 被 15000 ms 客户端 SLO 过度缩小，探索项远大于 reward 差异。
- 旧 TTFT 可能把 role-only SSE 块当作首 token；旧吞吐统计也可能包含失败请求。
- 旧 Least-Inflight 依赖可能为空的心跳 `inflight`，平局时会固定选择第一个实例。
- 单一问题、固定顺序和不配对种子不足以证明上下文在线学习有效。

本阶段验证的命题是：**在异构计算能力、动态队列和实例级 KV 就绪代价共同存在时，紧凑的二级 contextual bandit 能否在线学习路由代价与 TTFT 的关系，并改善平均 TTFT 或系统吞吐。**

实例级知识驻留是下一阶段命题，必须使用独立 Redis/KV 域后再验证。

## 2. 源码与环境门禁

正式实验只使用可定位的 Git 提交。禁止直接修改服务器后继续产出论文数据。

```bash
cd /llm-stack/cr0624
EXPECTED_COMMIT=<git_commit> bash scripts/verify_source_sync.sh
```

每次实验必须保存：

- Git commit 和 `env/experiment_source.sha256` 校验结果；
- Docker image ID、模型路径和模型哈希；
- GPU 型号、驱动、CUDA、PyTorch、vLLM、LMCache 与 Redis 配置；
- 完整启动参数、矩阵配置和 `run-order.tsv`；
- 每个请求的原始 JSONL，不只保存聚合结果。

任何门禁失败都不得进入论文性能表。

## 3. 已实现的正确性修复

- 使用共享的 3 维模型 `[bias, compute_delta, kv_ready_delta]`；两个代价都先减去候选实例中的最小值，所有实例共有的网络代价严格归零。
- `compute_delta` 由本地预约时间线、KV 重用后的 residual prefill 和离线标定的实例 prefill capacity 构成。
- `kv_ready_delta` 由实例级 KV 驻留、KDN 传输排队、后台网卡带宽余量、实际 KV delivery EWMA 和 RTT 构成；请求路径不查询 Prometheus 或远程服务。
- 当前共享 Redis/KDN 环境使用 `PROXY_RL_KV_RESIDENCY_SCOPE=global` 和 `PROXY_RL_KV_LINK_SCOPE=global`；独立缓存与独立链路实验必须显式改为 `instance`。
- 探索奖励只依赖实例样本数，不依赖较大的计算或 KV 就绪代价。
- warmup 按实例已分配请求数均衡，即使反馈延迟也不会偏向某个实例。
- 每个策略在训练前直接预热全部 vLLM 引擎；这些请求绕过 Proxy，不污染 LinUCB 样本。
- 从选路到请求完成维护本地 route reservation，并纳入 Least-Inflight 和 LinUCB 队列状态。
- 状态使用全生命周期 inflight 与 prefill 压力；已知成本特征的系数约束为非正。
- UCB 探索项只作用于请求上下文，不因候选实例负载高、样本少而奖励拥塞状态。
- KDN 独占物理传输排队；Proxy 不再重复模拟同一传输等待。
- KV 注入使用 `SET NX`，区分 cold injection 与 resident hit。
- TTFT 只在首个 content/reasoning token 到达时记录，不计 role-only 块。
- reward 为缩放并裁剪的直接 TTFT：`-min(TTFT / scale_ms, clip)`。
- 所有候选实例的特征、分数、排除原因、选择阶段和实例级更新数写入 trace。

## 4. 正确性冒烟测试

在正式矩阵前，先用完整 NQ workload 运行至少 120 个请求。必须满足：

- `success_rate=1.0`；
- `missing_meta_requests=0`、`trace_warning_requests=0`；
- 每个成功请求都有真实 `client_ttft_ms`；
- 四个实例均注册，控制端口与被选实例一致；
- warmup 后出现 `selection_phase=linucb`；
- `rl_updated_requests` 等于成功且可观测首 token 的请求数；
- `kv_ack_ok_requests` 正常，且 resident hit 与实际传输分别统计；
- 不发生未解释的 fallback 或持续缺失 Prometheus 状态。

共享 Redis 热缓存场景允许 `kv_transfer_requests=0`，但必须同时看到 resident-hit 证据，并在论文中明确说明。

## 5. 第一阶段：异构计算与动态队列

建议拓扑为 TP4/TP2/TP1/TP1：

```bash
export INSTANCE_COUNT=4
export INSTANCE_GPU_GROUPS='0,1,2,3;4,5;6;7'
export INSTANCE_TP_SIZES='4,2,1,1'
export INSTANCE_PREFILL_CAPACITY_RATIOS='<离线标定后归一化的四个值>'
```

正式比较 `round_robin`、修正后的 `least_inflight` 和 `linucb`。默认参数先固定为：

```bash
export LINUCB_ALPHA=0.05
export LINUCB_LAMBDA=1.0
export WARMUP_REQUESTS=200
export ENGINE_WARMUP_REQUESTS_PER_INSTANCE=2
export REWARD_SCALE_MS=1000
export REWARD_CLIP=5
```

运行矩阵：

```bash
cd /llm-stack/cr0624
PROJECT_HOST=/llm-stack/cr0624 \
MODEL_DIR=<model_dir> \
MODEL_NAME=deepseek-r1-distill-qwen-7b \
CONTAINER=cr0624-rl \
REDIS_CONTAINER=lmcache-redis \
INSTANCE_GPU_GROUPS='0,1,2,3;4,5;6;7' \
INSTANCE_TP_SIZES='4,2,1,1' \
CONCURRENCIES=1,4,8,16 \
REPEATS=3 REQUESTS=100 WARMUP_REQUESTS=200 \
ENGINE_WARMUP_REQUESTS_PER_INSTANCE=2 \
bash scripts/run_policy_matrix.sh
```

矩阵脚本对每个策略单独重启服务栈，先直接预热所有 vLLM 引擎，再执行策略 warmup；同一并发度和重复轮次使用配对种子，策略执行顺序轮换。共享 Redis 只在矩阵开始前统一预热，不能让某个策略单独改变测量条件。

先用并发度或 RPS 扫描找出低载、中载和接近饱和三个区域。论文主表至少使用 5 个独立重复；3 次只用于初步诊断。

## 6. 指标与统计

主指标：

- 成功请求的 mean client TTFT；
- successful req/s；
- output token/s；
- success rate。

辅助指标：median/P95 TTFT、wall time、实例选择分布、GPU 利用率、fallback、KV resident hit/transfer、`rl_context_build_us`、`rl_bandit_score_us` 和 LinUCB 更新轨迹。P99 不作为本阶段优越性的主要证据；调度开销必须单独报告，不能隐藏在端到端 TTFT 中。

每个负载点报告重复实验的均值、标准差和 95% 置信区间。策略比较使用同一 workload、同一随机种子和同一缓存初始状态。调参种子与最终报告种子必须分离。

LinUCB warmup 是算法成本的一部分：

- 主稳态表报告完成固定 warmup 后的测量阶段；
- 另表报告包含 warmup 的端到端结果和达到稳定收益所需请求数。

## 7. 参数与状态消融

固定中载和异构拓扑后，每次只改变一个因素：

- `alpha`: 0、0.02、0.05、0.1、0.2；
- `lambda`: 0.1、1、10；
- warmup: 0、40、100、200；
- reward scale: 500、1000、2000 ms；
- 上下文消融：仅 `compute_delta`；仅 `kv_ready_delta`；完整的两个相对代价；
- reward：直接 TTFT、裁剪 TTFT、TTFT 加失败惩罚。

若 LinUCB 只在某个偶然参数上获胜，不能形成普适结论。应优先检查状态是否可观测、量纲是否平衡以及环境是否真的存在可学习差异。

## 8. 第二阶段：实例级知识驻留

共享 Redis 无法支撑“知识密集型任务调度”主张。该阶段需要：

- 每实例独立 KV 存储或独立 Redis namespace/端口；
- 可观测的实例-知识驻留矩阵、命中概率与剩余容量；
- 不同实例到 KDN 的真实或可控带宽与排队差异；
- 热点偏斜、知识块长度、缓存容量和负载突发的组合 workload；
- 路由后只温热被选实例，从而形成可验证的长期决策影响。

实例级知识驻留不再单独增加模型维度，而是通过 `missing_kv_ratio` 改变 KV 就绪代价。网络与知识都相同时，`kv_ready_delta=0`；仅调 `alpha` 不能弥补候选实例之间缺少可学习差异。

## 9. 进入论文阶段的判据

只有同时满足以下条件，才开始撰写性能结论：

- 本地、Git、服务器源码和环境指纹一致且可复现；
- 真实 KV 路径、真实首 token 和成功吞吐均可审计；
- LinUCB 在至少一个与论文命题一致的异构 workload 上，对强基线的 mean TTFT 或吞吐有稳定优势和置信区间；
- 在同构或不适用场景中不夸大收益，明确算法适用边界；
- 消融能解释收益来自状态、reward 和在线适应，而不是运行顺序或偶然参数。
