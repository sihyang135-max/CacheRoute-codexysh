# RL4 路由信号 smoke 协议（2026-07-22）

## 唯一目标与结论边界

唯一目标是完成 Issue #4 阶段 1.0 的 RR/LinUCB 请求级信号 smoke：证明四实例注册容量与昨日标定一致，LinUCB trace 能观察到 `compute_delta_norm`、`kv_ready_delta_norm`、exploit、exploration bonus、总分、选择实例和更新计数；验证 warmup 分布并进入 online-learning。

本实验不比较策略性能，不形成收敛或论文结论。Least-Inflight、模型保存/冻结、失败奖励分类、正式 workload 加权容量和并发 4/8 均不在本次范围内。

## 验收标准

- RR 与 LinUCB 各 50–100 请求；默认各 80，请求数大于 30-request warmup。
- 两组原始请求全部成功，无缺失 meta 或 trace warning。
- RR 使用四个安全候选且分布均衡。
- LinUCB warmup 恰为配置值、四实例分布均衡，之后出现 `selection_phase=linucb`。
- 注册兼容字段 `compute_capacity_ratio` 承载的孤立 Prefill 相对速度系数集合等于三轮聚合得到的 `0.6123,0.7720,0.9924,1.0077`；不得将其解释为吞吐容量或物理容量。
- 至少 90% 的 LinUCB 请求可见非零 compute 相对信号；共享 Redis/global scope 允许 KV 相对信号为零。
- online-learning 的总分等于 exploit + exploration bonus，最终选择最大总分候选。
- update 计数单调且每个成功请求均记录更新；无安全候选排除。
- 结果包绑定精确 commit、配置、环境指纹、标定聚合及两组原始 JSONL 的 SHA-256。
- 标定聚合必须报告真实 HTTP 启动偏差的 median/P95/max，且 P95 不超过预先固定的 5 ms；同时报告各长度、各实例三轮原始 TTFT median 及 CV。

机器验收命令由统一入口自动调用；`validation.json.status` 必须为 `passed`。

## 服务器执行

服务器只 checkout 和运行，不编辑代码。将 `<SMOKE_COMMIT>` 替换为 PR 中给出的完整 40 位 SHA；禁止使用“最新分支状态”。

```bash
cd /llm-stack/CacheRoute-codexysh
git fetch origin experiment/linucb-signal-smoke-20260722
git checkout --detach <SMOKE_COMMIT>
test "$(git rev-parse HEAD)" = "<SMOKE_COMMIT>"
test -z "$(git status --porcelain)"
EXPECTED_COMMIT=<SMOKE_COMMIT> REQUIRE_CLEAN_WORKTREE=1 \
  bash scripts/verify_source_sync.sh
bash -n scripts/start_rl4_signal_smoke.sh
bash -n scripts/run_rl4_signal_smoke.sh
```

先把昨日三轮结果目录写成机器可读聚合。三个目录必须分别包含 `summary.json` 与 `requests.jsonl`：

```bash
python3 scripts/aggregate_prefill_calibration.py \
  /path/to/prefill-r1 \
  /path/to/prefill-r2 \
  /path/to/prefill-r3 \
  --max-start-skew-p95-ms 5 \
  --output /path/to/prefill-aggregate-20260721.json
```

再运行统一 smoke 入口：

```bash
EXPECTED_COMMIT=<SMOKE_COMMIT> \
PROJECT_HOST="$PWD" \
MODEL_DIR=/workspace/llm-stack/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
MODEL_NAME=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
CONTAINER=cr0720-rl \
REDIS_CONTAINER=lmcache-redis \
CALIBRATION_AGGREGATE=/path/to/prefill-aggregate-20260721.json \
RUN_ID=signal-smoke-20260722-a1 \
bash scripts/run_rl4_signal_smoke.sh
```

入口会输出结果目录和 `.tar.gz` 路径。失败时也会生成包含 `exit_code`、已有原始文件和 SHA-256 的部分结果包。

## 两次失败熔断

同一问题首次失败后只做诊断并保留 `a1` 结果包。只有针对明确、非代码环境原因完成恢复后，才允许用新 `RUN_ID=signal-smoke-20260722-a2` 重试一次。同一问题第二次失败后立即停止，不继续补丁或重试：阻塞项报告给负责人，非阻塞项写入 backlog。

收到服务器结果包后先运行诊断，不能直接大范围改代码：

```bash
tar -xzf signal-smoke-20260722-a1.tar.gz
python3 -m json.tool signal-smoke-20260722-a1/validation.json
(cd signal-smoke-20260722-a1 && sha256sum -c SHA256SUMS)
```

## 同步回本机（Windows PowerShell）

```powershell
.\scripts\sync_rl4_result_bundle.ps1 `
  -Server user@linux-server `
  -RemoteArchive /absolute/server/path/signal-smoke-20260722-a1.tar.gz `
  -Destination C:\experiment-results\rl4
```

本机只接收并校验结果包，不把结果写回服务器代码工作区。
