# LMCache 能力矩阵（阶段 A）

状态说明：**已确认**仅指源码中存在；**未验证**表示没有服务器运行时证据，不能用于正式实验结论。

| 能力 | 当前分支（`182f038`） | 上游现代 v1（`ad087d1`） | 当前结论与后续 Gate |
| --- | --- | --- | --- |
| 运行时路线 | **已确认 legacy 配置**：`LMCACHE_CONFIG_FILE` + YAML `remote_url` | 上游包含 v1 运行时兼容层、MP 启动脚本和文档 | 先做独立 spike；不得混用 |
| vLLM/LMCache 版本 | 文档目标为 vLLM 0.13.x、LMCache 0.3.x；具体表述有 0.3.11/0.3.12 不一致 | 上游 v1 文档/脚本为独立路线 | 实际 `pip show` 是 Gate A 未完成项 |
| 真实请求前 lookup | **未发现**公开 LMCache lookup API 调用 | 上游 roundtrip validator 可验证 metrics | 不能用 QueueManager resident 集代替官方 lookup |
| matched tokens | **未发现**从 LMCache/vLLM metrics 读取的实现 | 上游 validator 使用 lookup / prefetch 证据 | 需在阶段 C 的 spike 中确认可观测指标 |
| L1/L2 区分 | `PrometheusCache` 仅采集 vLLM KV 使用率；QueueManager resident 是 ack 驱动近似状态 | 上游 v1 路线有 LMCache MP / L1/L2 指标与消费验证资料 | 当前不能证明指定实例的完整本地命中 |
| request-level TEXT bypass | `Injection_type=text` 改变知识注入路径，但未发现 LMCache 官方按请求禁读开关 | 待由目标 runtime 能力决定 | 阶段 G 前必须实测；不能假定 text 即无缓存命中 |
| KDN 独立服务 | **已确认**：`kdn_server/kdn_api.py` 独立 FastAPI，含 KV/text 数据目录 | 上游继续发展为 versioned KDN gateway/contracts | 当前先验证现有服务，不整体迁移 |
| KDN Scheduler 注册/心跳 | **已确认**：register/heartbeat/unregister；包含 pending/active/queue EMA | 上游也有该基础，并新增契约层 | 适合直接用于状态候选；需在双机实测 |
| Redis 跨机地址 | **已确认**：支持 loopback rewrite 或 force host | 上游也有相关网络调试基础 | 正式配置应显式真实 IP；rewrite 仅调试 |
| KDN→Redis 传输证据 | 当前可获得注入 ack 与本地传输状态 | 上游 validator 包含 build/inspect/inject/consume 思路 | 尚不能证明 LMCache/vLLM 消费；需补运行时 metrics |
| roundtrip validator | 当前分支 **缺失** `scripts/validate_v1_kdn_roundtrip.py` | 上游存在，约 505 行，提供 build/inspect/inject/consume 分阶段校验 | 仅可选择性适配验证思想；不得直接复制到 legacy runtime |
| multi-instance 状态 | 当前有 InstancePool、Prometheus KV usage 和可配置 resident scope | v1 需要单实例/多 MP server 方案实测 | 共享 Redis 不等于各实例 L1 可见性 |

## 当前路线选择建议

阶段 A 不能直接选择 modern v1，也不能把当前 legacy stack 直接当作最终答案。建议阶段 C 新建隔离分支并只启动一个 7B TP1 Instance，按以下顺序做真实验证：

1. 无缓存文本 prefill；
2. KDN build KV 并检查产物；
3. 注入 Redis；
4. 清理/重启本地 L1 或 GPU 状态；
5. 请求前 lookup（若 runtime 提供）；
6. 远端加载；
7. 用 LMCache 与 vLLM metrics 证明实际消费。

只有在每一项都有可保存的日志/metric 后，才比较 legacy 与 v1 的多实例扩展、bypass、L1/L2 区分和当前 KDN 兼容性。
