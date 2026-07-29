# 当前分支与上游 KDN / LMCache 差异审计（阶段 A）

## 固定对比对象

| 项目 | 提交 |
| --- | --- |
| 当前实验分支 | `sihyang135-max/CacheRoute-codexysh@182f038cca9a5a153757c1537c2839693ade633a` |
| 上游 main | `AstraNetLab/CacheRoute@ad087d1c9ffc015a8dc7b29c53a00a77b9392f4b` |
| 共同祖先 | `7894ecf7f2d2c01a72272e0a156311b15e645d16` |

该上游提交已保存为本地只读审计引用 `refs/codex/upstream-main`。本分析仅比较源码树；不代表两边任一部署已运行。

## 当前分支已保留、应先验证而非重写的能力

| 能力 | 当前证据 |
| --- | --- |
| 独立 KDN HTTP 服务 | `kdn_server/kdn_api.py` 有独立 FastAPI 路由，包括 text 注册、KV 构建、搜索、注入与状态接口 |
| 独立数据目录 | `kdn_server/text_database/`、`kdn_server/KV_database/`、`text_db.py`、`kv_builder.py`、`kv_injector.py` |
| Scheduler 注册生命周期 | `kdn_server/sclient/scheduler_client.py` 与 KDN startup/shutdown 调用 register、heartbeat、unregister |
| 网络队列状态 | `pending_transfers`、`active_transfers`、`network_queue_ms_ema` 由 KDN heartbeat 传给 `scheduler/resource/control_plane.py`，并进入 KDNPool |
| Redis 地址重写 | `KDN_REDIS_REWRITE_ENABLE`、`KDN_FORCE_REDIS_HOST`、`KDN_REWRITE_LOOPBACK_TO` |
| 当前实验扩展 | `proxy/strategy/linucb.py`、四实例启动/健康检查/策略矩阵脚本、Prometheus KV usage 采集、Proxy 传输 resident 跟踪 |

结论：当前分支不需要“把 KDN 拆出去”。阶段 E 应验证并配置已有能力。

## 上游存在、当前分支缺失的现代 v1 能力

`git diff --name-status refs/codex/upstream-main..HEAD` 显示以下上游文件在当前分支不存在：

| 上游文件/区域 | 用途 | 当前处理原则 |
| --- | --- | --- |
| `scripts/validate_v1_kdn_roundtrip.py` | build → inspect → inject → consume 的 v1 roundtrip 验证 | 复用验证目标和证据模型；不得直接用于 legacy runtime |
| `doc/quickstart_v1.md`、`doc/runtime_compatibility_v1.md`、`doc/v1_migration_closeout.md` | v1 部署与兼容性边界 | 作为阶段 C 阅读材料，不是合并指令 |
| `env/docker/cu130/scripts/activate_v1.sh`、`check_v1_environment.py`、`start_lmcache_mp.sh`、`start_vllm_mp.sh` | CUDA 13、LMCache MP、vLLM v1 启动路径 | 只在独立 spike 证明环境可行时选择性移植 |
| `cacheroute_compat/`、`core/runtime_compat.py` | legacy/v1 profile 与 key-layout 兼容 | 先比较当前 KV builder、key hash 与 Connector 需求；不整体覆盖 |
| `kdn_server/contracts/`、`domain/`、`gateway/` | 版本化 KDN 服务契约 | 不属于阶段 A 的必要迁移；当前 FastAPI KDN 保持不动 |

## 当前分支的上游外实验改动

相对上游，当前分支新增或深度修改了：

- Proxy instance-only LinUCB 及 reward 更新逻辑；
- 4 Instance Docker/容器启动、health check、policy matrix、warmup 与实验 JSONL 分析工具；
- Proxy Prometheus KV-usage cache、实例控制面和 QueueManager 的 resident/传输状态；
- Scheduler、Proxy、KDN 中用于实验的状态、日志和接口调整；
- `env/README_RL_5090x8.md` 与实验 source manifest。

因此以下操作在未通过阶段 C 前均被禁止：直接 merge 上游 main、整体覆盖 `kdn_server/` / `proxy/` / `scheduler/`、直接替换 Docker 运行时、把 v1 启动参数加入 legacy 服务栈。

## 缺口与最小迁移决策表

| 缺口 | 是否阻塞下一步 | 最小动作 | 验收 |
| --- | --- | --- | --- |
| 实际 vLLM/LMCache/Connector 版本未知 | 阻塞阶段 C | 在服务器收集 package/image/env 证据 | 与文档和脚本一致或明确差异 |
| 官方 lookup / matched-token 指标未知 | 阻塞阶段 F/G | legacy 与 v1 单实例 spike | 请求前状态和请求后消费可对照 |
| 按请求 TEXT bypass 未证实 | 阻塞阶段 G | 在目标 runtime 中做受控 cache 清理与 metrics 实验 | TEXT 不发生不允许的缓存消费 |
| 当前没有 v1 roundtrip validator | 不阻塞 A/B，阻塞 v1 正式采用 | 先参考上游验证阶段；必要时最小适配 | build/inspect/inject/consume 均有真实证据 |
| KDN 跨机网络尚未部署 | 不阻塞 A/B | 保持 Redis 在推理节点，后续配置真实 IP | KDN 网卡传输 + Redis 写入 + vLLM 消费 |

## 阶段 A 结论

当前代码适合继续做阶段 B 的数据盘点；阶段 C 前必须获得服务器运行时证据。上游代码是候选能力来源，而不是可直接覆盖当前实验分支的版本。
