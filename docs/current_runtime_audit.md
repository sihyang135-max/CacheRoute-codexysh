# 当前运行时审计（阶段 A）

审计日期：2026-07-29
本地仓库提交：`182f038cca9a5a153757c1537c2839693ade633a`（`main`）
上游对比提交：`AstraNetLab/CacheRoute@ad087d1c9ffc015a8dc7b29c53a00a77b9392f4b`

## 结论

当前分支的可确认运行时路线是 **legacy LMCache**，而不是 LMCache MP：启动资料和四实例脚本均通过 `LMCACHE_CONFIG_FILE` 读取 YAML，并使用 `remote_url: redis://…`。当前代码中没有 LMCache MP Server、`LMCacheMPConnector` 或请求前真实 LMCache Lookup API 的实现。

这不是对远程实验服务器的版本断言。当前 Windows 审计机没有 `python`、Docker CLI 或可用于 7B 实验的 GPU，因此不能声称已验证服务器实际运行版本、容器镜像、Redis namespace、Connector 实例或真实缓存命中。下列“源码证据”与“服务器实测缺口”必须严格区分。

## 固定基线

| 项目 | 证据 | 结论 |
| --- | --- | --- |
| 当前提交 | `git log -1` | `182f038`，主题为异步 KDN index refresh 修复 |
| 分支 | `git status --branch` | `main...origin/main` |
| 上游基线 | `git ls-remote` 后抓取到 `refs/codex/upstream-main` | `ad087d1` |
| 共同祖先 | `git merge-base HEAD refs/codex/upstream-main` | `7894ecf` |
| 工作区差异 | `.gitattributes` 要求 LF，但历史中部分受管文件为 CRLF | 当前显示的既有修改经 `git diff --ignore-space-at-eol` 证实仅为换行符，不是本阶段产生的代码改动 |

## 源码可确认的运行时路线

### Legacy LMCache 配置

- `env/config/lmcache_with_redis.yaml` 使用 `remote_url: "redis://127.0.0.1:6379"` 与 `remote_serde: "cachegen"`。
- `README.md` 的启动说明导出 `LMCACHE_CONFIG_FILE`，并将环境描述为 vLLM `0.13` + LMCache `3.11`。
- `env/README.md` 记录的源码安装目标为 vLLM `0.13` + LMCache `0.3.12`；镜像名为 `cacheroute:vllm0.13-lmcache3.11-pytorch2.9.1`。文档中的 `3.11` 与 `0.3.12` 表述不一致，不能替代 `pip show` 或镜像 digest。
- `scripts/start_rl_4instance_docker.sh` 与 `scripts/start_rl_4instance_in_container.sh` 默认传入同一 legacy 配置文件。

### 当前 Connector 边界

- `instance/instance_api.py` 只通过 `VLLM_BASE_URL` 调用 vLLM 的 OpenAI 兼容 `/v1/chat/completions` 和 `/v1/completions`；没有直接导入 LMCache Connector。
- LMCache 是否加载远端 KV 由 vLLM 进程的 `LMCACHE_CONFIG_FILE` 决定，而非由 CacheRoute 的一个可审计 Connector 类决定。
- `proxy/metrics/prometheus_cache.py` 只解析 vLLM KV 使用率 gauge（`kv_usage`），不解析 lookup、matched tokens、L1 hit 或 L2 prefetch 指标。
- 因此当前代码不能在请求前给出官方的“该实例是否完整命中该知识 KV”的证据；`QueueManager` 的 resident 集是由 CacheRoute 已观察到的传输 ack 维护的近似状态，不能等同于 LMCache 的真实 L1/L2 查表。

## 本机检查结果（不是服务器结果）

| 检查 | 结果 | 含义 |
| --- | --- | --- |
| `python --version` | 命令不可用 | 无法运行项目测试或读取本机 Python 包版本 |
| `docker version` | 命令不可用 | 无法验证容器、image ID 或容器内运行时 |
| `nvidia-smi` | `GeForce MX250`, driver `441.12` | 不具备本项目 7B 单卡/多实例实验条件 |

## 仍需在实验服务器执行的只读采样

在任何运行时迁移或业务代码修改前，收集并保存以下输出到运行记录：

```bash
git rev-parse HEAD
git status --short
docker image inspect <实际镜像> --format '{{.Id}}'
docker exec <容器> python3 -m pip show vllm lmcache torch redis
docker exec <容器> env | grep -E 'LMCACHE|VLLM|REDIS|KDN|SCHEDULER|PROXY'
docker exec <容器> python3 -c 'import vllm, lmcache; print(vllm.__version__); print(lmcache.__version__)'
redis-cli -h <redis-host> -p <redis-port> INFO keyspace
```

同时保存每个 Instance 的 `VLLM_METRICS_URL` 原始 metrics 样本，以及 KDN `/health`、Scheduler `/debug/status`、Proxy/Instance 控制面状态。缺少这些实测证据时，阶段 A 只能判定为“源码审计完成、运行时实测待补”。

## 阶段 A Gate 判定

- 源码、上游和本机可见运行时审计：**完成**。
- 远程服务器的真实版本、镜像、Redis 与 metrics 证据：**未完成，需人工提供服务器访问或命令输出**。
- 运行时路线选择（阶段 C）：**不得开始**。
