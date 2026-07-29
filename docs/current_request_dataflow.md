# 当前请求、KV 注入与 TTFT 数据流（阶段 A）

本文描述 `182f038` 中实际能从源码追踪到的链路；它不是对未运行服务的运行时证明。

## 请求主链路

```text
Client / perf_client
  │  OpenAI 请求 + RAG + Injection_type
  ▼
Scheduler :7001
  │  SchedulerRequest.build_request()
  │  从 ProxyPool / KDNPool 选择 Proxy、KDN，写入 Task
  ▼
Proxy :900x
  │  select_instance()
  │  可选 IWS：只改写 Injection_type（text / kvcache）
  │  QueueManager：准备知识、执行 KDN 注入、排队并记录 trace
  ▼
Instance :1900x
  │  转发 OpenAI body
  ▼
vLLM :1800x
  │  由 legacy LMCache YAML 决定本地/Redis KV 行为
  ▼
首 token SSE → Proxy trace → Client / perf_client 统计
```

## 调度与资源控制链路

```text
KDN :9101 ── register / heartbeat / unregister ──► Scheduler control plane :7002
                                                   │
                                                   ├── KDNPool: items, qps_1m,
                                                   │   pending_transfers, active_transfers,
                                                   │   network_queue_ms_ema
                                                   ▼
Scheduler request path ──► 选 KDN + Proxy ──► Proxy

Instance ── register / heartbeat / unregister ──► Proxy control plane
                                                  │
                                                  ▼
                                      InstancePool + Prometheus KV-usage snapshot
```

源码位置：

- Scheduler 请求入口与 KDNPool 快照：`scheduler/scheduler.py`。
- Scheduler KDN 控制面：`scheduler/resource/control_plane.py`。
- KDN 注册客户端：`kdn_server/sclient/scheduler_client.py`。
- KDN 运行时注册与心跳循环：`kdn_server/kdn_api.py`。
- Proxy Instance 选择：`proxy/proxy.py::select_instance`。

## 三种当前实际路径

| 选择值 | 当前代码行为 | 可记录证据 | 不能保证的事项 |
| --- | --- | --- | --- |
| `text` | `QueueManager` 将检索到的上下文写入 OpenAI 请求体 | `text_actual_path=text_inject` | 不能仅凭该字段证明 LMCache 对同一前缀被绕过；必须用服务器 metrics 或独立清缓存实测 |
| `kvcache` | KDN 为已就绪知识调用 `/knowledge/inject_ready_kv`，随后请求送入 vLLM | `kvcache_actual_path=kv_inject`、注入 ack、传输计时、`actual_vllm_internal_ms` | 不能仅凭注入 ack 证明 LMCache 从 Redis 加载且 vLLM 使用了对应 KV |
| fallback | KV 未就绪或注入失败时回落为文本 | `no_kv_ready_fallback_text` 或 `kv_inject_failed_fallback_text` | 当前 trace 不把它提升为独立、严格的 `actual_mode` 契约 |

相关源码位置：`proxy/queue/manager.py` 的准备阶段和 `proxy/queue/knowledge.py::inject_rag_into_instance_body`。

## KDN → Redis 注入链路

1. KDN 保存文本及 `KV_database/<kid>` 产物。
2. Proxy 对已就绪 kid 调用 KDN 的 `/knowledge/inject_ready_kv`。
3. KDN 的网络模拟器可串行化传输，并通过 heartbeat 上报 `pending_transfers`、`active_transfers`、`network_queue_ms_ema`。
4. KDN 使用请求中的 Redis host；启用 `KDN_REDIS_REWRITE_ENABLE` 时，`KDN_FORCE_REDIS_HOST` 可强制覆盖，或 `KDN_REWRITE_LOOPBACK_TO` 仅改写 loopback 地址。
5. Redis 写入完成后，Proxy 记录 ack 与本地 resident 状态；Instance 再将请求转给 vLLM。

对双机实验，这段链路只有在 KDN 的 `request_host`/`resolved_host`、网卡流量、Redis 写入和 vLLM 的加载证据同时出现时才能认定为真正远端 KV reuse。

## 当前 TTFT 时间点

当前 trace 已可区分部分阶段：

```text
proxy_enqueue_ms
  → 知识准备 / KDN 注入 / ready 队列
  → forward_start_ms
  → first_token_ms
```

- `actual_know_prepare_ms`：Proxy 侧知识准备相关阶段。
- `actual_ready_queue_ms`：ready enqueue 到下游 forward 的等待。
- `actual_vllm_internal_ms = first_token_ms - forward_start_ms`：包含下游 vLLM 队列与 prefill，不能直接当作纯 prefill。
- LinUCB 当前 reward 使用 `first_token_ms - proxy_enqueue_ms`。

因此当前代码可以记录端到端 Proxy TTFT，但还没有把 “KDN→Redis 网络传输”、“LMCache L2 load”、“残余 prefill” 通过官方 runtime counter 独立证明。

## 当前 LinUCB 实际动作空间

`proxy/strategy/linucb.py` 是按 **instance_id** 建臂的 disjoint LinUCB。其请求上下文只含 `bias`、`compute_delta_norm` 和 `kv_ready_delta_norm`；`proxy/proxy.py` 仅在 streaming 且 `Injection_type == kvcache` 时调用它。

因此当前实现是“给定 kvcache 模式下的实例选择”，不是 Issue 所要求的 `(instance, mode)` 联合动作。Proxy 的 IWS 会在之后根据预测成本改写 `Injection_type`，两者目前是两套串行机制。
