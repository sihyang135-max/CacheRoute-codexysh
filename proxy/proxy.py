"""
Proxy_v1.py
---------
作为 Scheduler 的“下游代理”示例：

- 异步接收 Scheduler 转发的 Request payload（JSON）
- 简单解析其中的关键信息（Request_ID、Prompt、Service、Task 等）,还原为内部 Request 结构
- 基于 Request 中的信息，构造“OpenAI 风格”的 HTTP 请求体
- 调用下游 Instance
    * /v1/chat/completions  -> 流式 text/event-stream
    * /v1/completions       -> 非流式 JSON
- 将 Instance 的响应透传回 Scheduler（chat 为流式，completions 为一次性 JSON）

后续你可以在这里接入真正的 vLLM / OpenAI / 其它后端服务。
"""
from __future__ import annotations

import os
import json
import asyncio
import logging
import time
import uvicorn

from contextlib import asynccontextmanager
from dataclasses import fields
from typing import Any, Dict, List, Tuple, AsyncGenerator, Optional

from fastapi import FastAPI, Request as FastAPIRequest
from fastapi.responses import JSONResponse, StreamingResponse

from core import Request as SchedulerRequest, Prompt, Service, Task
# from core.config import INSTANCE_BASE_URL
from core import forward_request
from core import config

from proxy.sclient.scheduler_client import SchedulerControlClient
from proxy.resource.instance_pool import InstancePool
from proxy.resource import p_control_plane
from proxy.resource.hb_log import HeartbeatReporter, hb_report_loop
from proxy.strategy.factory import build_instance_strategy
from proxy.strategy.linucb import (
    LinUCBStrategy,
    Decision,
    classify_feedback,
    ttft_reward,
)
from proxy.strategy.candidate_filter import filter_safe_instances
from proxy.queue import QueueManager, ProxyTask
from proxy.metrics.prometheus_cache import PrometheusCache

SCHEDULER_CP_URL = os.environ.get("SCHEDULER_CP_URL", config.SCHEDULER_CP_URL).rstrip("/")
# KDN_BASE_URL = os.environ.get("KDN_BASE_URL", config.KDN_BASE_URL).rstrip("/")

# PROXY_PORT = int(os.environ.get("PROXY_PORT", "8002"))
PROXY_ADVERTISE_HOST = os.environ.get("PROXY_ADVERTISE_HOST", config.PROXY_DP_HOST)
PROXY_ADVERTISE_PORT = int(os.environ.get("PROXY_ADVERTISE_PORT", str(config.PROXY_DP_PORT)))
PROXY_ID = os.environ.get("PROXY_ID", f"hp_{PROXY_ADVERTISE_HOST}:{PROXY_ADVERTISE_PORT}")
PROXY_HEARTBEAT_S = float(os.environ.get("PROXY_HEARTBEAT_S", config.HEARTBEAT_INTERVAL_S))
PROXY_MAX_CAPACITY = int(os.environ.get("PROXY_MAX_CAPACITY", config.PROXY_MAX_CAPACITY))
PROXY_INSTANCE_COUNT = int(os.environ.get("PROXY_INSTANCE_COUNT", config.PROXY_INSTANCE_COUNT))
PROXY_KV_MEM_PER_INSTANCE_GB = float(os.environ.get("PROXY_KV_MEM_PER_INSTANCE_GB", config.PROXY_KV_MEM_PER_INSTANCE_GB))
PROXY_KV_CACHE_UPDATE_POLICY = os.environ.get("PROXY_KV_CACHE_UPDATE_POLICY", config.PROXY_KV_CACHE_UPDATE_POLICY)
PROXY_INJECTION_STRATEGY = os.environ.get("PROXY_INJECTION_STRATEGY", "default").strip().lower()
IWS_KDN_QUEUE_PENALTY_ALPHA = float(os.environ.get("IWS_KDN_QUEUE_PENALTY_ALPHA", "0.5"))
IWS_DECISION_MARGIN_MS = int(os.environ.get("IWS_DECISION_MARGIN_MS", "100"))
if PROXY_INJECTION_STRATEGY not in {"default", "iws"}:
    logger.warning(
        "[Proxy] invalid PROXY_INJECTION_STRATEGY=%s, fallback to default",
        PROXY_INJECTION_STRATEGY,
    )
    PROXY_INJECTION_STRATEGY = "default"

# NOTE:
# This is a TEMPORARY fallback for legacy request path.
# It MUST be removed once instance_pool-based routing is enabled.
INSTANCE_PORT = int(os.environ.get("INSTANCE_PORT", "9001"))

logger = logging.getLogger("proxy")
logging.basicConfig(level=logging.INFO)


def _load_static_kdn_links() -> Dict[str, Any]:
    raw_links = os.environ.get("PROXY_KDN_LINKS_JSON", "").strip()
    if not raw_links:
        return {}
    try:
        parsed = json.loads(raw_links)
        if isinstance(parsed, dict):
            return parsed
        logger.warning("[Proxy] PROXY_KDN_LINKS_JSON is not dict, ignored")
    except Exception as e:
        logger.warning("[Proxy] parse PROXY_KDN_LINKS_JSON failed: %s", e)
    return {}


async def _build_proxy_topology_meta() -> Dict[str, Any]:
    static_links = _load_static_kdn_links()
    dynamic_links = await p_control_plane.get_kdn_links_snapshot()
    merged_links = dict(static_links)
    merged_links.update(dynamic_links)
    return {"kdn_links": merged_links} if merged_links else {}


def _squelch_noisy_loggers():
    # http client
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    # uvicorn access log（可选，避免每次请求一行）
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    # 如果你用了 asyncio/anyio 也很吵，再加：
    # logging.getLogger("asyncio").setLevel(logging.WARNING)
    # logging.getLogger("anyio").setLevel(logging.WARNING)

# ======================= Proxy初始化 =======================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Proxy 生命周期：
      - startup: 向 scheduler(control plane) 注册
      - running: 周期心跳，保证 proxy_pool 不过期
      - shutdown: 优雅注销（非强依赖，kill -9 情况靠 TTL 清理）
    """
    _squelch_noisy_loggers()
    app.state.injection_strategy_name = PROXY_INJECTION_STRATEGY  # type: ignore
    logger.info("[Proxy] injection strategy=%s", app.state.injection_strategy_name)
    # --- 初始化实例池，并注入proxy控制平面 ---
    ttl_s = int(os.environ.get("PROXY_INSTANCE_TTL_S", config.INSTANCE_ALIVE_TTL_S))
    app.state.instance_pool = InstancePool(ttl_s=ttl_s)  # type: ignore
    p_control_plane.set_pool(app.state.instance_pool)  # type: ignore
    app.state.prometheus_cache = PrometheusCache()  # type: ignore
    app.state._metrics_stop = asyncio.Event()  # type: ignore

    # --- 加载proxy调度策略（业务面使用） ---
    strategy_name = os.environ.get("PROXY_INSTANCE_STRATEGY", "round_robin")
    try:
        app.state.instance_strategy = build_instance_strategy(strategy_name)  # type: ignore
        strategy = app.state.instance_strategy  # type: ignore
        if isinstance(strategy, LinUCBStrategy):
            if strategy.frozen and not config.PROXY_RL_MODEL_LOAD_PATH:
                raise ValueError("PROXY_RL_FROZEN=1 requires PROXY_RL_MODEL_LOAD_PATH")
            if config.PROXY_RL_MODEL_LOAD_PATH:
                strategy.load_model(config.PROXY_RL_MODEL_LOAD_PATH)
                logger.info(
                    "[Proxy][LinUCB] loaded model path=%s frozen=%s updates=%s",
                    config.PROXY_RL_MODEL_LOAD_PATH,
                    strategy.frozen,
                    strategy.effective_updates,
                )
        logger.info("[Proxy] instance strategy=%s", strategy_name)
    except Exception as e:
        # 策略初始化失败是致命的（否则业务面无法选择 instance）
        logger.error("[Proxy] invalid instance strategy=%s err=%s", strategy_name, str(e))
        raise

    # ---尝试启动proxy控制平面，用于与Instance交互来动态刷新Instance池 ---
    cp_host = os.environ.get("PROXY_CP_HOST", config.PROXY_CP_HOST)
    cp_port = int(os.environ.get("PROXY_CP_PORT", config.PROXY_CP_PORT))

    cp_config = uvicorn.Config(
        p_control_plane.control_plane,
        host=cp_host,
        port=cp_port,
        log_level="info",
        access_log=False,
        # 重要：不要启用 reload / workers，embedded 场景保持单进程单实例
    )
    cp_server = uvicorn.Server(cp_config)
    app.state._cp_server = cp_server  # type: ignore

    async def _run_cp():
        await cp_server.serve()

    app.state._cp_task = asyncio.create_task(_run_cp())  # type: ignore
    logger.info("[Proxy] control plane started: http://%s:%s", cp_host, cp_port)

    # --- 启用scheduler客户端，尝试与scheduler交互并注册、与scheduler保活 ---
    client = SchedulerControlClient(SCHEDULER_CP_URL, timeout_s=5.0)
    app.state._sched_client = client  # type: ignore
    app.state._proxy_id = PROXY_ID    # type: ignore
    app.state._hb_stop = asyncio.Event()  # type: ignore

    # --- 心跳日志聚合（输出层）---
    app.state._hb_reporter = HeartbeatReporter(interval_s=30.0)  # type: ignore
    app.state._hb_report_task = asyncio.create_task(  # type: ignore
        hb_report_loop(
            reporter=app.state._hb_reporter,  # type: ignore
            logger=logger,
            proxy_id=PROXY_ID,
            stop_event=app.state._hb_stop,  # type: ignore
        )
    )

    # 1) register（失败不应阻塞业务启动：允许 proxy 单独跑）
    try:
        # CacheRoute 第二阶段：
        # 可选注入 KDN->Proxy 静态拓扑信息，供 Scheduler 词典序策略使用。
        # 环境变量示例：
        # PROXY_KDN_LINKS_JSON='{"kdn_a":{"bandwidth_tier":3,"latency_tier":1}}'
        proxy_meta: Dict[str, Any] = {"version": "proxy_v1"}
        proxy_meta.update(await _build_proxy_topology_meta())

        reg = await client.register(
            proxy_id=PROXY_ID,
            host=PROXY_ADVERTISE_HOST,
            port=PROXY_ADVERTISE_PORT,
            endpoints=["chat/completions", "completions"],
            meta=proxy_meta,
            max_capacity=PROXY_MAX_CAPACITY,
            instance_count=PROXY_INSTANCE_COUNT,
            kv_mem_per_instance_gb=PROXY_KV_MEM_PER_INSTANCE_GB,
            kv_cache_update_policy=PROXY_KV_CACHE_UPDATE_POLICY,
        )
        # 用 scheduler 建议的心跳周期覆盖本地默认
        interval = float(reg.heartbeat_interval_s) if reg.heartbeat_interval_s else PROXY_HEARTBEAT_S
        app.state._hb_interval = interval  # type: ignore
        logger.info("[Proxy] registered to scheduler: cp=%s proxy_id=%s advertise=%s:%s hb=%ss",
                    SCHEDULER_CP_URL, reg.proxy_id, PROXY_ADVERTISE_HOST, PROXY_ADVERTISE_PORT, interval)
    except Exception as e:
        # 不阻塞业务面：注册失败时 proxy 仍可本地转发（只是 scheduler 看不到它）
        app.state._hb_interval = PROXY_HEARTBEAT_S  # type: ignore
        logger.warning("[Proxy] register failed (non-fatal): cp=%s err=%s", SCHEDULER_CP_URL, str(e))

    # 2) heartbeat loop
    async def _hb_loop():
        reporter: HeartbeatReporter = app.state._hb_reporter  # type: ignore
        while not app.state._hb_stop.is_set():  # type: ignore
            try:
                await client.heartbeat(
                    proxy_id=PROXY_ID,
                    meta_patch=await _build_proxy_topology_meta(),
                )
                await reporter.record(ok=True)
            except Exception as e:
                # 不逐条 warning，避免刷屏；只记录窗口统计
                await reporter.record(ok=False, err=str(e))
                # 真要立即看到异常堆栈：你可以改成 logger.debug(..., exc_info=True)
                logger.debug("[Proxy] heartbeat failed", exc_info=True)

            await asyncio.sleep(float(getattr(app.state, "_hb_interval", PROXY_HEARTBEAT_S)))  # type: ignore

    app.state._hb_task = asyncio.create_task(_hb_loop())  # type: ignore

    async def _metrics_loop() -> None:
        """Refresh metrics out of band; routing never blocks on Prometheus."""
        cache: PrometheusCache = app.state.prometheus_cache  # type: ignore
        stop: asyncio.Event = app.state._metrics_stop  # type: ignore
        while not stop.is_set():
            instances = app.state.instance_pool.list(include_dead=False)  # type: ignore
            jobs = []
            for item in instances:
                url = str((item.meta or {}).get("metrics_url") or "").strip()
                if url:
                    jobs.append(cache.refresh(item.instance_id, url, config.PROXY_RL_PROMETHEUS_TIMEOUT_S))
            if jobs:
                await asyncio.gather(*jobs, return_exceptions=True)
            try:
                await asyncio.wait_for(stop.wait(), timeout=config.PROXY_RL_PROMETHEUS_INTERVAL_S)
            except asyncio.TimeoutError:
                pass

    app.state._metrics_task = asyncio.create_task(_metrics_loop())  # type: ignore

    try:
        yield
    finally:
        strategy = getattr(app.state, "instance_strategy", None)  # type: ignore
        if (
            isinstance(strategy, LinUCBStrategy)
            and strategy.model_save_path
            and not strategy.frozen
        ):
            try:
                strategy.checkpoint_if_due(force=True)
            except Exception as exc:
                logger.error("[Proxy][LinUCB] final model save failed: %s", exc)
        # 关闭控制平面
        try:
            srv = getattr(app.state, "_cp_server", None)  # type: ignore
            t = getattr(app.state, "_cp_task", None)  # type: ignore
            if srv is not None:
                srv.should_exit = True
                srv.force_exit = True
            if t is not None:
                # 不要长时间 await；给它一个很短的机会退出即可
                try:
                    await asyncio.wait_for(t, timeout=2.0)
                except Exception:
                    t.cancel()
        except Exception:
            pass

        # 向scheduler汇报
        try:
            app.state._hb_stop.set()  # type: ignore
            task = getattr(app.state, "_hb_task", None)  # type: ignore
            if task:
                task.cancel()
            rpt = getattr(app.state, "_hb_report_task", None)  # type: ignore
            if rpt:
                rpt.cancel()
            app.state._metrics_stop.set()  # type: ignore
            metrics_task = getattr(app.state, "_metrics_task", None)  # type: ignore
            if metrics_task:
                metrics_task.cancel()
        except Exception:
            pass

        try:
            await client.unregister(proxy_id=PROXY_ID)
            logger.info("[Proxy] unregistered from scheduler: proxy_id=%s", PROXY_ID)
        except Exception as e:
            logger.warning("[Proxy] unregister failed (ignore): err=%s", str(e))

        try:
            await client.close()
        except Exception:
            pass

proxy = FastAPI(title="CacheRoute Proxy v1", lifespan=lifespan)
queue_mgr = QueueManager()              #创建任务队列管理器


#--------------------------------------------------------------
# ======================= 公共内部处理函数 =======================
#--------------------------------------------------------------

def _dataclass_from_dict(dc_cls, data: Dict[str, Any]):
    """
    安全地从 dict 构造 dataclass：
      - 只取 dataclass 中定义过的字段，避免因为多余字段报错
      - 必填字段如果缺失，会抛 TypeError，说明上游传的结构不对
    """
    if data is None:
        data = {}
    field_names = {f.name for f in fields(dc_cls)}
    filtered = {k: v for k, v in data.items() if k in field_names}
    return dc_cls(**filtered)


def recover_request_from_payload(payload: Dict[str, Any]) -> SchedulerRequest:
    """
        将 Scheduler 发送来的 JSON payload 恢复成 Request / Prompt / Service / Task 三个 dataclass。
    """
    req_id = payload.get("Request_ID", 0)
    req_type = payload.get("Request_type", "request")

    prompt_dict = payload.get("Prompt") or {}
    service_dict = payload.get("Service") or {}
    task_dict = payload.get("Task") or {}

    prompt_obj = _dataclass_from_dict(Prompt, prompt_dict)
    service_obj = _dataclass_from_dict(Service, service_dict)
    task_obj = _dataclass_from_dict(Task, task_dict)

    req_obj = SchedulerRequest(
        Request_ID=req_id,
        Request_type=req_type,
        Prompt=prompt_obj,
        Service=service_obj,
        Task=task_obj,
    )
    logger.info(
        "[Proxy] 恢复 Request 成功: Request_ID=%s, Endpoint_type=%s, model=%s",
        req_obj.Request_ID,
        getattr(req_obj.Service, "Endpoint_type", None),
        req_obj.Prompt.model,
    )
    return req_obj


def build_body_for_instance(req_obj: SchedulerRequest, mode: str) -> Dict[str, Any]:
    """
        根据 Request 构造发给 Instance 的 OpenAI 风格 body：
          - mode="chat"        -> /v1/chat/completions
          - mode="completions" -> /v1/completions
    """
    prompt = req_obj.Prompt
    model = prompt.model
    user_prompt = prompt.user_prompt
    max_tokens = getattr(prompt, "max_tokens", None)
    temperature = getattr(prompt, "temperature", None)
    top_p = getattr(prompt, "top_p", None)
    stream = getattr(prompt, "stream", False)
    # print(f"[Proxy]stream={stream}")

    if mode == "chat":
        # Instance 的 chat 接口按 OpenAI chat/completions 风格：
        # messages = [{role: "user", content: "..."}]
        body: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "user", "content": user_prompt},
            ],
            "stream": stream,
        }
        if stream:
            body["stream_options"] = {"include_usage": True}
    else:
        # completions：prompt + 非流式
        body = {
            "model": model,
            "prompt": user_prompt,
            "stream": False,
        }

        # 可选参数补上（有就带，没有就算了）
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    if temperature is not None:
        body["temperature"] = temperature
    if top_p is not None:
        body["top_p"] = top_p

    return body


def _instance_control_port(instance: Any) -> int:
    meta = getattr(instance, "meta", {}) or {}
    try:
        port = int(meta.get("control_port") or config.INSTANCE_CP_PORT)
    except (TypeError, ValueError):
        port = int(config.INSTANCE_CP_PORT)
    return port if port > 0 else int(config.INSTANCE_CP_PORT)


def build_cacheroute_meta(task: ProxyTask) -> Dict[str, Any]:
    return {
        "trace": task.trace,
        "kv_ack": task.kv_ack,
        "kv_ready_kids": task.kv_ready_kids,
        "text_only_kids": task.text_only_kids,
        "miss_kids": task.miss_kids,
        "error": task.error,
    }


def _sse_meta_event(task: ProxyTask) -> bytes:
    payload = build_cacheroute_meta(task)
    return (
        "event: cacheroute_meta\n"
        f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
    ).encode("utf-8")


def _linucb_reward(task: ProxyTask) -> float:
    first = task.trace.get("first_token_ms")
    enqueued = task.trace.get("proxy_enqueue_ms")
    if not isinstance(first, int) or not isinstance(enqueued, int):
        return -float(config.PROXY_RL_REWARD_CLIP)
    return ttft_reward(
        ttft_ms=max(0, first - enqueued),
        scale_ms=config.PROXY_RL_REWARD_TTFT_SCALE_MS,
        clip=config.PROXY_RL_REWARD_CLIP,
    )


def _update_linucb_once(task: ProxyTask) -> None:
    if task.trace.get("rl_update_decided"):
        return
    features = task.trace.get("rl_features")
    strategy = getattr(proxy.state, "instance_strategy", None)
    if not isinstance(strategy, LinUCBStrategy) or not isinstance(features, list):
        return
    task.trace["rl_update_decided"] = 1
    first = task.trace.get("first_token_ms")
    enqueued = task.trace.get("proxy_enqueue_ms")
    outcome, reward_source, should_update = classify_feedback(
        first,
        enqueued,
        task_error=str(task.error or ""),
        explicit_outcome=str(task.trace.get("outcome_class") or ""),
    )
    task.trace["outcome_class"] = outcome
    task.trace["reward_source"] = reward_source
    task.trace["rl_updated"] = 0
    if isinstance(first, int) and isinstance(enqueued, int):
        task.trace["rl_observed_ttft_ms"] = max(0, first - enqueued)
    task.trace["rl_slo_ttft_ms"] = int(
        getattr(getattr(task.req_obj, "Service", None), "SLO_TTFT", 0) or 0
    )
    task.trace["rl_reward_ttft_scale_ms"] = float(config.PROXY_RL_REWARD_TTFT_SCALE_MS)
    task.trace["rl_reward_clip"] = float(config.PROXY_RL_REWARD_CLIP)
    reward = _linucb_reward(task) if should_update else None
    if reward is not None:
        task.trace["rl_reward_milli"] = int(reward * 1000)
    if not should_update:
        task.trace["rl_update_reason"] = reward_source
        return
    updated = strategy.update(task.instance_id, features, float(reward))
    task.trace["rl_updated"] = 1 if updated else 0
    task.trace["rl_update_reason"] = "updated" if updated else "frozen"
    task.trace["rl_model_frozen"] = 1 if strategy.frozen else 0
    if updated:
        try:
            snapshot = strategy.checkpoint_if_due()
            if snapshot is not None:
                task.trace["rl_parameter_snapshot_updates"] = snapshot["effective_updates"]
        except Exception as exc:
            task.trace["rl_checkpoint_error"] = str(exc)
            logger.exception("[Proxy][LinUCB] checkpoint failed")
    task.trace["rl_effective_updates_after"] = strategy.effective_updates
    task.trace["rl_arm_updates_after"] = strategy.arm_updates


async def _wrap_chat_stream_with_meta(task: ProxyTask, queue_mgr: QueueManager) -> AsyncGenerator[bytes, None]:
    """
    转发下游 chat SSE，但把 [DONE] 延后，先插入一条 cacheroute_meta 事件。
    """
    pending = b""
    done_seen = False

    try:
        async for chunk in queue_mgr.iter_response(task):
            if not chunk:
                continue

            # Role-only SSE chunks are not tokens. Update only after QueueManager
            # has observed non-empty content/reasoning content.
            if isinstance(task.trace.get("first_token_ms"), int):
                _update_linucb_once(task)

            pending += chunk

            while b"\n" in pending:
                line, pending = pending.split(b"\n", 1)
                full_line = line + b"\n"

                if line.startswith(b"data:"):
                    data = line[len(b"data:"):].strip()
                    if data == b"[DONE]":
                        done_seen = True
                        continue

                yield full_line
    except asyncio.CancelledError:
        task.error = "client_cancelled"
        task.trace["outcome_class"] = "client_cancelled"
        task.trace["client_cancelled_ms"] = int(time.time() * 1000)
        raise
    except Exception as e:
        task.error = f"stream_wrap_failed: {e}"
        task.trace["stream_exception_ms"] = int(time.time() * 1000)
        logger.exception("[Proxy] stream wrapper failed rid=%s", task.request_id)
    finally:
        _update_linucb_once(task)
        if pending:
            yield pending
            pending = b""
        yield _sse_meta_event(task)
        yield b"data: [DONE]\n\n"


async def select_instance(app: FastAPI, req_obj: SchedulerRequest) -> Tuple[Any, Optional[Decision], Dict[str, Any]]:
    """
    业务面选择一个 instance。
    - 输入：当前存活实例列表（由 InstancePool 提供）
    - 输出：一个 InstanceInfo（至少有 host/port/instance_id）
    """
    pool = app.state.instance_pool  # type: ignore
    strategy = app.state.instance_strategy  # type: ignore

    instances = pool.list(include_dead=False)
    selection_trace: Dict[str, Any] = {
        "configured_instance_strategy": str(getattr(strategy, "name", type(strategy).__name__)),
        "alive_instance_count": len(instances),
    }
    if not instances:
        selection_trace.update({"selection_phase": "failed", "selection_reason": "no_alive_instances"})
        return None, None, selection_trace

    def local_inflight_hint() -> Dict[str, Dict[str, int]]:
        return {
            "instance_inflight": {
                item.instance_id: queue_mgr.get_route_reservation(item.instance_id)
                for item in instances
            }
        }

    def choose_and_reserve(
        applied_strategy: Any,
        phase: str,
        reason: str,
    ) -> Tuple[Any, None, Dict[str, Any]]:
        hint = (
            local_inflight_hint()
            if getattr(applied_strategy, "name", "") == "least_inflight"
            else req_obj
        )
        chosen = applied_strategy.select(instances, hint=hint)
        queue_mgr.reserve_route(chosen.instance_id)
        selection_trace.update({
            "applied_instance_strategy": str(
                getattr(applied_strategy, "name", type(applied_strategy).__name__)
            ),
            "selection_phase": phase,
            "selection_reason": reason,
            "selected_instance_id": chosen.instance_id,
            "route_reservation_after": queue_mgr.get_route_reservation(
                chosen.instance_id
            ),
        })
        return chosen, None, selection_trace

    try:
        cache: PrometheusCache = app.state.prometheus_cache  # type: ignore
        safe_instances, excluded_instances = filter_safe_instances(
            instances,
            metric_for=cache.get,
            queue_snapshot_for=queue_mgr.get_instance_rl_snapshot,
            metric_stale_s=config.PROXY_RL_PROMETHEUS_STALE_S,
            metric_failure_limit=config.PROXY_RL_PROMETHEUS_FAILURE_LIMIT,
            kv_usage_limit=config.PROXY_RL_KV_USAGE_LIMIT,
        )
        selection_trace.update({
            "safe_instance_count": len(safe_instances),
            "excluded_instances": excluded_instances,
            # Preserve the existing keys consumed by experiment analysis.
            "linucb_safe_instance_count": len(safe_instances),
            "linucb_excluded_instances": excluded_instances,
        })
        if not safe_instances:
            selection_trace.update({
                "selection_phase": "failed",
                "selection_reason": "no_safe_instances",
            })
            return None, None, selection_trace

        # Every policy and LinUCB fallback receives this same ordered list.
        # No later branch may restore the broader alive list.
        instances = safe_instances

        if not config.PROXY_RL_ENABLED:
            applied_strategy = (
                build_instance_strategy("least_inflight")
                if isinstance(strategy, LinUCBStrategy)
                else strategy
            )
            return choose_and_reserve(
                applied_strategy,
                "fallback" if isinstance(strategy, LinUCBStrategy) else "baseline",
                "linucb_disabled"
                if isinstance(strategy, LinUCBStrategy)
                else "configured_non_linucb_strategy",
            )
        if not isinstance(strategy, LinUCBStrategy):
            return choose_and_reserve(
                strategy,
                "baseline",
                "configured_non_linucb_strategy",
            )

        service = req_obj.Service
        prompt = req_obj.Prompt
        injection_type = str(
            getattr(service, "Injection_type", "") or ""
        ).strip().lower()
        if not bool(getattr(prompt, "stream", False)):
            return choose_and_reserve(
                build_instance_strategy("least_inflight"),
                "fallback",
                "linucb_requires_streaming",
            )
        if injection_type != "kvcache":
            return choose_and_reserve(
                build_instance_strategy("least_inflight"),
                "fallback",
                "linucb_requires_kvcache",
            )

        context_started_ns = time.perf_counter_ns()
        kdn_addr = str(getattr(req_obj.Task, "KDN_server_addr", "") or "")
        knowledge_len = max(0, int(getattr(service, "Knowledge_length", 0) or 0))
        knowledge_list = [
            str(kid) for kid in (getattr(service, "Knowledge_List", []) or [])
        ]
        kv_len = (knowledge_len // 256) * 256
        kv_size_mb = kv_len * config.PROXY_RL_DEFAULT_KV_MB_PER_TOKEN
        kvcache_service = queue_mgr.estimate_kvcache_service_ms(req_obj)
        residual_prefill_ms = max(
            0.0,
            float(kvcache_service.get("residual_prefill_ms", 0) or 0),
        )
        contexts: Dict[str, Dict[str, Any]] = {}
        for item in instances:
            meta = item.meta or {}
            compute_capacity = max(
                1e-6,
                float(meta.get("prefill_capacity_ratio", 1.0) or 1.0),
            )
            compute_wait_ms = queue_mgr.get_compute_wait_snapshot_ms(item.instance_id)
            compute_cost_ms = (
                float(compute_wait_ms) + residual_prefill_ms
            ) / compute_capacity

            delivery = queue_mgr.get_kv_delivery_snapshot(
                item.instance_id,
                kdn_addr,
                knowledge_list,
            )
            link = p_control_plane.get_instance_kdn_link_snapshot(
                item.instance_id,
                kdn_addr,
            )
            observed_mbps = float(delivery["observed_delivery_mbps"])
            reported_mbps = float(
                link.get("bandwidth_mbps", config.INSTANCE_DEFAULT_LINK_BW_MBPS)
                or config.INSTANCE_DEFAULT_LINK_BW_MBPS
            )
            bw_mbps = max(
                1.0,
                min(observed_mbps, reported_mbps)
                if observed_mbps > 0.0
                else reported_mbps,
            )
            latency_ms = max(
                0.0,
                float(link.get("latency_ms", link.get("rtt_ms", 0.0)) or 0.0),
            )
            missing_ratio = min(1.0, max(0.0, float(delivery["missing_ratio"])))
            transfer_ms = (
                kv_size_mb * missing_ratio * 8.0 * 1000.0 / bw_mbps
                if kv_size_mb > 0.0 and missing_ratio > 0.0
                else 0.0
            )
            pending_transfers = max(0.0, float(delivery["pending_transfers"]))
            kv_ready_cost_ms = (
                latency_ms + transfer_ms * (1.0 + pending_transfers)
                if transfer_ms > 0.0
                else 0.0
            )
            contexts[item.instance_id] = {
                "compute_cost_ms": compute_cost_ms,
                "kv_ready_cost_ms": kv_ready_cost_ms,
                "compute_wait_ms": float(compute_wait_ms),
                "residual_prefill_ms": residual_prefill_ms,
                "compute_capacity_ratio": compute_capacity,
                "missing_kv_ratio": missing_ratio,
                "pending_kv_transfers": pending_transfers,
                "kv_transfer_ms": transfer_ms,
                "kv_bandwidth_mbps": bw_mbps,
                "kv_bandwidth_source": (
                    "delivery_ewma+topology_headroom"
                    if observed_mbps > 0.0
                    else "topology_headroom"
                ),
                "kv_latency_ms": latency_ms,
            }
        context_build_us = (time.perf_counter_ns() - context_started_ns) // 1000
        selection_trace.update({
            "rl_context_build_us": int(context_build_us),
        })

        score_started_ns = time.perf_counter_ns()
        decision = strategy.choose(instances, contexts)
        score_us = (time.perf_counter_ns() - score_started_ns) // 1000
        chosen = next(item for item in instances if item.instance_id == decision.instance_id)
        queue_mgr.reserve_route(chosen.instance_id)
        selection_trace.update({
            "applied_instance_strategy": "linucb",
            "selection_phase": decision.phase,
            "selection_reason": "linucb_decision",
            "selected_instance_id": chosen.instance_id,
            "rl_effective_updates": decision.effective_updates,
            "rl_effective_selections": strategy.effective_selections,
            "rl_arm_updates": strategy.arm_updates,
            "rl_arm_selections": strategy.arm_selections,
            "rl_feature_names": strategy.feature_names,
            "rl_candidate_scores": decision.candidate_scores,
            "rl_candidate_exploit_scores": decision.candidate_exploit_scores,
            "rl_candidate_exploration_bonuses": decision.candidate_exploration_bonuses,
            "rl_candidate_features": decision.candidate_features,
            "rl_candidate_costs": contexts,
            "rl_bandit_score_us": int(score_us),
            "route_reservation_after": queue_mgr.get_route_reservation(
                chosen.instance_id
            ),
        })
        return chosen, decision, selection_trace
    except Exception as e:
        logger.warning("[Proxy] instance select failed: err=%s", str(e))
        selection_trace.update({
            "selection_phase": "failed",
            "selection_reason": "selection_exception",
            "selection_error": str(e),
        })
        return None, None, selection_trace


#--------------------------------------------------------------
# ======================= 本地代理方法路由 =======================
#--------------------------------------------------------------

@proxy.post("/v1/chat/completions")
async def proxy_chat_completions(request: FastAPIRequest):
    """
    接收来自 Scheduler 的 /v1/chat/completions 请求（payload为 Request JSON）。
    转发为 OpenAI chat/completions body 到 Worker（流式）
    """
    proxy_recv_ms = int(time.time() * 1000)
    try:
        payload: Dict[str, Any] = await request.json()
    except Exception as e:
        logger.exception("[Proxy] chat/completions 解析 JSON 失败")
        return JSONResponse(
            status_code=400,
            content={"error": "invalid_json", "detail": str(e)},
        )

    # 恢复内部 Request
    try:
        req_obj = recover_request_from_payload(payload)
    except Exception as e:
        logger.exception("[Proxy] 恢复 Request 失败")
        return JSONResponse(
            status_code=400,
            content={"error": "invalid_request_payload", "detail": str(e)},
        )

    # 构造 Instance 请求体
    instance_body = build_body_for_instance(req_obj, mode="chat")

    route_select_start_ms = int(time.time() * 1000)
    chosen, rl_decision, selection_trace = await select_instance(proxy, req_obj)
    route_select_end_ms = int(time.time() * 1000)
    if not chosen:
        no_safe = selection_trace.get("selection_reason") == "no_safe_instances"
        return JSONResponse(
            status_code=503,
            content={
                "error": "no_safe_instance" if no_safe else "no_instance",
                "detail": (
                    "proxy has no safe instance"
                    if no_safe else "proxy has no alive instance"
                ),
                "selection_trace": selection_trace,
            },
        )

    host = chosen.host
    port = int(chosen.port)
    url_path = "/v1/chat/completions"
    logger.info("[Proxy] instance chosen(chat): id=%s addr=%s:%s", getattr(chosen, "instance_id", "?"), host, port)
    strategy_name = getattr(proxy.state, "injection_strategy_name", "default")
    if strategy_name == "iws":
        original_mode = getattr(req_obj.Service, "Injection_type", "text")
        applied_mode = original_mode
        try:
            costs = await queue_mgr.estimate_iws_costs(
                req_obj=req_obj,
                instance_id=chosen.instance_id,
                kdn_addr=getattr(req_obj.Task, "KDN_server_addr", None),
            )
            rag_enabled = bool(getattr(req_obj.Service, "Enable_know_injection", False))
            knowledge_len = int(getattr(req_obj.Service, "Knowledge_length", 0) or 0)
            knowledge_list = getattr(req_obj.Service, "Knowledge_List", []) or []
            if (not rag_enabled) or knowledge_len <= 0 or (not knowledge_list):
                iws_suggest = "text"
                iws_reason = "no_rag_or_empty_knowledge"
            else:
                text_total_ms = float(costs.get("text_total_ms") or 0.0)
                kvcache_total_ms = float(costs.get("kvcache_total_ms") or 0.0)
                kv_queue_wait_ms = float(costs.get("kv_queue_wait_ms") or 0.0)
                text_net_wait_ms = float(costs.get("text_net_wait_ms") or 0.0)
                ready_wait_ms = float(costs.get("ready_wait_ms") or 0.0)
                kv_prepare_ms = float(costs.get("kvcache_prepare_ms") or 0.0)
                kdn_queue_penalty_ms = IWS_KDN_QUEUE_PENALTY_ALPHA * kv_queue_wait_ms
                kvcache_score_ms = kvcache_total_ms + kdn_queue_penalty_ms
                text_score_ms = text_total_ms
                costs["kdn_queue_penalty_ms"] = kdn_queue_penalty_ms
                costs["kvcache_score_ms"] = kvcache_score_ms
                costs["text_score_ms"] = text_score_ms
                if kvcache_score_ms + IWS_DECISION_MARGIN_MS < text_score_ms:
                    iws_suggest = "kvcache"
                    if kv_prepare_ms <= ready_wait_ms:
                        iws_reason = "kvcache_hidden_and_score_better"
                    else:
                        iws_reason = "kvcache_score_better"
                else:
                    iws_suggest = "text"
                    if kv_queue_wait_ms > 0:
                        iws_reason = "text_due_to_kdn_congestion"
                    elif text_net_wait_ms > 0:
                        iws_reason = "text_despite_active_kv_wait"
                    else:
                        iws_reason = "text_score_better"
            req_obj.Service.Injection_type = iws_suggest
            applied_mode = iws_suggest
            logger.info(
                "[Proxy][IWS] rid=%s original=%s iws_suggest=%s applied=%s reason=%s ready_wait=%s "
                "text_prepare_wait=%s text_net_wait=%s text_fetch_fixed=%s kdn_active_until=%s "
                "kv_prepare=%s kv_hidden=%s kv_queue_wait=%s text_total=%s kvcache_total=%s "
                "text_overlap_hidden=%s text_total_formula=%s "
                "text_score=%s kvcache_score=%s kdn_queue_penalty=%s iws_alpha=%s iws_margin=%s "
                "text_service=%s kvcache_service=%s kv_transfer=%s redis_load=%s residual_prefill=%s "
                "effective_len=%s residual_tokens=%s bw=%s bw_src=%s",
                req_obj.Request_ID,
                original_mode,
                iws_suggest,
                applied_mode,
                iws_reason,
                costs.get("ready_wait_ms"),
                costs.get("text_prepare_wait_ms"),
                costs.get("text_net_wait_ms"),
                costs.get("text_fetch_fixed_ms"),
                costs.get("kdn_active_until_ms"),
                costs.get("kvcache_prepare_ms"),
                costs.get("kv_hidden_by_ready_wait"),
                costs.get("kv_queue_wait_ms"),
                costs.get("text_total_ms"),
                costs.get("kvcache_total_ms"),
                costs.get("text_overlap_hidden_ms"),
                "overlap",
                costs.get("text_score_ms"),
                costs.get("kvcache_score_ms"),
                costs.get("kdn_queue_penalty_ms"),
                IWS_KDN_QUEUE_PENALTY_ALPHA,
                IWS_DECISION_MARGIN_MS,
                costs.get("text_service_ms"),
                costs.get("kvcache_service_ms"),
                costs.get("kv_transfer_ms"),
                costs.get("redis_load_ms"),
                costs.get("residual_prefill_ms"),
                costs.get("effective_knowledge_len"),
                costs.get("residual_tokens"),
                costs.get("bandwidth_mbps"),
                costs.get("bandwidth_source"),
            )
        except Exception as e:
            logger.warning(
                "[Proxy][IWS] estimate_iws_costs failed rid=%s err=%s keep=%s",
                req_obj.Request_ID,
                str(e),
                original_mode,
            )

    # ====================================
    # 送入队列enqueue -> manager -> forward
    # ====================================
    route_enqueued = False
    try:
        # 1) 封装任务（注：chosen 来自 RR，具备 instance_id/host/port 字段 :contentReference[oaicite:5]{index=5}）
        task = ProxyTask(
            request_id=getattr(req_obj, "Request_ID", None),
            req_obj=req_obj,
            instance_body=instance_body,
            instance_id=chosen.instance_id,
            instance_host=chosen.host,
            instance_port=int(chosen.port),
            instance_control_port=_instance_control_port(chosen),
            kdn_addr=getattr(req_obj.Task, "KDN_server_addr", None),
            url_path=url_path,
        )
        task.trace["proxy_recv_ms"] = proxy_recv_ms
        task.trace["route_select_start_ms"] = route_select_start_ms
        task.trace["route_select_end_ms"] = route_select_end_ms
        task.trace.update(selection_trace)
        task.trace["selected_instance_control_port"] = int(task.instance_control_port or config.INSTANCE_CP_PORT)
        if rl_decision is not None:
            task.trace["rl_features"] = rl_decision.features
            task.trace["rl_score_milli"] = int(rl_decision.score * 1000)

        await queue_mgr.enqueue_prepare(task)
        route_enqueued = True

        stream_gen = _wrap_chat_stream_with_meta(task, queue_mgr)
        return StreamingResponse(stream_gen, media_type="text/event-stream")

    except Exception as e:
        if not route_enqueued:
            queue_mgr.release_route(chosen.instance_id)
        logger.exception("[Proxy] 调用 Worker(chat) 失败")
        return JSONResponse(
            status_code=502,
            content={"error": "worker_chat_failed", "detail": str(e)},
        )



@proxy.post("/v1/completions")
async def proxy_completions(request: FastAPIRequest):
    """
    接收来自 Scheduler 的 /v1/completions 请求。
    Demo 里逻辑与 chat/completions 相同，只是留出扩展空间。
    """
    proxy_recv_ms = int(time.time() * 1000)
    try:
        payload: Dict[str, Any] = await request.json()
    except Exception as e:
        logger.exception("[Proxy] completions 解析 JSON 失败")
        return JSONResponse(
            status_code=400,
            content={"error": "invalid_json", "detail": str(e)},
        )

    # 恢复内部 Request
    try:
        req_obj = recover_request_from_payload(payload)
    except Exception as e:
        logger.exception("[Proxy] 恢复 Request 失败")
        return JSONResponse(
            status_code=400,
            content={"error": "invalid_request_payload", "detail": str(e)},
        )

    # 构造 Instance 请求体
    instance_body = build_body_for_instance(req_obj, mode="completions")

    route_select_start_ms = int(time.time() * 1000)
    chosen, rl_decision, selection_trace = await select_instance(proxy, req_obj)
    route_select_end_ms = int(time.time() * 1000)
    if not chosen:
        no_safe = selection_trace.get("selection_reason") == "no_safe_instances"
        return JSONResponse(
            status_code=503,
            content={
                "error": "no_safe_instance" if no_safe else "no_instance",
                "detail": (
                    "proxy has no safe instance"
                    if no_safe else "proxy has no alive instance"
                ),
                "selection_trace": selection_trace,
            },
        )

    host = chosen.host
    port = int(chosen.port)
    url_path = "/v1/completions"

    logger.info("[Proxy] instance chosen(completions): id=%s addr=%s:%s", getattr(chosen, "instance_id", "?"), host, port)
    strategy_name = getattr(proxy.state, "injection_strategy_name", "default")
    if strategy_name == "iws":
        original_mode = getattr(req_obj.Service, "Injection_type", "text")
        applied_mode = original_mode
        try:
            costs = await queue_mgr.estimate_iws_costs(
                req_obj=req_obj,
                instance_id=chosen.instance_id,
                kdn_addr=getattr(req_obj.Task, "KDN_server_addr", None),
            )
            rag_enabled = bool(getattr(req_obj.Service, "Enable_know_injection", False))
            knowledge_len = int(getattr(req_obj.Service, "Knowledge_length", 0) or 0)
            knowledge_list = getattr(req_obj.Service, "Knowledge_List", []) or []
            if (not rag_enabled) or knowledge_len <= 0 or (not knowledge_list):
                iws_suggest = "text"
                iws_reason = "no_rag_or_empty_knowledge"
            else:
                text_total_ms = float(costs.get("text_total_ms") or 0.0)
                kvcache_total_ms = float(costs.get("kvcache_total_ms") or 0.0)
                kv_queue_wait_ms = float(costs.get("kv_queue_wait_ms") or 0.0)
                text_net_wait_ms = float(costs.get("text_net_wait_ms") or 0.0)
                ready_wait_ms = float(costs.get("ready_wait_ms") or 0.0)
                kv_prepare_ms = float(costs.get("kvcache_prepare_ms") or 0.0)
                kdn_queue_penalty_ms = IWS_KDN_QUEUE_PENALTY_ALPHA * kv_queue_wait_ms
                kvcache_score_ms = kvcache_total_ms + kdn_queue_penalty_ms
                text_score_ms = text_total_ms
                costs["kdn_queue_penalty_ms"] = kdn_queue_penalty_ms
                costs["kvcache_score_ms"] = kvcache_score_ms
                costs["text_score_ms"] = text_score_ms
                if kvcache_score_ms + IWS_DECISION_MARGIN_MS < text_score_ms:
                    iws_suggest = "kvcache"
                    if kv_prepare_ms <= ready_wait_ms:
                        iws_reason = "kvcache_hidden_and_score_better"
                    else:
                        iws_reason = "kvcache_score_better"
                else:
                    iws_suggest = "text"
                    if kv_queue_wait_ms > 0:
                        iws_reason = "text_due_to_kdn_congestion"
                    elif text_net_wait_ms > 0:
                        iws_reason = "text_despite_active_kv_wait"
                    else:
                        iws_reason = "text_score_better"
            req_obj.Service.Injection_type = iws_suggest
            applied_mode = iws_suggest
            logger.info(
                "[Proxy][IWS] rid=%s original=%s iws_suggest=%s applied=%s reason=%s ready_wait=%s "
                "text_prepare_wait=%s text_net_wait=%s text_fetch_fixed=%s kdn_active_until=%s "
                "kv_prepare=%s kv_hidden=%s kv_queue_wait=%s text_total=%s kvcache_total=%s "
                "text_overlap_hidden=%s text_total_formula=%s "
                "text_score=%s kvcache_score=%s kdn_queue_penalty=%s iws_alpha=%s iws_margin=%s "
                "text_service=%s kvcache_service=%s kv_transfer=%s redis_load=%s residual_prefill=%s "
                "effective_len=%s residual_tokens=%s bw=%s bw_src=%s",
                req_obj.Request_ID,
                original_mode,
                iws_suggest,
                applied_mode,
                iws_reason,
                costs.get("ready_wait_ms"),
                costs.get("text_prepare_wait_ms"),
                costs.get("text_net_wait_ms"),
                costs.get("text_fetch_fixed_ms"),
                costs.get("kdn_active_until_ms"),
                costs.get("kvcache_prepare_ms"),
                costs.get("kv_hidden_by_ready_wait"),
                costs.get("kv_queue_wait_ms"),
                costs.get("text_total_ms"),
                costs.get("kvcache_total_ms"),
                costs.get("text_overlap_hidden_ms"),
                "overlap",
                costs.get("text_score_ms"),
                costs.get("kvcache_score_ms"),
                costs.get("kdn_queue_penalty_ms"),
                IWS_KDN_QUEUE_PENALTY_ALPHA,
                IWS_DECISION_MARGIN_MS,
                costs.get("text_service_ms"),
                costs.get("kvcache_service_ms"),
                costs.get("kv_transfer_ms"),
                costs.get("redis_load_ms"),
                costs.get("residual_prefill_ms"),
                costs.get("effective_knowledge_len"),
                costs.get("residual_tokens"),
                costs.get("bandwidth_mbps"),
                costs.get("bandwidth_source"),
            )
        except Exception as e:
            logger.warning(
                "[Proxy][IWS] estimate_iws_costs failed rid=%s err=%s keep=%s",
                req_obj.Request_ID,
                str(e),
                original_mode,
            )

    # ==================================
    # 送入队列enqueue -> drain -> forward
    # ==================================
    route_enqueued = False
    try:
        task = ProxyTask(
            request_id=getattr(req_obj, "Request_ID", None),
            req_obj=req_obj,
            instance_body=instance_body,
            instance_id=chosen.instance_id,
            instance_host=chosen.host,
            instance_port=int(chosen.port),
            instance_control_port=_instance_control_port(chosen),
            kdn_addr=getattr(req_obj.Task, "KDN_server_addr", None),
            url_path=url_path,
        )
        task.trace["proxy_recv_ms"] = proxy_recv_ms
        task.trace["route_select_start_ms"] = route_select_start_ms
        task.trace["route_select_end_ms"] = route_select_end_ms
        task.trace.update(selection_trace)
        task.trace["selected_instance_control_port"] = int(task.instance_control_port or config.INSTANCE_CP_PORT)

        await queue_mgr.enqueue_prepare(task)
        route_enqueued = True

        content_bytes = b""
        async for chunk in queue_mgr.iter_response(task):
            if chunk:
                content_bytes += chunk

        # completions 是非流式：worker 应该返回一次性 JSON
        if not content_bytes:
            return JSONResponse(
                status_code=502,
                content={"error": "empty_worker_response", "detail": "instance returned empty body"},
            )

        # 尝试按 JSON 解析；解析失败就原样返回文本，便于排查
        try:
            obj = json.loads(content_bytes.decode("utf-8", errors="replace"))
            obj["_cacheroute_meta"] = build_cacheroute_meta(task)
            return JSONResponse(status_code=200, content=obj)
        except Exception:
            return JSONResponse(
                status_code=200,
                content={
                    "raw": content_bytes.decode("utf-8", errors="replace"),
                    "_cacheroute_meta": build_cacheroute_meta(task),
                }
            )

    except Exception as e:
        if not route_enqueued:
            queue_mgr.release_route(chosen.instance_id)
        logger.exception("[Proxy] 调用 Worker(completions) 失败")
        return JSONResponse(
            status_code=502,
            content={"error": "worker_completions_failed", "detail": str(e)},
        )
