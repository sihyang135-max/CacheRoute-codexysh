"""Asynchronous Prometheus cache; routing only reads memory snapshots."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Dict, Optional

import httpx


@dataclass(frozen=True)
class InstanceMetricsSnapshot:
    kv_usage: float
    collected_at_s: float
    failures: int = 0

    def is_fresh(self, stale_s: float) -> bool:
        return (time.time() - self.collected_at_s) <= stale_s


class PrometheusCache:
    """Parses the KV cache utilization gauge across common vLLM metric spellings."""

    _KV_NAMES = (
        "vllm:kv_cache_usage_perc",
        "vllm_kv_cache_usage_perc",
        "vllm:gpu_cache_usage_perc",
        "vllm_gpu_cache_usage_perc",
        "vllm_gpu_kv_cache_usage_perc",
    )

    def __init__(self) -> None:
        self._items: Dict[str, InstanceMetricsSnapshot] = {}

    @staticmethod
    def _metric_value(text: str) -> Optional[float]:
        for name in PrometheusCache._KV_NAMES:
            match = re.search(rf"^{re.escape(name)}(?:\{{[^}}]*\}})?\s+([-+0-9.eE]+)\s*$", text, flags=re.MULTILINE)
            if match:
                value = float(match.group(1))
                return value / 100.0 if value > 1.0 else value
        return None

    async def refresh(self, instance_id: str, metrics_url: str, timeout_s: float) -> None:
        old = self._items.get(instance_id)
        try:
            async with httpx.AsyncClient(timeout=timeout_s) as client:
                response = await client.get(metrics_url)
                response.raise_for_status()
            value = self._metric_value(response.text)
            if value is None:
                raise ValueError("KV usage metric not found")
            self._items[instance_id] = InstanceMetricsSnapshot(
                kv_usage=min(1.0, max(0.0, value)), collected_at_s=time.time(), failures=0
            )
        except Exception:
            self._items[instance_id] = InstanceMetricsSnapshot(
                kv_usage=old.kv_usage if old else 1.0,
                collected_at_s=old.collected_at_s if old else 0.0,
                failures=(old.failures + 1) if old else 1,
            )

    def get(self, instance_id: str) -> Optional[InstanceMetricsSnapshot]:
        return self._items.get(instance_id)
