from __future__ import annotations

import threading
from typing import Any, List, Optional

from .base import BaseInstanceStrategy, InstanceLike


class LeastInflightStrategy(BaseInstanceStrategy):
    """Safe fallback; uses heartbeat load when available."""
    name = "least_inflight"

    def __init__(self) -> None:
        self._tie_cursor = 0
        self._lock = threading.Lock()

    def select(self, instances: List[InstanceLike], hint: Optional[Any] = None) -> InstanceLike:
        if not instances:
            raise RuntimeError("no instances")
        local_inflight = {}
        if isinstance(hint, dict):
            local_inflight = hint.get(
                "instance_inflight",
                hint.get("inflight_by_instance", {}),
            )

        def load(item: InstanceLike) -> int:
            if item.instance_id in local_inflight:
                return int(local_inflight[item.instance_id])
            return int(getattr(getattr(item, "load", None), "inflight", 0) or 0)

        loads = [load(item) for item in instances]
        minimum = min(loads)
        tied = [
            item for item, item_load in zip(instances, loads)
            if item_load == minimum
        ]
        with self._lock:
            chosen = tied[self._tie_cursor % len(tied)]
            self._tie_cursor += 1
            return chosen
