from __future__ import annotations

from typing import Any, List, Optional

from .base import BaseInstanceStrategy, InstanceLike


class LeastInflightStrategy(BaseInstanceStrategy):
    """Safe fallback; uses heartbeat load when available."""
    name = "least_inflight"

    def select(self, instances: List[InstanceLike], hint: Optional[Any] = None) -> InstanceLike:
        if not instances:
            raise RuntimeError("no instances")
        local_inflight = hint.get("instance_inflight", {}) if isinstance(hint, dict) else {}

        def load(item: InstanceLike) -> int:
            if item.instance_id in local_inflight:
                return int(local_inflight[item.instance_id])
            return int(getattr(getattr(item, "load", None), "inflight", 0) or 0)

        return min(
            instances,
            key=lambda item: (load(item), item.instance_id),
        )
