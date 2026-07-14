from __future__ import annotations

from typing import Any, List, Optional

from .base import BaseInstanceStrategy, InstanceLike


class LeastInflightStrategy(BaseInstanceStrategy):
    """Safe fallback; uses heartbeat load when available."""
    name = "least_inflight"

    def select(self, instances: List[InstanceLike], hint: Optional[Any] = None) -> InstanceLike:
        if not instances:
            raise RuntimeError("no instances")
        return min(
            instances,
            key=lambda item: (int(getattr(getattr(item, "load", None), "inflight", 0) or 0), item.instance_id),
        )
