from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence, Tuple

from .base import InstanceLike


def filter_safe_instances(
    instances: Sequence[InstanceLike],
    *,
    metric_for: Callable[[str], Any],
    queue_snapshot_for: Callable[[str], Dict[str, int]],
    metric_stale_s: float,
    metric_failure_limit: int,
    kv_usage_limit: float,
    queue_limit: int = 256,
) -> Tuple[List[InstanceLike], Dict[str, str]]:
    """Return one deterministic safety-filtered candidate set for all policies."""
    safe: List[InstanceLike] = []
    excluded: Dict[str, str] = {}

    for item in sorted(instances, key=lambda candidate: candidate.instance_id):
        metric = metric_for(item.instance_id)
        if metric is None:
            excluded[item.instance_id] = "metrics_missing"
            continue
        if not metric.is_fresh(metric_stale_s):
            excluded[item.instance_id] = "metrics_stale"
            continue
        if metric.failures >= metric_failure_limit:
            excluded[item.instance_id] = "metrics_failures"
            continue
        if metric.kv_usage >= kv_usage_limit:
            excluded[item.instance_id] = "kv_usage_limit"
            continue

        queue = queue_snapshot_for(item.instance_id)
        if (
            queue["prepare_queue_size"] >= queue_limit
            or queue["ready_queue_size"] >= queue_limit
        ):
            excluded[item.instance_id] = "proxy_queue_limit"
            continue
        safe.append(item)

    return safe, excluded
