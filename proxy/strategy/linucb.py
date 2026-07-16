"""Compact load-safe LinUCB for Proxy -> Instance routing."""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from core import config
from .base import BaseInstanceStrategy, InstanceLike


@dataclass(frozen=True)
class Decision:
    instance_id: str
    features: List[float]
    score: float
    phase: str
    effective_updates: int
    candidate_scores: Dict[str, float]
    candidate_exploit_scores: Dict[str, float]
    candidate_exploration_bonuses: Dict[str, float]
    candidate_features: Dict[str, List[float]]


def ttft_reward(ttft_ms: float, scale_ms: float, clip: float) -> float:
    """Return a bounded reward that directly minimizes observed TTFT."""
    if scale_ms <= 0 or clip <= 0:
        raise ValueError("scale_ms and clip must be positive")
    return -min(max(0.0, float(ttft_ms)) / float(scale_ms), float(clip))


class LinUCBStrategy(BaseInstanceStrategy):
    """Learn one shared TTFT model from two action-dependent costs.

    The request path supplies predicted compute and KV-ready costs in
    milliseconds. They are centered across the current candidates before
    scoring, so a cost shared by every instance is exactly zero and cannot
    affect the route. Exploration depends only on per-instance sample count;
    high load or a slow KV path therefore never earns a larger bonus.
    """

    name = "linucb"

    def __init__(
        self,
        alpha: float = config.PROXY_RL_ALPHA,
        ridge: float = config.PROXY_RL_LAMBDA,
        warmup_requests: int = config.PROXY_RL_WARMUP_REQUESTS,
        compute_scale_ms: float = config.PROXY_RL_COMPUTE_COST_SCALE_MS,
        kv_ready_scale_ms: float = config.PROXY_RL_KV_READY_COST_SCALE_MS,
    ) -> None:
        self.alpha = max(0.0, float(alpha))
        self.ridge = max(1e-6, float(ridge))
        self.warmup_requests = max(0, int(warmup_requests))
        self.compute_scale_ms = max(1.0, float(compute_scale_ms))
        self.kv_ready_scale_ms = max(1.0, float(kv_ready_scale_ms))
        self._feature_names = (
            "bias",
            "compute_delta_norm",
            "kv_ready_delta_norm",
        )
        self._dim = len(self._feature_names)
        self._A_inv = np.eye(self._dim, dtype=float) / self.ridge
        self._b = np.zeros(self._dim, dtype=float)
        self._arm_updates: Dict[str, int] = {}
        self._arm_selections: Dict[str, int] = {}
        self._effective_updates = 0
        self._effective_selections = 0
        self._tie_cursor = 0
        self._lock = threading.Lock()

    @staticmethod
    def _cost(row: Dict[str, Any], key: str) -> float:
        value = float(row.get(key, 0.0) or 0.0)
        return value if math.isfinite(value) and value > 0.0 else 0.0

    def _candidate_vectors(
        self,
        rows: Sequence[tuple[InstanceLike, Dict[str, Any]]],
    ) -> Dict[str, np.ndarray]:
        compute = [self._cost(row, "compute_cost_ms") for _, row in rows]
        kv_ready = [self._cost(row, "kv_ready_cost_ms") for _, row in rows]
        min_compute = min(compute)
        min_kv_ready = min(kv_ready)
        vectors: Dict[str, np.ndarray] = {}
        for (item, _), compute_ms, kv_ready_ms in zip(rows, compute, kv_ready):
            vectors[item.instance_id] = np.asarray(
                [
                    1.0,
                    min(4.0, max(0.0, compute_ms - min_compute) / self.compute_scale_ms),
                    min(4.0, max(0.0, kv_ready_ms - min_kv_ready) / self.kv_ready_scale_ms),
                ],
                dtype=float,
            )
        return vectors

    def _score(
        self,
        instance_id: str,
        x: np.ndarray,
        theta: np.ndarray,
    ) -> tuple[float, float, float]:
        exploit = float(theta @ x)
        samples = int(self._arm_updates.get(instance_id, 0))
        bonus = self.alpha / math.sqrt(self.ridge + samples)
        return exploit + bonus, exploit, bonus

    def choose(
        self,
        instances: Sequence[InstanceLike],
        contexts: Dict[str, Dict[str, Any]],
    ) -> Decision:
        if not instances:
            raise RuntimeError("no instances")
        rows = [(item, contexts.get(item.instance_id)) for item in instances]
        rows = [(item, row) for item, row in rows if isinstance(row, dict)]
        if not rows:
            raise RuntimeError("no valid LinUCB contexts")

        with self._lock:
            for item, _ in rows:
                self._arm_updates.setdefault(item.instance_id, 0)
                self._arm_selections.setdefault(item.instance_id, 0)
            vectors = self._candidate_vectors(rows)
            candidate_features = {
                instance_id: vector.tolist()
                for instance_id, vector in vectors.items()
            }

            if self._effective_updates < self.warmup_requests:
                minimum = min(self._arm_selections[item.instance_id] for item, _ in rows)
                least_sampled = [
                    (item, row)
                    for item, row in rows
                    if self._arm_selections[item.instance_id] == minimum
                ]
                minimum_cost = min(
                    self._cost(row, "compute_cost_ms")
                    + self._cost(row, "kv_ready_cost_ms")
                    for _, row in least_sampled
                )
                tied = [
                    (item, row)
                    for item, row in least_sampled
                    if math.isclose(
                        self._cost(row, "compute_cost_ms")
                        + self._cost(row, "kv_ready_cost_ms"),
                        minimum_cost,
                        abs_tol=1e-9,
                    )
                ]
                item, _ = tied[self._tie_cursor % len(tied)]
                self._tie_cursor += 1
                self._arm_selections[item.instance_id] += 1
                self._effective_selections += 1
                zeros = {candidate.instance_id: 0.0 for candidate, _ in rows}
                return Decision(
                    instance_id=item.instance_id,
                    features=candidate_features[item.instance_id],
                    score=0.0,
                    phase="warmup",
                    effective_updates=self._effective_updates,
                    candidate_scores=zeros,
                    candidate_exploit_scores=dict(zeros),
                    candidate_exploration_bonuses=dict(zeros),
                    candidate_features=candidate_features,
                )

            candidate_scores: Dict[str, float] = {}
            candidate_exploit_scores: Dict[str, float] = {}
            candidate_exploration_bonuses: Dict[str, float] = {}
            theta = self._A_inv @ self._b
            # Both learned variables are costs. Confounded samples must not
            # make a larger predicted delay look beneficial.
            theta[1:] = np.minimum(theta[1:], 0.0)
            for item, _ in rows:
                score, exploit, bonus = self._score(
                    item.instance_id,
                    vectors[item.instance_id],
                    theta,
                )
                candidate_scores[item.instance_id] = score
                candidate_exploit_scores[item.instance_id] = exploit
                candidate_exploration_bonuses[item.instance_id] = bonus

            maximum = max(candidate_scores.values())
            tied_ids = [
                item.instance_id
                for item, _ in rows
                if math.isclose(candidate_scores[item.instance_id], maximum, abs_tol=1e-12)
            ]
            chosen_id = tied_ids[self._tie_cursor % len(tied_ids)]
            self._tie_cursor += 1
            self._arm_selections[chosen_id] += 1
            self._effective_selections += 1
            return Decision(
                instance_id=chosen_id,
                features=candidate_features[chosen_id],
                score=candidate_scores[chosen_id],
                phase="linucb",
                effective_updates=self._effective_updates,
                candidate_scores=candidate_scores,
                candidate_exploit_scores=candidate_exploit_scores,
                candidate_exploration_bonuses=candidate_exploration_bonuses,
                candidate_features=candidate_features,
            )

    def select(self, instances: List[InstanceLike], hint: Optional[Any] = None) -> InstanceLike:
        contexts = (hint or {}).get("linucb_contexts", {}) if isinstance(hint, dict) else {}
        decision = self.choose(instances, contexts)
        return next(item for item in instances if item.instance_id == decision.instance_id)

    def update(self, instance_id: str, features: Sequence[float], reward: float) -> None:
        x = np.asarray(features, dtype=float)
        if x.shape != (self._dim,) or not np.all(np.isfinite(x)) or not math.isfinite(float(reward)):
            return
        with self._lock:
            # Sherman-Morrison keeps the request feedback update O(d^2), with
            # d=3, and avoids solve/inverse calls in the routing path.
            projected = self._A_inv @ x
            denominator = 1.0 + float(x @ projected)
            if denominator <= 1e-12:
                return
            self._A_inv -= np.outer(projected, projected) / denominator
            self._b += x * float(reward)
            key = str(instance_id)
            self._arm_updates[key] = int(self._arm_updates.get(key, 0)) + 1
            self._arm_selections.setdefault(key, 0)
            self._effective_updates += 1

    @property
    def effective_updates(self) -> int:
        with self._lock:
            return self._effective_updates

    @property
    def effective_selections(self) -> int:
        with self._lock:
            return self._effective_selections

    @property
    def arm_updates(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._arm_updates)

    @property
    def arm_selections(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._arm_selections)

    @property
    def feature_names(self) -> List[str]:
        return list(self._feature_names)
