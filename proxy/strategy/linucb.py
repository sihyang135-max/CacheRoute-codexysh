"""Low-overhead shared-parameter LinUCB for Proxy -> Instance routing."""
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
    candidate_features: Dict[str, List[float]]


def ttft_reward(ttft_ms: float, scale_ms: float, clip: float) -> float:
    """Return a bounded reward that directly minimizes observed TTFT."""
    if scale_ms <= 0 or clip <= 0:
        raise ValueError("scale_ms and clip must be positive")
    return -min(max(0.0, float(ttft_ms)) / float(scale_ms), float(clip))


class LinUCBStrategy(BaseInstanceStrategy):
    """One shared linear model, safe for a dynamically changing instance pool."""

    name = "linucb"

    def __init__(
        self,
        alpha: float = config.PROXY_RL_ALPHA,
        ridge: float = config.PROXY_RL_LAMBDA,
        warmup_requests: int = config.PROXY_RL_WARMUP_REQUESTS,
    ) -> None:
        self.alpha = max(0.0, float(alpha))
        self.ridge = max(1e-6, float(ridge))
        self.warmup_requests = max(0, int(warmup_requests))
        self._dim = 7  # bias + six documented state features
        self._A = np.eye(self._dim, dtype=float) * self.ridge
        self._b = np.zeros(self._dim, dtype=float)
        self._effective_updates = 0
        self._lock = threading.Lock()

    @staticmethod
    def _vector(row: Dict[str, Any]) -> np.ndarray:
        # Values are normalized by the context builder; clip protects the model
        # from stale/invalid monitoring input without adding request-path I/O.
        vals = [
            1.0,
            float(row["prompt_norm"]),
            float(row["kv_norm"]),
            float(row["prefill_norm"]),
            float(row["decode_norm"]),
            float(row["kv_usage"]),
            float(row["net_norm"]),
        ]
        return np.asarray([min(4.0, max(-4.0, v)) for v in vals], dtype=float)

    def choose(self, instances: Sequence[InstanceLike], contexts: Dict[str, Dict[str, Any]]) -> Decision:
        if not instances:
            raise RuntimeError("no instances")
        rows = [(it, contexts.get(it.instance_id)) for it in instances]
        rows = [(it, row) for it, row in rows if row is not None]
        if not rows:
            raise RuntimeError("no valid LinUCB contexts")

        with self._lock:
            candidate_features = {
                it.instance_id: self._vector(row).tolist()
                for it, row in rows
            }
            if self._effective_updates < self.warmup_requests:
                # Deterministic Least-InFlight-like cold start using local queue state.
                it, row = min(rows, key=lambda pair: (float(pair[1].get("prefill_raw", 0)) + float(pair[1].get("decode_raw", 0)), pair[0].instance_id))
                return Decision(
                    instance_id=it.instance_id,
                    features=candidate_features[it.instance_id],
                    score=0.0,
                    phase="warmup",
                    effective_updates=self._effective_updates,
                    candidate_scores={item.instance_id: 0.0 for item, _ in rows},
                    candidate_features=candidate_features,
                )

            theta = np.linalg.solve(self._A, self._b)
            inv_A = np.linalg.inv(self._A)
            best: Optional[Decision] = None
            candidate_scores: Dict[str, float] = {}
            for it, row in rows:
                x = self._vector(row)
                bonus = self.alpha * math.sqrt(max(0.0, float(x @ inv_A @ x)))
                score = float(theta @ x) + bonus
                candidate_scores[it.instance_id] = score
                candidate = Decision(
                    instance_id=it.instance_id,
                    features=x.tolist(),
                    score=score,
                    phase="linucb",
                    effective_updates=self._effective_updates,
                    candidate_scores={},
                    candidate_features=candidate_features,
                )
                if best is None or candidate.score > best.score or (candidate.score == best.score and candidate.instance_id < best.instance_id):
                    best = candidate
            assert best is not None
            return Decision(
                instance_id=best.instance_id,
                features=best.features,
                score=best.score,
                phase=best.phase,
                effective_updates=best.effective_updates,
                candidate_scores=candidate_scores,
                candidate_features=candidate_features,
            )

    def select(self, instances: List[InstanceLike], hint: Optional[Any] = None) -> InstanceLike:
        contexts = (hint or {}).get("linucb_contexts", {}) if isinstance(hint, dict) else {}
        chosen = self.choose(instances, contexts)
        return next(it for it in instances if it.instance_id == chosen.instance_id)

    def update(self, features: Sequence[float], reward: float) -> None:
        x = np.asarray(features, dtype=float)
        if x.shape != (self._dim,) or not np.all(np.isfinite(x)) or not math.isfinite(float(reward)):
            return
        with self._lock:
            self._A += np.outer(x, x)
            self._b += x * float(reward)
            self._effective_updates += 1

    @property
    def effective_updates(self) -> int:
        with self._lock:
            return self._effective_updates
