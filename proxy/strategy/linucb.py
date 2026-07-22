"""Standard disjoint LinUCB with compact routing context."""
from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
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


def classify_feedback(
    first_token_ms: Any,
    enqueued_ms: Any,
    task_error: str = "",
    explicit_outcome: str = "",
) -> tuple[str, str, bool]:
    """Classify whether request feedback is attributable to the selected arm."""
    error = str(task_error or "")
    explicit = str(explicit_outcome or "").strip()
    if isinstance(first_token_ms, int) and isinstance(enqueued_ms, int):
        return "success", "observed_ttft", True
    if explicit == "client_cancelled" or error == "client_cancelled":
        return "client_cancelled", "ignored_client_cancelled", False
    if error.startswith("ready_failed:"):
        outcome = "timeout" if "timeout" in error.lower() else "backend_failure"
        return outcome, "instance_failure_penalty", True
    if error.startswith("stream_wrap_failed:"):
        return "proxy_stream_failure", "ignored_proxy_failure", False
    return "missing_trace", "ignored_missing_first_token", False


class LinUCBStrategy(BaseInstanceStrategy):
    """Run standard disjoint LinUCB on two action-dependent costs.

    The request path supplies predicted compute and KV-ready costs in
    milliseconds. They are centered across the current candidates before
    scoring, so a cost shared by every instance is exactly zero. Each instance
    keeps its own linear model and standard context-dependent confidence bound.
    """

    name = "linucb"
    model_format_version = 1

    def __init__(
        self,
        alpha: float = config.PROXY_RL_ALPHA,
        ridge: float = config.PROXY_RL_LAMBDA,
        warmup_requests: int = config.PROXY_RL_WARMUP_REQUESTS,
        compute_scale_ms: float = config.PROXY_RL_COMPUTE_COST_SCALE_MS,
        kv_ready_scale_ms: float = config.PROXY_RL_KV_READY_COST_SCALE_MS,
        frozen: bool = config.PROXY_RL_FROZEN,
        model_save_path: str = config.PROXY_RL_MODEL_SAVE_PATH,
        source_commit: str = config.PROXY_RL_SOURCE_COMMIT,
        parameter_snapshot_path: str = config.PROXY_RL_PARAMETER_SNAPSHOT_PATH,
        parameter_snapshot_interval: int = config.PROXY_RL_PARAMETER_SNAPSHOT_INTERVAL,
    ) -> None:
        self.alpha = max(0.0, float(alpha))
        self.ridge = max(1e-6, float(ridge))
        self.warmup_requests = max(0, int(warmup_requests))
        self.compute_scale_ms = max(1.0, float(compute_scale_ms))
        self.kv_ready_scale_ms = max(1.0, float(kv_ready_scale_ms))
        self.frozen = bool(frozen)
        self.model_save_path = str(model_save_path or "").strip()
        self.source_commit = str(source_commit or "").strip()
        self._model_created_at_unix_ns = time.time_ns()
        self.parameter_snapshot_path = str(parameter_snapshot_path or "").strip()
        self.parameter_snapshot_interval = max(1, int(parameter_snapshot_interval))
        self._feature_names = (
            "bias",
            "compute_delta_norm",
            "kv_ready_delta_norm",
        )
        self._dim = len(self._feature_names)
        self._arms: Dict[str, Dict[str, Any]] = {}
        self._arm_selections: Dict[str, int] = {}
        self._effective_updates = 0
        self._effective_selections = 0
        self._tie_cursor = 0
        self._last_snapshot_updates = -1
        self._last_snapshot_theta: Dict[str, np.ndarray] = {}
        self._loaded_instance_ids: Optional[set[str]] = None
        self._lock = threading.Lock()

    def _state_payload_locked(self) -> Dict[str, Any]:
        return {
            "model_format_version": self.model_format_version,
            "feature_names": list(self._feature_names),
            "dim": self._dim,
            "alpha": self.alpha,
            "lambda": self.ridge,
            "warmup_requests": self.warmup_requests,
            "compute_scale_ms": self.compute_scale_ms,
            "kv_ready_scale_ms": self.kv_ready_scale_ms,
            "source_commit": self.source_commit,
            "created_at_unix_ns": self._model_created_at_unix_ns,
            "effective_updates": self._effective_updates,
            "effective_selections": self._effective_selections,
            "tie_cursor": self._tie_cursor,
            "arm_selections": dict(self._arm_selections),
            "matrix_kind": "A_inv",
            "arms": {
                instance_id: {
                    "A_inv": arm["A_inv"].tolist(),
                    "b": arm["b"].tolist(),
                    "updates": int(arm["updates"]),
                }
                for instance_id, arm in sorted(self._arms.items())
            },
        }

    @staticmethod
    def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    def save_model(self, path: Optional[str] = None) -> Dict[str, Any]:
        raw_path = str(path or self.model_save_path or "").strip()
        if not raw_path:
            raise ValueError("LinUCB model save path is empty")
        target = Path(raw_path)
        with self._lock:
            payload = self._state_payload_locked()
        payload["saved_at_unix_ns"] = time.time_ns()
        self._atomic_write_json(target, payload)
        return payload

    def load_model(self, path: str) -> Dict[str, Any]:
        source = Path(str(path or "").strip())
        if not source.is_file():
            raise ValueError(f"LinUCB model file does not exist: {source}")
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"invalid LinUCB model JSON: {source}") from exc
        expected = {
            "model_format_version": self.model_format_version,
            "feature_names": list(self._feature_names),
            "dim": self._dim,
            "alpha": self.alpha,
            "lambda": self.ridge,
            "warmup_requests": self.warmup_requests,
            "compute_scale_ms": self.compute_scale_ms,
            "kv_ready_scale_ms": self.kv_ready_scale_ms,
            "source_commit": self.source_commit,
            "matrix_kind": "A_inv",
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                raise ValueError(
                    f"LinUCB model metadata mismatch for {key}: "
                    f"expected={value!r} actual={payload.get(key)!r}"
                )
        arms_payload = payload.get("arms")
        if not isinstance(arms_payload, dict):
            raise ValueError("LinUCB model arms must be an object")
        restored: Dict[str, Dict[str, Any]] = {}
        for instance_id, raw in arms_payload.items():
            if not isinstance(raw, dict):
                raise ValueError(f"invalid LinUCB arm: {instance_id}")
            A_inv = np.asarray(raw.get("A_inv"), dtype=float)
            b = np.asarray(raw.get("b"), dtype=float)
            if (
                A_inv.shape != (self._dim, self._dim)
                or b.shape != (self._dim,)
                or not np.all(np.isfinite(A_inv))
                or not np.all(np.isfinite(b))
            ):
                raise ValueError(f"invalid LinUCB arm dimensions or values: {instance_id}")
            restored[str(instance_id)] = {
                "A_inv": A_inv,
                "b": b,
                "updates": int(raw.get("updates", 0)),
            }
        with self._lock:
            self._arms = restored
            self._arm_selections = {
                str(key): int(value)
                for key, value in dict(payload.get("arm_selections") or {}).items()
            }
            self._effective_updates = int(payload.get("effective_updates", 0))
            self._effective_selections = int(payload.get("effective_selections", 0))
            self._tie_cursor = int(payload.get("tie_cursor", 0))
            self._model_created_at_unix_ns = int(payload.get("created_at_unix_ns", 0))
            self._loaded_instance_ids = set(restored)
            self._last_snapshot_updates = self._effective_updates
            self._last_snapshot_theta = {
                instance_id: arm["A_inv"] @ arm["b"]
                for instance_id, arm in restored.items()
            }
        return payload

    @property
    def runtime_mode(self) -> str:
        if self._loaded_instance_ids is None:
            return "fresh-training"
        return "loaded-frozen" if self.frozen else "loaded-training"

    def parameter_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            arms = {}
            for instance_id, arm in sorted(self._arms.items()):
                theta = arm["A_inv"] @ arm["b"]
                previous = self._last_snapshot_theta.get(instance_id)
                arms[instance_id] = {
                    "updates": int(arm["updates"]),
                    "theta": theta.tolist(),
                    "theta_l2_norm": float(np.linalg.norm(theta)),
                    "theta_delta_l2_norm": (
                        float(np.linalg.norm(theta - previous))
                        if previous is not None
                        else None
                    ),
                }
            return {
                "model_format_version": self.model_format_version,
                "feature_names": list(self._feature_names),
                "effective_updates": self._effective_updates,
                "effective_selections": self._effective_selections,
                "frozen": self.frozen,
                "arms": arms,
            }

    def checkpoint_if_due(self, force: bool = False) -> Optional[Dict[str, Any]]:
        with self._lock:
            updates = self._effective_updates
            already_recorded = updates == self._last_snapshot_updates
        due = force or updates == self.warmup_requests or (
            updates > 0 and updates % self.parameter_snapshot_interval == 0
        )
        if not due or already_recorded:
            return None
        snapshot = self.parameter_snapshot()
        snapshot["recorded_at_unix_ns"] = time.time_ns()
        if self.parameter_snapshot_path:
            target = Path(self.parameter_snapshot_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(snapshot, ensure_ascii=False, sort_keys=True) + "\n")
        if self.model_save_path and not self.frozen:
            self.save_model()
        with self._lock:
            self._last_snapshot_updates = updates
            self._last_snapshot_theta = {
                instance_id: np.asarray(raw["theta"], dtype=float)
                for instance_id, raw in snapshot["arms"].items()
            }
        return snapshot

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

    def _arm(self, instance_id: str) -> Dict[str, Any]:
        key = str(instance_id)
        arm = self._arms.get(key)
        if arm is None:
            arm = {
                "A_inv": np.eye(self._dim, dtype=float) / self.ridge,
                "b": np.zeros(self._dim, dtype=float),
                "updates": 0,
            }
            self._arms[key] = arm
        return arm

    def _score(
        self,
        x: np.ndarray,
        arm: Dict[str, Any],
    ) -> tuple[float, float, float]:
        A_inv = arm["A_inv"]
        theta = A_inv @ arm["b"]
        exploit = float(theta @ x)
        uncertainty = max(0.0, float(x @ A_inv @ x))
        bonus = self.alpha * math.sqrt(uncertainty)
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
        candidate_ids = {item.instance_id for item, _ in rows}
        if self._loaded_instance_ids is not None and candidate_ids != self._loaded_instance_ids:
            raise RuntimeError(
                "loaded LinUCB instance set mismatch: "
                f"expected={sorted(self._loaded_instance_ids)} actual={sorted(candidate_ids)}"
            )

        with self._lock:
            for item, _ in rows:
                self._arm(item.instance_id)
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
            for item, _ in rows:
                score, exploit, bonus = self._score(
                    vectors[item.instance_id],
                    self._arm(item.instance_id),
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

    def update(self, instance_id: str, features: Sequence[float], reward: float) -> bool:
        if self.frozen:
            return False
        x = np.asarray(features, dtype=float)
        if x.shape != (self._dim,) or not np.all(np.isfinite(x)) or not math.isfinite(float(reward)):
            return False
        with self._lock:
            # Sherman-Morrison keeps the request feedback update O(d^2), with
            # d=3, and avoids solve/inverse calls in the routing path.
            arm = self._arm(str(instance_id))
            projected = arm["A_inv"] @ x
            denominator = 1.0 + float(x @ projected)
            if denominator <= 1e-12:
                return False
            arm["A_inv"] -= np.outer(projected, projected) / denominator
            arm["b"] += x * float(reward)
            arm["updates"] = int(arm["updates"]) + 1
            key = str(instance_id)
            self._arm_selections.setdefault(key, 0)
            self._effective_updates += 1
            return True

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
            return {
                instance_id: int(arm["updates"])
                for instance_id, arm in self._arms.items()
            }

    @property
    def arm_selections(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._arm_selections)

    @property
    def feature_names(self) -> List[str]:
        return list(self._feature_names)
