#!/usr/bin/env python3
"""Validate the bounded Issue #8 train/load/freeze closure artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{number}: expected JSON object")
        rows.append(value)
    return rows


def fixed_context_scores(model: dict[str, Any]) -> dict[str, float]:
    """Recompute deterministic scores for a fixed four-context audit set."""
    alpha = float(model["alpha"])
    scores: dict[str, float] = {}
    for index, (instance_id, arm) in enumerate(sorted(model["arms"].items())):
        x = [1.0, index * 0.25, 0.0]
        matrix = arm["A_inv"]
        b = arm["b"]
        theta = [sum(float(matrix[i][j]) * float(b[j]) for j in range(3)) for i in range(3)]
        exploit = sum(theta[i] * x[i] for i in range(3))
        uncertainty = sum(x[i] * float(matrix[i][j]) * x[j] for i in range(3) for j in range(3))
        scores[instance_id] = exploit + alpha * math.sqrt(max(0.0, uncertainty))
    return scores


def validate(
    training: list[dict[str, Any]], frozen: list[dict[str, Any]], snapshots: list[dict[str, Any]],
    model_before: dict[str, Any], model_after: dict[str, Any], analysis: dict[str, Any],
    expected_commit: str, config: dict[str, Any], before_hash: str, after_hash: str,
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    checks["exact_commit_bound"] = config.get("git_commit") == expected_commit
    checks["concurrency_one"] = config.get("concurrency") == 1
    checks["global_kv_scope"] = config.get("kv_residency_scope") == "global" and config.get("kv_link_scope") == "global"
    checks["training_request_count"] = len(training) == int(config.get("training_requests", -1)) == 240
    checks["frozen_request_count"] = len(frozen) == int(config.get("frozen_requests", -1)) == 80
    checks["all_requests_successful"] = all(row.get("success") is True for row in [*training, *frozen])
    train_traces = [row.get("trace") or {} for row in training]
    frozen_traces = [row.get("trace") or {} for row in frozen]
    checks["training_updates_visible"] = all(t.get("rl_updated") in (1, True) for t in train_traces)
    checks["frozen_never_updates"] = all(t.get("rl_updated") in (0, False) and t.get("rl_update_reason") == "frozen" for t in frozen_traces)
    checks["frozen_flag_visible"] = all(t.get("rl_model_frozen") in (1, True) for t in frozen_traces)
    checks["runtime_modes_explicit"] = all(t.get("rl_runtime_mode") == "fresh-training" for t in train_traces) and all(t.get("rl_runtime_mode") == "loaded-frozen" for t in frozen_traces)
    checks["reward_audit_fields"] = all(t.get("outcome_class") == "success" and t.get("reward_source") == "observed_ttft" for t in [*train_traces, *frozen_traces])
    checks["model_format_v1"] = model_before.get("model_format_version") == model_after.get("model_format_version") == 1
    checks["model_source_commit_bound"] = model_before.get("source_commit") == model_after.get("source_commit") == expected_commit
    checks["model_hash_unchanged"] = before_hash == after_hash
    invariant_keys = ["feature_names", "dim", "alpha", "lambda", "warmup_requests", "compute_scale_ms", "kv_ready_scale_ms", "effective_updates", "arms"]
    checks["model_state_unchanged"] = all(model_before.get(key) == model_after.get(key) for key in invariant_keys)
    before_scores, after_scores = fixed_context_scores(model_before), fixed_context_scores(model_after)
    max_score_delta = max(abs(before_scores[key] - after_scores[key]) for key in before_scores)
    checks["fixed_context_scores_match_1e_9"] = set(before_scores) == set(after_scores) and max_score_delta <= 1e-9
    update_counts = [int(item.get("effective_updates", -1)) for item in snapshots]
    checks["snapshots_monotonic_unique"] = bool(update_counts) and update_counts == sorted(set(update_counts))
    interval = int(config.get("parameter_snapshot_interval", 20))
    warmup = int(config.get("warmup_requests", 30))
    expected_points = sorted(set([warmup, *range(interval, 241, interval)]))
    checks["snapshot_boundaries_complete"] = update_counts == expected_points
    finite_snapshots = True
    for item in snapshots:
        if item.get("feature_names") != ["bias", "compute_delta_norm", "kv_ready_delta_norm"]:
            finite_snapshots = False
        for arm in (item.get("arms") or {}).values():
            values = [*(arm.get("theta") or []), arm.get("theta_l2_norm"), *( [] if arm.get("theta_delta_l2_norm") is None else [arm.get("theta_delta_l2_norm")])]
            finite_snapshots &= len(arm.get("theta") or []) == 3 and all(isinstance(v, (int, float)) and math.isfinite(float(v)) for v in values)
    checks["snapshot_theta_finite"] = finite_snapshots
    window_results = analysis.get("windows") or {}
    checks["analysis_has_100_200"] = set(window_results) == {"100", "200"}
    checks["analysis_does_not_claim_convergence"] = all((window_results.get(size, {}).get("assessment") or {}).get("status") in {"insufficient_data", "not_converged"} for size in ("100", "200"))
    details = {
        "training_requests": len(training), "frozen_requests": len(frozen),
        "effective_updates": model_after.get("effective_updates"),
        "snapshot_update_counts": update_counts,
        "model_sha256_before": before_hash, "model_sha256_after": after_hash,
        "fixed_context_scores_before": before_scores, "fixed_context_scores_after": after_scores,
        "max_fixed_context_score_delta": max_score_delta,
    }
    return {"status": "passed" if all(checks.values()) else "failed", "checks": checks, "details": details}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-jsonl", required=True, type=Path)
    parser.add_argument("--frozen-jsonl", required=True, type=Path)
    parser.add_argument("--snapshots-jsonl", required=True, type=Path)
    parser.add_argument("--model-before", required=True, type=Path)
    parser.add_argument("--model-after", required=True, type=Path)
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--environment", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    before_hash, after_hash = sha256(args.model_before), sha256(args.model_after)
    result = validate(load_jsonl(args.training_jsonl), load_jsonl(args.frozen_jsonl), load_jsonl(args.snapshots_jsonl), load_json(args.model_before), load_json(args.model_after), load_json(args.analysis), args.expected_commit, load_json(args.config), before_hash, after_hash)
    result.update({"schema_version": 1, "git_commit": args.expected_commit, "claim_boundary": "facility acceptance only; no performance conclusion", "artifact_sha256": {name: sha256(path) for name, path in {"training": args.training_jsonl, "frozen": args.frozen_jsonl, "snapshots": args.snapshots_jsonl, "model_before": args.model_before, "model_after": args.model_after, "analysis": args.analysis, "config": args.config, "environment": args.environment}.items()}})
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
