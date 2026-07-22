#!/usr/bin/env python3
"""Validate Issue #4 phase-1.0 RR/LinUCB signal smoke traces."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        row = json.loads(raw)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        rows.append(row)
    return rows


def _balanced(counts: Counter[str]) -> bool:
    return bool(counts) and max(counts.values()) - min(counts.values()) <= 1


def _all_success(rows: Sequence[Dict[str, Any]]) -> bool:
    return bool(rows) and all(row.get("success") is True for row in rows)


def validate(
    rr_rows: Sequence[Dict[str, Any]],
    linucb_rows: Sequence[Dict[str, Any]],
    expected_speed_factors: Sequence[float],
    warmup_requests: int,
) -> Dict[str, Any]:
    checks: Dict[str, bool] = {}
    details: Dict[str, Any] = {}
    checks["request_count_50_to_100_each"] = all(
        50 <= len(rows) <= 100 for rows in (rr_rows, linucb_rows)
    )
    checks["requests_exceed_warmup"] = len(linucb_rows) > warmup_requests
    checks["all_requests_successful"] = _all_success(rr_rows) and _all_success(linucb_rows)
    checks["no_missing_meta_or_trace_warnings"] = all(
        not row.get("meta_missing") and not row.get("trace_warnings")
        for row in [*rr_rows, *linucb_rows]
    )

    rr_traces = [row.get("trace") or {} for row in rr_rows]
    linucb_traces = [row.get("trace") or {} for row in linucb_rows]
    rr_selected = Counter(str(trace.get("selected_instance_id")) for trace in rr_traces)
    checks["rr_applied_and_balanced"] = (
        len(rr_selected) == 4
        and _balanced(rr_selected)
        and all(trace.get("applied_instance_strategy") == "round_robin" for trace in rr_traces)
    )

    phases = Counter(str(trace.get("selection_phase")) for trace in linucb_traces)
    warmup_traces = [trace for trace in linucb_traces if trace.get("selection_phase") == "warmup"]
    online_traces = [trace for trace in linucb_traces if trace.get("selection_phase") == "linucb"]
    warmup_selected = Counter(
        str(trace.get("selected_instance_id")) for trace in warmup_traces
    )
    checks["warmup_count_and_distribution"] = (
        len(warmup_traces) == warmup_requests
        and len(warmup_selected) == 4
        and _balanced(warmup_selected)
    )
    checks["online_learning_phase_reached"] = bool(online_traces)
    checks["linucb_applied"] = all(
        trace.get("applied_instance_strategy") == "linucb" for trace in linucb_traces
    )

    expected_names = ["bias", "compute_delta_norm", "kv_ready_delta_norm"]
    signal_traces = 0
    observed_ratios: set[float] = set()
    score_consistent = True
    chose_max_score = True
    exploit_nonzero = False
    exploration_positive = False
    exclusions_empty = True
    kv_delta_nonzero_requests = 0
    for trace in linucb_traces:
        features = trace.get("rl_candidate_features") or {}
        costs = trace.get("rl_candidate_costs") or {}
        if trace.get("rl_feature_names") == expected_names and len(features) == 4:
            feature_rows = [value for value in features.values() if isinstance(value, list)]
            if len(feature_rows) == 4 and any(
                len(value) == 3 and float(value[1]) > 0.0 for value in feature_rows
            ):
                signal_traces += 1
            if any(len(value) == 3 and float(value[2]) > 0.0 for value in feature_rows):
                kv_delta_nonzero_requests += 1
        for cost in costs.values():
            if isinstance(cost, dict) and isinstance(
                cost.get("compute_capacity_ratio"), (int, float)
            ):
                observed_ratios.add(round(float(cost["compute_capacity_ratio"]), 4))
        if trace.get("linucb_excluded_instances"):
            exclusions_empty = False

    for trace in online_traces:
        scores = trace.get("rl_candidate_scores") or {}
        exploits = trace.get("rl_candidate_exploit_scores") or {}
        bonuses = trace.get("rl_candidate_exploration_bonuses") or {}
        chosen = str(trace.get("selected_instance_id"))
        if not scores or set(scores) != set(exploits) or set(scores) != set(bonuses):
            score_consistent = False
            chose_max_score = False
            continue
        for instance_id, score in scores.items():
            expected_score = float(exploits[instance_id]) + float(bonuses[instance_id])
            if not math.isclose(float(score), expected_score, rel_tol=1e-7, abs_tol=1e-9):
                score_consistent = False
            exploit_nonzero = exploit_nonzero or not math.isclose(
                float(exploits[instance_id]), 0.0, abs_tol=1e-12
            )
            exploration_positive = exploration_positive or float(bonuses[instance_id]) > 0.0
        if chosen not in scores or float(scores[chosen]) < max(map(float, scores.values())) - 1e-9:
            chose_max_score = False

    required_signal_traces = max(1, math.ceil(len(linucb_traces) * 0.9))
    checks["compute_context_signal_visible"] = signal_traces >= required_signal_traces
    checks["registered_speed_factors_match_calibration"] = observed_ratios == {
        round(float(value), 4) for value in expected_speed_factors
    }
    checks["candidate_scores_equal_exploit_plus_bonus"] = score_consistent
    checks["online_choice_follows_total_score"] = chose_max_score
    checks["exploit_and_exploration_visible"] = exploit_nonzero and exploration_positive
    checks["no_safety_candidate_exclusions"] = exclusions_empty
    updated_count = sum(bool(trace.get("rl_updated")) for trace in linucb_traces)
    effective_updates = [
        int(trace["rl_effective_updates"])
        for trace in linucb_traces
        if isinstance(trace.get("rl_effective_updates"), int)
    ]
    checks["updates_visible_and_monotonic"] = (
        updated_count == len(linucb_rows)
        and bool(effective_updates)
        and effective_updates == sorted(effective_updates)
        and max(effective_updates) >= warmup_requests
    )

    details.update({
        "rr_selected_instances": dict(sorted(rr_selected.items())),
        "linucb_selection_phases": dict(sorted(phases.items())),
        "linucb_warmup_selected_instances": dict(sorted(warmup_selected.items())),
        "linucb_all_selected_instances": dict(sorted(Counter(
            str(trace.get("selected_instance_id")) for trace in linucb_traces
        ).items())),
        "observed_legacy_compute_capacity_ratio_fields": sorted(observed_ratios),
        "expected_relative_isolated_prefill_speed_factors": [
            round(float(value), 4) for value in expected_speed_factors
        ],
        "compute_signal_requests": signal_traces,
        "compute_signal_requests_required": required_signal_traces,
        "kv_delta_nonzero_requests": kv_delta_nonzero_requests,
        "kv_delta_interpretation": (
            "nonzero instance differences observed"
            if kv_delta_nonzero_requests
            else "zero is valid for shared Redis/global scope"
        ),
        "linucb_updated_requests": updated_count,
        "max_effective_updates_before_selection": max(effective_updates, default=None),
    })
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rr-jsonl", required=True, type=Path)
    parser.add_argument("--linucb-jsonl", required=True, type=Path)
    parser.add_argument("--expected-speed-factors", required=True)
    parser.add_argument("--warmup-requests", required=True, type=int)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--environment", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    expected_speed_factors = [
        float(item) for item in args.expected_speed_factors.split(",")
    ]
    if len(expected_speed_factors) != 4 or args.warmup_requests < 0:
        raise ValueError("four expected speed factors and a non-negative warmup are required")
    result = validate(
        load_jsonl(args.rr_jsonl),
        load_jsonl(args.linucb_jsonl),
        expected_speed_factors,
        args.warmup_requests,
    )
    result.update({
        "schema_version": 1,
        "experiment_kind": "rl4_rr_linucb_signal_smoke",
        "git_commit": args.expected_commit,
        "artifact_sha256": {
            "rr_raw_jsonl": sha256_file(args.rr_jsonl),
            "linucb_raw_jsonl": sha256_file(args.linucb_jsonl),
            "config": sha256_file(args.config),
            "environment": sha256_file(args.environment),
            "calibration_aggregate": sha256_file(args.calibration),
        },
        "claim_boundary": "signal validation only; no performance conclusion",
    })
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
