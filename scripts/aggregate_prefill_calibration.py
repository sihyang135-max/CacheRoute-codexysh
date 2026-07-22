#!/usr/bin/env python3
"""Aggregate independent Prefill calibration rounds with reproducibility hashes."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def coefficient_of_variation(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = statistics.fmean(values)
    if mean == 0.0:
        raise ValueError("cannot compute CV for a zero-mean sequence")
    return statistics.stdev(values) / abs(mean)


def percentile_nearest_rank(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("cannot compute percentile of an empty sequence")
    ordered = sorted(float(value) for value in values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


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


def start_skews_ms(
    rows: Sequence[Dict[str, Any]], expected_ports: Sequence[int]
) -> List[float]:
    groups: Dict[tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("phase") != "measured":
            continue
        groups[(str(row.get("length_label")), int(row["prompt_index"]))].append(row)
    skews = []
    expected = {int(port) for port in expected_ports}
    for group_key, group in groups.items():
        ports = {int(row["port"]) for row in group}
        starts = [int(row["request_started_at_unix_ns"]) for row in group]
        if ports != expected or len(group) != len(expected):
            raise ValueError(f"incomplete concurrent request group: {group_key}")
        skews.append((max(starts) - min(starts)) / 1_000_000.0)
    if not skews:
        raise ValueError("no measured concurrent request groups found")
    return skews


def _same_protocol(summaries: Sequence[Dict[str, Any]], key: str) -> Any:
    if any(key not in summary for summary in summaries):
        raise ValueError(f"calibration summary is missing protocol field {key}")
    first = summaries[0].get(key)
    if any(summary.get(key) != first for summary in summaries[1:]):
        raise ValueError(f"calibration protocol mismatch for {key}")
    return first


def aggregate(
    run_dirs: Sequence[Path], max_cv: float, max_start_skew_p95_ms: float = 5.0
) -> Dict[str, Any]:
    if len(run_dirs) != 3:
        raise ValueError("exactly three independent calibration run directories are required")
    summaries: List[Dict[str, Any]] = []
    request_rows: List[List[Dict[str, Any]]] = []
    source_files: List[Dict[str, Any]] = []
    for run_dir in run_dirs:
        summary_path = run_dir / "summary.json"
        requests_path = run_dir / "requests.jsonl"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if not isinstance(summary, dict):
            raise ValueError(f"{summary_path}: expected a JSON object")
        summaries.append(summary)
        request_rows.append(load_jsonl(requests_path))
        source_files.append({
            "run_dir": str(run_dir),
            "summary_sha256": sha256_file(summary_path),
            "requests_sha256": sha256_file(requests_path),
        })

    if any(int(summary.get("failures", -1)) != 0 for summary in summaries):
        raise ValueError("calibration aggregation refuses rounds with request failures")

    protocol_keys = (
        "git_commit",
        "model",
        "ports",
        "tp_sizes",
        "length_labels",
        "prompt_repetitions",
        "warmup_per_instance",
        "measured_per_instance",
        "parallel_instances_per_prompt",
    )
    protocol = {key: _same_protocol(summaries, key) for key in protocol_keys}
    seeds = [int(summary["seed"]) for summary in summaries]
    run_ids = [str(summary["run_id"]) for summary in summaries]
    if len(set(seeds)) != 3 or len(set(run_ids)) != 3:
        raise ValueError("calibration rounds must have unique seeds and run IDs")
    ports = [str(port) for port in protocol["ports"]]
    labels = [str(label) for label in protocol["length_labels"]]
    by_length: Dict[str, Any] = {}
    cv_values: List[float] = []
    for label in labels:
        per_port: Dict[str, Any] = {}
        for port in ports:
            values = [
                float(summary["by_length"][label][
                    "capacity_ratios_relative_to_mean_tp1_median"
                ][port])
                for summary in summaries
            ]
            cv = coefficient_of_variation(values)
            cv_values.append(cv)
            raw_ttft_medians = [
                float(summary["by_length"][label]["ports"][port]["ttft_median_ms"])
                for summary in summaries
            ]
            per_port[port] = {
                "relative_isolated_prefill_speed_factor_round_values": values,
                "relative_isolated_prefill_speed_factor_median": round(
                    statistics.median(values), 4
                ),
                "relative_isolated_prefill_speed_factor_cv": round(cv, 6),
                "raw_ttft_median_ms_round_values": raw_ttft_medians,
                "raw_ttft_median_ms_across_rounds": round(
                    statistics.median(raw_ttft_medians), 3
                ),
                "raw_ttft_median_ms_cv": round(
                    coefficient_of_variation(raw_ttft_medians), 6
                ),
            }
        by_length[label] = per_port

    per_round_scalars: List[Dict[str, float]] = []
    for summary in summaries:
        per_round_scalars.append({
            port: float(summary["equal_weight_median_capacity_ratios"][port])
            for port in ports
        })
    diagnostic_speed_factors = {
        port: round(statistics.median([row[port] for row in per_round_scalars]), 4)
        for port in ports
    }
    speed_factor_cv = {
        port: round(
            coefficient_of_variation([row[port] for row in per_round_scalars]), 6
        )
        for port in ports
    }
    cv_values.extend(speed_factor_cv.values())

    per_round_start_skews = [
        start_skews_ms(rows, protocol["ports"]) for rows in request_rows
    ]
    all_start_skews = [value for values in per_round_start_skews for value in values]
    start_skew_p95_ms = percentile_nearest_rank(all_start_skews, 0.95)
    start_skew = {
        "measured_prompt_groups": len(all_start_skews),
        "median_ms": round(statistics.median(all_start_skews), 3),
        "p95_ms": round(start_skew_p95_ms, 3),
        "max_ms": round(max(all_start_skews), 3),
        "p95_threshold_ms": max_start_skew_p95_ms,
        "passed": start_skew_p95_ms <= max_start_skew_p95_ms,
        "per_round": [
            {
                "run_id": run_id,
                "measured_prompt_groups": len(values),
                "median_ms": round(statistics.median(values), 3),
                "p95_ms": round(percentile_nearest_rank(values, 0.95), 3),
                "max_ms": round(max(values), 3),
            }
            for run_id, values in zip(run_ids, per_round_start_skews)
        ],
    }

    input_summary = {
        "protocol": protocol,
        "rounds": [
            {"run_id": run_id, "seed": seed}
            for run_id, seed in zip(run_ids, seeds)
        ],
    }
    observed_max_cv = max(cv_values)
    passed = observed_max_cv < max_cv and start_skew["passed"]
    return {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "git_commit": protocol["git_commit"],
        "round_count": 3,
        "seeds": seeds,
        "input_summary": input_summary,
        "input_summary_sha256": canonical_sha256(input_summary),
        "source_files": source_files,
        "by_length": by_length,
        "per_round_cross_length_median_speed_factors": per_round_scalars,
        "diagnostic_relative_isolated_prefill_speed_factors": diagnostic_speed_factors,
        "diagnostic_speed_factors_csv": ",".join(
            f"{diagnostic_speed_factors[port]:.4f}" for port in ports
        ),
        "diagnostic_speed_factor_cv": speed_factor_cv,
        "max_relative_speed_factor_cv": round(observed_max_cv, 6),
        "max_relative_speed_factor_cv_threshold": max_cv,
        "concurrent_http_start_skew_ms": start_skew,
        "terminology": (
            "relative isolated-prefill speed factor; not throughput or physical capacity"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate three independent Prefill calibration result directories."
    )
    parser.add_argument("run_dirs", nargs=3, type=Path)
    parser.add_argument("--max-cv", type=float, default=0.05)
    parser.add_argument("--max-start-skew-p95-ms", type=float, default=5.0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not 0.0 < args.max_cv < 1.0:
        raise ValueError("--max-cv must be between 0 and 1")
    if args.max_start_skew_p95_ms <= 0.0:
        raise ValueError("--max-start-skew-p95-ms must be positive")
    result = aggregate(
        args.run_dirs, args.max_cv, args.max_start_skew_p95_ms
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
