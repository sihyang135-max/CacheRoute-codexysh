#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


def load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{number}: expected a JSON object")
        rows.append(value)
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: Iterable[float], fraction: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def ttft_summary(rows: List[Dict[str, Any]]) -> Dict[str, float | None]:
    values = [
        float(row["client_ttft_ms"])
        for row in rows
        if row.get("success") is True and isinstance(row.get("client_ttft_ms"), (int, float))
    ]
    return {
        "mean_ms": statistics.fmean(values) if values else None,
        "median_ms": statistics.median(values) if values else None,
        "p95_ms": percentile(values, 0.95),
    }


def improvement(rr: float | None, linucb: float | None) -> float | None:
    if rr is None or linucb is None or rr == 0:
        return None
    return (rr - linucb) / rr


def validate(
    config: Dict[str, Any],
    order: List[Dict[str, str]],
    rows_by_file: Dict[str, List[Dict[str, Any]]],
    model_before: Dict[str, Any],
    model_after: Dict[str, Any],
    model_hash_before: str,
    model_hash_after: str,
    expected_commit: str,
) -> Dict[str, Any]:
    repeats = int(config.get("repeats", 0))
    requests = int(config.get("measure_requests", 0))
    by_pair: Dict[int, List[Dict[str, str]]] = defaultdict(list)
    for entry in order:
        by_pair[int(entry["pair"])].append(entry)

    expected_pairs = list(range(1, repeats + 1))
    schedule_ok = list(sorted(by_pair)) == expected_pairs
    seeds_ok = True
    policies_ok = True
    row_counts_ok = True
    success_ok = True
    strategy_trace_ok = True
    frozen_markers_ok = True
    frozen_never_updates = True
    paired: List[Dict[str, Any]] = []

    for pair in expected_pairs:
        entries = by_pair.get(pair, [])
        expected_policy_order = (
            ["round_robin", "linucb"] if pair % 2 == 1 else ["linucb", "round_robin"]
        )
        if [entry.get("strategy") for entry in entries] != expected_policy_order:
            schedule_ok = False
        if len(entries) != 2 or len({entry.get("seed") for entry in entries}) != 1:
            seeds_ok = False
        if {entry.get("strategy") for entry in entries} != {"round_robin", "linucb"}:
            policies_ok = False

        summaries: Dict[str, Dict[str, float | None]] = {}
        for entry in entries:
            strategy = entry.get("strategy", "")
            rows = rows_by_file.get(entry.get("raw_file", ""), [])
            if len(rows) != requests:
                row_counts_ok = False
            if not rows or any(row.get("success") is not True for row in rows):
                success_ok = False
            summaries[strategy] = ttft_summary(rows)
            for row in rows:
                trace = row.get("trace") or {}
                applied = trace.get("applied_instance_strategy")
                if applied != strategy:
                    strategy_trace_ok = False
                if strategy == "linucb":
                    if trace.get("rl_model_frozen") not in (1, True):
                        frozen_markers_ok = False
                    if trace.get("rl_runtime_mode") != "loaded-frozen":
                        frozen_markers_ok = False
                    if trace.get("rl_updated") not in (0, False):
                        frozen_never_updates = False

        rr = summaries.get("round_robin", {})
        linucb = summaries.get("linucb", {})
        paired.append({
            "pair": pair,
            "order": "rr-first" if pair % 2 == 1 else "linucb-first",
            "seed": entries[0].get("seed") if entries else None,
            "round_robin": rr,
            "frozen_linucb": linucb,
            "relative_improvement": {
                "mean": improvement(rr.get("mean_ms"), linucb.get("mean_ms")),
                "median": improvement(rr.get("median_ms"), linucb.get("median_ms")),
                "p95": improvement(rr.get("p95_ms"), linucb.get("p95_ms")),
            },
        })

    checks = {
        "config_commit_matches": config.get("git_commit") == expected_commit,
        "concurrency_is_one": config.get("concurrency") == 1,
        "global_kv_scope_declared": (
            config.get("kv_residency_scope") == "global"
            and config.get("kv_link_scope") == "global"
        ),
        "complete_alternating_schedule": schedule_ok and len(order) == repeats * 2,
        "same_seed_within_each_pair": seeds_ok,
        "exactly_rr_and_linucb_per_pair": policies_ok,
        "request_counts_match": row_counts_ok,
        "all_requests_succeeded": success_ok,
        "applied_strategy_matches": strategy_trace_ok,
        "linucb_is_loaded_frozen": frozen_markers_ok,
        "frozen_linucb_never_updates": frozen_never_updates,
        "model_hash_unchanged": model_hash_before == model_hash_after,
        "model_source_commit_matches": (
            model_before.get("source_commit") == expected_commit
            and model_after.get("source_commit") == expected_commit
        ),
        "model_has_training_updates": int(model_before.get("effective_updates", 0))
        >= int(config.get("training_requests", 0)),
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "model_sha256_before": model_hash_before,
        "model_sha256_after": model_hash_after,
        "paired_ttft": paired,
        "claim_boundary": (
            "descriptive paired TTFT output; use experiment pairs as independent repeats; "
            "global KV scope does not support an instance-local KV-affinity claim"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and summarize paired RR vs frozen-LinUCB results."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-order", type=Path, required=True)
    parser.add_argument("--model-before", type=Path, required=True)
    parser.add_argument("--model-after", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.run_order.open(encoding="utf-8", newline="") as handle:
        order = list(csv.DictReader(handle, delimiter="\t"))
    rows_by_file = {
        entry["raw_file"]: load_jsonl(Path(entry["raw_file"]))
        for entry in order
    }
    result = validate(
        load_json(args.config),
        order,
        rows_by_file,
        load_json(args.model_before),
        load_json(args.model_after),
        sha256(args.model_before),
        sha256(args.model_after),
        args.expected_commit,
    )
    output = json.dumps(result, ensure_ascii=False, indent=2)
    print(output)
    args.output.write_text(output + "\n", encoding="utf-8")
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
