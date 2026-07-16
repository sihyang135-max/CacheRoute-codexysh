#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def percentile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def numeric(values: Iterable[Any]) -> List[float]:
    return [float(value) for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]


def stats(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"mean": None, "median": None, "p95": None, "min": None, "max": None}
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


def load_rows(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        row = json.loads(raw)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        rows.append(row)
    return rows


def summarize(path: Path, discard_first: int) -> Dict[str, Any]:
    all_rows = load_rows(path)
    rows = all_rows[discard_first:]
    successful = [row for row in rows if row.get("success") is True]
    ttft_ms = numeric(row.get("client_ttft_ms") for row in successful)
    wall_ms = numeric(row.get("wall_ms") for row in successful)
    completion_tokens = numeric(row.get("completion_tokens") for row in successful)

    starts = numeric(row.get("actual_send_ts") for row in rows)
    ends = numeric(
        float(row["actual_send_ts"]) + float(row["wall_ms"]) / 1000.0
        for row in rows
        if isinstance(row.get("actual_send_ts"), (int, float))
        and isinstance(row.get("wall_ms"), (int, float))
    )
    elapsed_s = max(ends) - min(starts) if starts and ends else None

    traces = [row.get("trace") or {} for row in rows]
    context_build_us = numeric(trace.get("rl_context_build_us") for trace in traces)
    bandit_score_us = numeric(trace.get("rl_bandit_score_us") for trace in traces)
    selected = Counter(str(trace.get("selected_instance_id") or "missing") for trace in traces)
    phases = Counter(str(trace.get("selection_phase") or "missing") for trace in traces)
    reasons = Counter(str(trace.get("selection_reason") or "missing") for trace in traces)
    applied = Counter(str(trace.get("applied_instance_strategy") or "missing") for trace in traces)
    excluded = Counter()
    for trace in traces:
        for reason in (trace.get("linucb_excluded_instances") or {}).values():
            excluded[str(reason)] += 1

    kv_acks = [row.get("kv_ack") or {} for row in rows]
    resident_hit_requests = sum(
        int(ack.get("resident_hit_count") or 0) > 0
        or bool(ack.get("resident_hit_kids"))
        for ack in kv_acks
    )
    total_tokens = sum(completion_tokens)
    return {
        "file": str(path),
        "discard_first": discard_first,
        "requests_total_in_file": len(all_rows),
        "requests_analyzed": len(rows),
        "successful_requests": len(successful),
        "success_rate": len(successful) / len(rows) if rows else None,
        "elapsed_s": elapsed_s,
        "successful_req_per_s": len(successful) / elapsed_s if elapsed_s and elapsed_s > 0 else None,
        "output_token_per_s": total_tokens / elapsed_s if elapsed_s and elapsed_s > 0 and completion_tokens else None,
        "client_ttft_ms": stats(ttft_ms),
        "wall_ms": stats(wall_ms),
        "completion_tokens_total": int(total_tokens) if completion_tokens else None,
        "rl_updated_requests": sum(bool(trace.get("rl_updated")) for trace in traces),
        "rl_scored_requests": sum(trace.get("rl_score_milli") is not None for trace in traces),
        "rl_context_build_us": stats(context_build_us),
        "rl_bandit_score_us": stats(bandit_score_us),
        "selected_instances": dict(sorted(selected.items())),
        "selection_phases": dict(sorted(phases.items())),
        "selection_reasons": dict(sorted(reasons.items())),
        "applied_strategies": dict(sorted(applied.items())),
        "linucb_exclusion_reasons": dict(sorted(excluded.items())),
        "kv_ack_ok_requests": sum(ack.get("ok") is True for ack in kv_acks),
        "kv_resident_hit_requests": resident_hit_requests,
        "kv_resident_hit_count_total": sum(
            int(ack.get("resident_hit_count") or 0) for ack in kv_acks
        ),
        "kv_transfer_requests": sum(
            int(ack.get("payload_bytes") or 0) > 0 or int(ack.get("keys_injected") or 0) > 0
            for ack in kv_acks
        ),
        "kv_payload_bytes_total": sum(int(ack.get("payload_bytes") or 0) for ack in kv_acks),
        "kv_network_queue_ms_total": sum(float(ack.get("network_queue_ms") or 0.0) for ack in kv_acks),
        "kv_network_transfer_ms_total": sum(float(ack.get("network_transfer_ms") or 0.0) for ack in kv_acks),
        "keys_injected_total": sum(int(ack.get("keys_injected") or 0) for ack in kv_acks),
        "missing_meta_requests": sum(bool(row.get("meta_missing")) for row in rows),
        "trace_warning_requests": sum(bool(row.get("trace_warnings")) for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize CacheRoute experiment JSONL files.")
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument(
        "--discard-first",
        type=int,
        default=0,
        help="discard the same number of initial requests from every run",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.discard_first < 0:
        raise ValueError("--discard-first must be >= 0")

    summaries = [summarize(path, args.discard_first) for path in args.files]
    output = json.dumps(summaries, ensure_ascii=False, indent=2)
    print(output)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
