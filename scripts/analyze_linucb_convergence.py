#!/usr/bin/env python3
"""Offline 100/200-request diagnostics for Issue #8 (not a performance claim)."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


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


def _mean(values: Iterable[Any]) -> float | None:
    numbers = [float(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return statistics.fmean(numbers) if numbers else None


def _selected_bonus(trace: dict[str, Any]) -> float | None:
    selected = str(trace.get("selected_instance_id") or "")
    bonuses = trace.get("rl_candidate_exploration_bonuses") or {}
    value = bonuses.get(selected) if isinstance(bonuses, dict) else None
    return float(value) if isinstance(value, (int, float)) else None


def summarize_window(rows: list[dict[str, Any]], start: int, size: int) -> dict[str, Any]:
    block = rows[start : start + size]
    traces = [row.get("trace") or {} for row in block]
    counts = Counter(str(t.get("selected_instance_id") or "missing") for t in traces)
    return {
        "start": start,
        "end_exclusive": start + size,
        "requests": size,
        "success_rate": sum(row.get("success") is True for row in block) / size,
        "mean_ttft_ms": _mean(row.get("client_ttft_ms") for row in block if row.get("success") is True),
        "mean_reward": _mean(
            float(t["rl_reward_milli"]) / 1000.0
            for t in traces if isinstance(t.get("rl_reward_milli"), (int, float))
        ),
        "mean_selected_exploration_bonus": _mean(_selected_bonus(t) for t in traces),
        "selection_ratio": {key: value / size for key, value in sorted(counts.items())},
    }


def _relative_change(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return abs(right - left) / max(abs(left), 1e-12)


def assess(windows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(windows) < 3:
        return {"status": "insufficient_data", "reason": "fewer_than_three_complete_windows"}
    recent = windows[-3:]
    ttft_changes = [_relative_change(a["mean_ttft_ms"], b["mean_ttft_ms"]) for a, b in zip(recent, recent[1:])]
    reward_changes = [_relative_change(a["mean_reward"], b["mean_reward"]) for a, b in zip(recent, recent[1:])]
    arms = set().union(*(set(w["selection_ratio"]) for w in recent))
    selection_changes = [
        max(abs(b["selection_ratio"].get(arm, 0.0) - a["selection_ratio"].get(arm, 0.0)) for arm in arms)
        for a, b in zip(recent, recent[1:])
    ]
    finite = all(value is not None and math.isfinite(value) for value in [*ttft_changes, *reward_changes])
    stable = finite and max(ttft_changes) < 0.05 and max(reward_changes) < 0.05 and max(selection_changes) < 0.05
    return {
        "status": "stable_diagnostic" if stable else "not_converged",
        "ttft_relative_changes": ttft_changes,
        "reward_relative_changes": reward_changes,
        "max_selection_ratio_change": max(selection_changes),
        "exploration_boundary": "reported_only_no_numeric_threshold_predefined",
    }


def analyze(rows: list[dict[str, Any]], window_sizes: tuple[int, ...] = (100, 200)) -> dict[str, Any]:
    phases = Counter(str((row.get("trace") or {}).get("selection_phase") or "missing") for row in rows)
    result: dict[str, Any] = {
        "schema_version": 1,
        "claim_boundary": "diagnostic only; no convergence or performance conclusion",
        "requests": len(rows),
        "selection_phases": dict(sorted(phases.items())),
        "windows": {},
    }
    for size in window_sizes:
        windows = [summarize_window(rows, start, size) for start in range(0, len(rows) - size + 1, size)]
        result["windows"][str(size)] = {"complete_windows": len(windows), "series": windows, "assessment": assess(windows)}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests-jsonl", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = analyze(load_jsonl(args.requests_jsonl))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
