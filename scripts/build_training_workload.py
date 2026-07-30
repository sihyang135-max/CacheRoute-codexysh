"""Build deterministic Uniform, Hotspot, or Dynamic Hotspot request sequences."""

from __future__ import annotations

import argparse
from pathlib import Path

try:  # Supports both `python scripts/...py` and `python -m scripts...`.
    from scripts.dataset_tools import build_workload, read_jsonl_records, write_json
except ModuleNotFoundError:  # pragma: no cover - exercised by direct CLI use
    from dataset_tools import build_workload, read_jsonl_records, write_json


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--input", type=Path, required=True, help="Validated normalized JSONL input.")
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--distribution", choices=("uniform", "hotspot", "dynamic_hotspot"), required=True)
    result.add_argument("--request-count", type=int, required=True)
    result.add_argument("--seed", type=int, required=True)
    result.add_argument("--hot-fraction", type=float, default=0.2)
    result.add_argument("--hot-probability", type=float, default=0.8)
    result.add_argument("--phase-size", type=int, default=50)
    return result


def main() -> int:
    args = parser().parse_args()
    workload = build_workload(
        read_jsonl_records(args.input),
        request_count=args.request_count,
        distribution=args.distribution,
        seed=args.seed,
        hot_fraction=args.hot_fraction,
        hot_probability=args.hot_probability,
        phase_size=args.phase_size,
    )
    write_json(args.output, workload)
    print(f"Wrote {args.output}: {len(workload['requests'])} requests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
