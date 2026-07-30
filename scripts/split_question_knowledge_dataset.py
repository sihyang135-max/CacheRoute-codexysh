"""Split a validated normalized dataset without exact or near-duplicate leakage."""

from __future__ import annotations

import argparse
from pathlib import Path

try:  # Supports both `python scripts/...py` and `python -m scripts...`.
    from scripts.dataset_tools import read_jsonl_records, split_records, write_json, write_jsonl_records
except ModuleNotFoundError:  # pragma: no cover - exercised by direct CLI use
    from dataset_tools import read_jsonl_records, split_records, write_json, write_jsonl_records


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--input", type=Path, required=True, help="Validated normalized JSONL input.")
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--seed", type=int, required=True)
    result.add_argument("--train-ratio", type=float, default=0.8)
    result.add_argument("--validation-ratio", type=float, default=0.1)
    result.add_argument("--near-duplicate-threshold", type=float, default=0.92)
    return result


def main() -> int:
    args = parser().parse_args()
    splits = split_records(
        read_jsonl_records(args.input),
        seed=args.seed,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        near_duplicate_threshold=args.near_duplicate_threshold,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, records in splits.items():
        write_jsonl_records(args.output_dir / f"{name}.jsonl", records)
    write_json(
        args.output_dir / "split_report.json",
        {"seed": args.seed, "counts": {name: len(records) for name, records in splits.items()}},
    )
    print("Wrote splits: " + ", ".join(f"{name}={len(records)}" for name, records in splits.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
