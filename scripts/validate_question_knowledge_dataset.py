"""Validate explicit mappings and optionally write only records safe for later splitting."""

from __future__ import annotations

import argparse
from pathlib import Path

try:  # Supports both `python scripts/...py` and `python -m scripts...`.
    from scripts.dataset_tools import (
        load_excel_records,
        load_tokenizer,
        records_without_blocking_issues,
        validate_records,
        write_json,
        write_jsonl_records,
    )
except ModuleNotFoundError:  # pragma: no cover - exercised by direct CLI use
    from dataset_tools import (
        load_excel_records,
        load_tokenizer,
        records_without_blocking_issues,
        validate_records,
        write_json,
        write_jsonl_records,
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", type=Path, required=True)
    result.add_argument("--knowledge-source", type=Path, required=True)
    result.add_argument("--report", type=Path, required=True)
    result.add_argument("--valid-output", type=Path, help="Optional JSONL output of only error-free records.")
    result.add_argument("--tokenizer-path")
    result.add_argument("--prompt-template-file", type=Path)
    result.add_argument("--near-duplicate-threshold", type=float, default=0.92)
    result.add_argument("--require-tokenization", action="store_true")
    result.add_argument("--allow-invalid", action="store_true", help="Return success after writing an audit report even when source data is invalid.")
    return result


def main() -> int:
    args = parser().parse_args()
    tokenizer = load_tokenizer(args.tokenizer_path)
    template = args.prompt_template_file.read_text(encoding="utf-8") if args.prompt_template_file else None
    records, load_issues = load_excel_records(args.dataset, args.knowledge_source, tokenizer, template)
    issues, near_pairs = validate_records(
        records,
        load_issues,
        args.near_duplicate_threshold,
        args.require_tokenization,
    )
    payload = {
        "valid_records_with_knowledge_text": len(records),
        "error_count": sum(issue.severity == "error" for issue in issues),
        "issues": [issue.to_dict() for issue in issues],
        "near_duplicate_pairs": near_pairs,
    }
    safe_records = records_without_blocking_issues(records, issues)
    payload["safe_record_count"] = len(safe_records)
    write_json(args.report, payload)
    if args.valid_output:
        write_jsonl_records(args.valid_output, safe_records)
    print(
        f"Wrote {args.report}: {payload['error_count']} errors, "
        f"{payload['safe_record_count']} safe records"
    )
    return 0 if args.allow_invalid or not payload["error_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
