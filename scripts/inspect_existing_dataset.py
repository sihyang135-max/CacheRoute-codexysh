"""Inspect explicit question--knowledge mappings without inferring file-order pairs."""

from __future__ import annotations

import argparse
from pathlib import Path

try:  # Supports both `python scripts/...py` and `python -m scripts...`.
    from scripts.dataset_tools import (
        load_excel_records,
        load_tokenizer,
        report_for_records,
        validate_records,
        write_json,
    )
except ModuleNotFoundError:  # pragma: no cover - exercised by direct CLI use
    from dataset_tools import (
        load_excel_records,
        load_tokenizer,
        report_for_records,
        validate_records,
        write_json,
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", type=Path, required=True, help="Excel dataset with explicit row mappings.")
    result.add_argument("--knowledge-source", type=Path, required=True, help="Knowledge .zip or directory of .txt files.")
    result.add_argument("--output", type=Path, required=True, help="JSON report output path.")
    result.add_argument("--tokenizer-path", help="Explicit final 7B tokenizer/model path. No token counts are fabricated when omitted.")
    result.add_argument("--prompt-template-file", type=Path, help="Template containing {knowledge_text} and {question}.")
    result.add_argument("--near-duplicate-threshold", type=float, default=0.92)
    result.add_argument("--target-per-bucket", type=int, default=150)
    return result


def main() -> int:
    args = parser().parse_args()
    tokenizer = load_tokenizer(args.tokenizer_path)
    template = args.prompt_template_file.read_text(encoding="utf-8") if args.prompt_template_file else None
    records, load_issues = load_excel_records(args.dataset, args.knowledge_source, tokenizer, template)
    issues, near_pairs = validate_records(records, load_issues, args.near_duplicate_threshold)
    report = report_for_records(
        records,
        issues,
        near_pairs,
        dataset_path=args.dataset,
        knowledge_source=args.knowledge_source,
        tokenizer_path=args.tokenizer_path,
        prompt_template_path=str(args.prompt_template_file) if args.prompt_template_file else None,
        target_per_bucket=args.target_per_bucket,
    )
    write_json(args.output, report)
    print(f"Wrote {args.output}: {report['valid_records_with_knowledge_text']} valid explicit mappings")
    print(f"Validation errors: {report['validation']['error_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
