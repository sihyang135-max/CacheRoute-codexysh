"""Convert all explicit YAML question--knowledge mappings into candidate JSONL.

This tool does not generate questions or infer pairings from file order.  Each
candidate is created only from fields selected on the same YAML object.
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:  # Supports both `python scripts/...py` and `python -m scripts...`.
    from scripts.dataset_tools import (
        load_tokenizer,
        load_yaml_records,
        records_without_blocking_issues,
        report_for_records,
        validate_records,
        write_json,
        write_jsonl_records,
    )
except ModuleNotFoundError:  # pragma: no cover - exercised by direct CLI use
    from dataset_tools import (
        load_tokenizer,
        load_yaml_records,
        records_without_blocking_issues,
        report_for_records,
        validate_records,
        write_json,
        write_jsonl_records,
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--input", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--report", type=Path, required=True)
    result.add_argument("--items-key", default="knowledge_items")
    result.add_argument("--question-field", default="question")
    result.add_argument("--knowledge-field", default="content")
    result.add_argument("--question-id-field", default="id")
    result.add_argument("--knowledge-id-field", default="id")
    result.add_argument("--source", default="raw_data_yaml")
    result.add_argument("--generation-method", default="existing_explicit_mapping")
    result.add_argument("--tokenizer-path", help="Final 7B tokenizer/model path; omitted means no token counts are fabricated.")
    result.add_argument("--prompt-template-file", type=Path, help="Template containing {knowledge_text} and {question}.")
    result.add_argument("--near-duplicate-threshold", type=float, default=0.92)
    result.add_argument("--target-per-bucket", type=int, default=150)
    return result


def main() -> int:
    args = parser().parse_args()
    tokenizer = load_tokenizer(args.tokenizer_path)
    prompt_template = (
        args.prompt_template_file.read_text(encoding="utf-8")
        if args.prompt_template_file
        else None
    )
    records, load_issues = load_yaml_records(
        args.input,
        items_key=args.items_key,
        question_field=args.question_field,
        knowledge_field=args.knowledge_field,
        question_id_field=args.question_id_field,
        knowledge_id_field=args.knowledge_id_field,
        source=args.source,
        generation_method=args.generation_method,
        tokenizer=tokenizer,
        prompt_template=prompt_template,
    )
    issues, near_pairs = validate_records(
        records, load_issues, args.near_duplicate_threshold
    )
    safe_records = records_without_blocking_issues(records, issues)
    write_jsonl_records(args.output, safe_records)
    report = report_for_records(
        records,
        issues,
        near_pairs,
        dataset_path=args.input,
        knowledge_source=None,
        tokenizer_path=args.tokenizer_path,
        prompt_template_path=(str(args.prompt_template_file) if args.prompt_template_file else None),
        target_per_bucket=args.target_per_bucket,
    )
    report["safe_record_count"] = len(safe_records)
    report["candidate_output"] = str(args.output)
    write_json(args.report, report)
    print(f"wrote {args.output} rows={len(safe_records)}")
    print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
