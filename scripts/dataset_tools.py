"""Shared dataset utilities for Phase B experiment preparation.

The functions in this module deliberately treat an explicit row-level mapping as
the only authoritative question--knowledge relationship.  File order is never
used to infer a mapping.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from pathlib import Path
import random
import re
from typing import Any, Iterable
import zipfile


LENGTH_BUCKETS = (
    ("short", 256, 512),
    ("medium", 512, 1024),
    ("long", 1024, 2048),
)


def canonical_text(value: str) -> str:
    """Normalize text for hashes and duplicate checks without changing source text."""
    return " ".join(value.split()).strip().casefold()


def content_hash(value: str) -> str:
    return sha256(canonical_text(value).encode("utf-8")).hexdigest()


def classify_length(token_count: int | None) -> str | None:
    if token_count is None:
        return None
    for name, lower, upper in LENGTH_BUCKETS:
        if lower <= token_count <= upper:
            return name
    return "out_of_scope"


@dataclass
class DatasetRecord:
    question_id: str
    knowledge_id: str
    question: str
    knowledge_text: str
    source: str
    generation_method: str
    mapping_evidence: str
    source_path: str
    content_hash: str
    question_tokens: int | None = None
    knowledge_tokens: int | None = None
    prompt_tokens: int | None = None
    length_bucket: str | None = None
    source_metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationIssue:
    severity: str
    code: str
    message: str
    question_ids: list[str] = field(default_factory=list)
    knowledge_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def require_openpyxl():
    try:
        import openpyxl
    except ImportError as exc:  # pragma: no cover - depends on caller environment
        raise RuntimeError(
            "Reading .xlsx input requires openpyxl. Install openpyxl before running "
            "the Phase B dataset tools."
        ) from exc
    return openpyxl


def load_tokenizer(tokenizer_path: str | Path | None):
    """Load the final model tokenizer only when the caller explicitly provides it."""
    if not tokenizer_path:
        return None
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - depends on caller environment
        raise RuntimeError(
            "--tokenizer-path was supplied, but transformers is not installed."
        ) from exc
    return AutoTokenizer.from_pretrained(
        str(tokenizer_path), local_files_only=True, use_fast=True
    )


def token_count(tokenizer: Any, text: str) -> int | None:
    if tokenizer is None:
        return None
    return len(tokenizer.encode(text, add_special_tokens=False))


def render_prompt(template: str | None, knowledge_text: str, question: str) -> str | None:
    if template is None:
        return None
    try:
        return template.format(knowledge_text=knowledge_text, question=question)
    except KeyError as exc:
        raise ValueError(
            "Prompt template may only use {knowledge_text} and {question}."
        ) from exc


def _read_knowledge_source(source: Path | None) -> dict[str, str]:
    if source is None:
        return {}
    if source.is_dir():
        return {
            path.stem: path.read_text(encoding="utf-8")
            for path in sorted(source.glob("*.txt"))
        }
    if source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as archive:
            return {
                Path(member).stem: archive.read(member).decode("utf-8")
                for member in sorted(archive.namelist())
                if member.lower().endswith(".txt")
            }
    raise ValueError("Knowledge source must be a directory of .txt files or a .zip archive.")


def _required_columns(header: dict[str, int]) -> None:
    required = {
        "sample_id",
        "dataset_source",
        "kdn_reference_document_id",
        "user_question",
    }
    missing = sorted(required - set(header))
    if missing:
        raise ValueError("Excel sheet is missing required columns: " + ", ".join(missing))


def load_excel_records(
    dataset_path: Path,
    knowledge_source: Path | None,
    tokenizer: Any = None,
    prompt_template: str | None = None,
) -> tuple[list[DatasetRecord], list[ValidationIssue]]:
    """Load explicit mappings from an Excel row plus its referenced knowledge file."""
    openpyxl = require_openpyxl()
    workbook = openpyxl.load_workbook(dataset_path, read_only=True, data_only=True)
    records: list[DatasetRecord] = []
    issues: list[ValidationIssue] = []
    knowledge_by_id = _read_knowledge_source(knowledge_source)

    try:
        for sheet in workbook.worksheets:
            rows = sheet.iter_rows(values_only=True)
            try:
                first_row = next(rows)
            except StopIteration:
                continue
            header = {
                str(value).strip(): index
                for index, value in enumerate(first_row)
                if value is not None and str(value).strip()
            }
            if not header:
                continue
            _required_columns(header)

            for row_number, row in enumerate(rows, start=2):
                def cell(name: str) -> str:
                    index = header[name]
                    value = row[index] if index < len(row) else None
                    return "" if value is None else str(value).strip()

                question_id = cell("sample_id")
                knowledge_id = cell("kdn_reference_document_id")
                question = cell("user_question")
                if not any((question_id, knowledge_id, question)):
                    continue
                evidence = f"{dataset_path.name}:{sheet.title}!{row_number}"
                if not question_id or not knowledge_id or not question:
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            code="missing_explicit_mapping_field",
                            message=(
                                "A row is missing sample_id, kdn_reference_document_id, or "
                                "user_question; no relationship was inferred."
                            ),
                            question_ids=[question_id] if question_id else [],
                            knowledge_ids=[knowledge_id] if knowledge_id else [],
                        )
                    )
                    continue
                knowledge_text = knowledge_by_id.get(knowledge_id)
                if knowledge_text is None:
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            code="knowledge_content_missing",
                            message=(
                                f"Explicit mapping {evidence} references {knowledge_id}, but no "
                                "matching knowledge text was found in the configured source."
                            ),
                            question_ids=[question_id],
                            knowledge_ids=[knowledge_id],
                        )
                    )
                    continue
                prompt = render_prompt(prompt_template, knowledge_text, question)
                source_metrics: dict[str, Any] = {}
                for column in ("question_len", "knowledge_len", "actual_doc_len", "Topic"):
                    if column in header:
                        source_metrics[column] = row[header[column]]
                knowledge_tokens = token_count(tokenizer, knowledge_text)
                records.append(
                    DatasetRecord(
                        question_id=question_id,
                        knowledge_id=knowledge_id,
                        question=question,
                        knowledge_text=knowledge_text,
                        source=cell("dataset_source") or "unknown",
                        generation_method="repository_existing",
                        mapping_evidence=evidence,
                        source_path=str(dataset_path),
                        content_hash=content_hash(knowledge_text),
                        question_tokens=token_count(tokenizer, question),
                        knowledge_tokens=knowledge_tokens,
                        prompt_tokens=token_count(tokenizer, prompt) if prompt is not None else None,
                        length_bucket=classify_length(knowledge_tokens),
                        source_metrics=source_metrics,
                    )
                )
    finally:
        workbook.close()
    return records, issues


def load_yaml_records(
    dataset_path: Path,
    *,
    items_key: str,
    question_field: str,
    knowledge_field: str,
    question_id_field: str,
    knowledge_id_field: str,
    source: str,
    generation_method: str,
    tokenizer: Any = None,
    prompt_template: str | None = None,
) -> tuple[list[DatasetRecord], list[ValidationIssue]]:
    """Load one explicit question--knowledge mapping from each YAML item.

    The caller must name the field that carries the question.  This makes any
    mapping from a legacy field such as ``text`` deliberate and auditable.
    """
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on caller environment
        raise RuntimeError("Reading YAML input requires PyYAML.") from exc
    payload = yaml.safe_load(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get(items_key), list):
        raise ValueError(f"Expected a top-level list at {items_key!r} in {dataset_path}.")
    records: list[DatasetRecord] = []
    issues: list[ValidationIssue] = []
    for index, item in enumerate(payload[items_key], start=1):
        evidence = f"{dataset_path.name}:{items_key}[{index - 1}]"
        if not isinstance(item, dict):
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="invalid_yaml_item",
                    message=f"{evidence} must be an object.",
                )
            )
            continue

        def value(name: str) -> str:
            raw = item.get(name)
            return "" if raw is None else str(raw).strip()

        question_id = value(question_id_field)
        knowledge_id = value(knowledge_id_field)
        question = value(question_field)
        knowledge_text = value(knowledge_field)
        if not all((question_id, knowledge_id, question, knowledge_text)):
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="missing_explicit_mapping_field",
                    message=(
                        f"{evidence} is missing one of {question_id_field}, "
                        f"{knowledge_id_field}, {question_field}, or {knowledge_field}."
                    ),
                    question_ids=[question_id] if question_id else [],
                    knowledge_ids=[knowledge_id] if knowledge_id else [],
                )
            )
            continue
        prompt = render_prompt(prompt_template, knowledge_text, question)
        knowledge_tokens = token_count(tokenizer, knowledge_text)
        records.append(
            DatasetRecord(
                question_id=question_id,
                knowledge_id=knowledge_id,
                question=question,
                knowledge_text=knowledge_text,
                source=source,
                generation_method=generation_method,
                mapping_evidence=(
                    f"{evidence} ({question_field} -> {knowledge_field})"
                ),
                source_path=str(dataset_path),
                content_hash=content_hash(knowledge_text),
                question_tokens=token_count(tokenizer, question),
                knowledge_tokens=knowledge_tokens,
                prompt_tokens=token_count(tokenizer, prompt) if prompt is not None else None,
                length_bucket=classify_length(knowledge_tokens),
                source_metrics={"raw_record_index": index - 1},
            )
        )
    return records, issues


def read_jsonl_records(path: Path) -> list[DatasetRecord]:
    records: list[DatasetRecord] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            try:
                records.append(DatasetRecord(**payload))
            except TypeError as exc:
                raise ValueError(f"Invalid record at {path}:{line_number}: {exc}") from exc
    return records


def write_jsonl_records(path: Path, records: Iterable[DatasetRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _duplicate_issues(records: list[DatasetRecord]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for field_name, code in (("question_id", "duplicate_question_id"), ("knowledge_id", "duplicate_knowledge_id")):
        groups: dict[str, list[DatasetRecord]] = {}
        for record in records:
            groups.setdefault(getattr(record, field_name), []).append(record)
        for value, group in groups.items():
            if len(group) > 1:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code=code,
                        message=f"{field_name} {value!r} is mapped by more than one record.",
                        question_ids=[item.question_id for item in group],
                        knowledge_ids=[item.knowledge_id for item in group],
                    )
                )
    return issues


def _exact_content_issues(records: list[DatasetRecord]) -> list[ValidationIssue]:
    groups: dict[str, list[DatasetRecord]] = {}
    for record in records:
        groups.setdefault(record.content_hash, []).append(record)
    return [
        ValidationIssue(
            severity="error",
            code="exact_duplicate_knowledge",
            message="More than one record contains the same canonical knowledge content.",
            question_ids=[item.question_id for item in group],
            knowledge_ids=[item.knowledge_id for item in group],
        )
        for group in groups.values()
        if len(group) > 1
    ]


def _word_shingles(value: str, width: int = 3) -> set[tuple[str, ...]]:
    words = re.findall(r"\w+", canonical_text(value))
    if len(words) < width:
        return {tuple(words)} if words else set()
    return {tuple(words[index : index + width]) for index in range(len(words) - width + 1)}


def near_duplicate_pairs(
    records: list[DatasetRecord], threshold: float = 0.92
) -> list[tuple[str, str, float]]:
    """Return deterministic near-duplicate knowledge pairs.

    Similarity is Jaccard overlap of normalized word trigrams.  This keeps the
    detector transparent and fast enough for the 450-record target while still
    flagging material copy/edit variants.
    """
    pairs: list[tuple[str, str, float]] = []
    ordered = sorted(records, key=lambda item: item.knowledge_id)
    shingles = {record.knowledge_id: _word_shingles(record.knowledge_text) for record in ordered}
    for index, left in enumerate(ordered):
        left_shingles = shingles[left.knowledge_id]
        for right in ordered[index + 1 :]:
            if left.content_hash == right.content_hash:
                continue
            right_shingles = shingles[right.knowledge_id]
            union = left_shingles | right_shingles
            similarity = len(left_shingles & right_shingles) / len(union) if union else 0.0
            if similarity >= threshold:
                pairs.append((left.knowledge_id, right.knowledge_id, round(similarity, 6)))
    return pairs


def validate_records(
    records: list[DatasetRecord],
    initial_issues: Iterable[ValidationIssue] = (),
    near_duplicate_threshold: float = 0.92,
    require_tokenization: bool = False,
) -> tuple[list[ValidationIssue], list[tuple[str, str, float]]]:
    issues = list(initial_issues)
    for record in records:
        for field_name in (
            "question_id",
            "knowledge_id",
            "question",
            "knowledge_text",
            "source",
            "generation_method",
        ):
            if not getattr(record, field_name).strip():
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="empty_required_field",
                        message=f"{field_name} is empty.",
                        question_ids=[record.question_id] if record.question_id else [],
                        knowledge_ids=[record.knowledge_id] if record.knowledge_id else [],
                    )
                )
        if record.content_hash != content_hash(record.knowledge_text):
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="content_hash_mismatch",
                    message="content_hash does not match canonical knowledge_text.",
                    question_ids=[record.question_id],
                    knowledge_ids=[record.knowledge_id],
                )
            )
        if require_tokenization and (
            record.question_tokens is None
            or record.knowledge_tokens is None
            or record.prompt_tokens is None
        ):
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="missing_final_tokenizer_counts",
                    message=(
                        "Final-model tokenizer counts for question, knowledge, and prompt "
                        "are required but unavailable."
                    ),
                    question_ids=[record.question_id],
                    knowledge_ids=[record.knowledge_id],
                )
            )
    issues.extend(_duplicate_issues(records))
    issues.extend(_exact_content_issues(records))
    near_pairs = near_duplicate_pairs(records, near_duplicate_threshold)
    for left, right, similarity in near_pairs:
        issues.append(
            ValidationIssue(
                severity="error",
                code="near_duplicate_knowledge",
                message=(
                    f"Knowledge documents {left} and {right} have canonical text "
                    f"similarity {similarity}."
                ),
                knowledge_ids=[left, right],
            )
        )
    return issues, near_pairs


def records_without_blocking_issues(
    records: list[DatasetRecord], issues: Iterable[ValidationIssue]
) -> list[DatasetRecord]:
    """Keep records unaffected by a record-specific validation error.

    This lets Phase B preserve valid explicit rows while reporting incomplete
    source rows. A global error with no affected IDs blocks every output record.
    """
    blocked_questions: set[str] = set()
    blocked_knowledge: set[str] = set()
    for issue in issues:
        if issue.severity != "error":
            continue
        if not issue.question_ids and not issue.knowledge_ids:
            return []
        blocked_questions.update(issue.question_ids)
        blocked_knowledge.update(issue.knowledge_ids)
    return [
        record
        for record in records
        if record.question_id not in blocked_questions
        and record.knowledge_id not in blocked_knowledge
    ]


def report_for_records(
    records: list[DatasetRecord],
    issues: list[ValidationIssue],
    near_pairs: list[tuple[str, str, float]],
    *,
    dataset_path: Path,
    knowledge_source: Path | None,
    tokenizer_path: str | None,
    prompt_template_path: str | None,
    target_per_bucket: int = 150,
) -> dict[str, Any]:
    bucket_counts = {name: 0 for name, _, _ in LENGTH_BUCKETS}
    out_of_scope = 0
    unknown_tokens = 0
    for record in records:
        if record.length_bucket in bucket_counts:
            bucket_counts[record.length_bucket] += 1
        elif record.length_bucket is None:
            unknown_tokens += 1
        else:
            out_of_scope += 1
    errors = [item for item in issues if item.severity == "error"]
    warnings = [item for item in issues if item.severity != "error"]
    return {
        "schema_version": 1,
        "dataset_path": str(dataset_path),
        "knowledge_source": str(knowledge_source) if knowledge_source else None,
        "mapping_policy": "explicit_row_mapping_only",
        "valid_records_with_knowledge_text": len(records),
        "tokenization": {
            "status": "computed" if tokenizer_path else "not_requested",
            "tokenizer_path": tokenizer_path,
            "prompt_template_path": prompt_template_path,
            "records_without_final_token_counts": unknown_tokens,
        },
        "length_buckets": {
            **bucket_counts,
            "out_of_scope": out_of_scope,
            "unknown": unknown_tokens,
            "target_per_bucket": target_per_bucket,
            "gaps": {
                name: max(target_per_bucket - count, 0)
                for name, count in bucket_counts.items()
            },
        },
        "validation": {
            "error_count": len(errors),
            "warning_count": len(warnings),
            "issues": [item.to_dict() for item in issues],
            "near_duplicate_pairs": [
                {"left_knowledge_id": left, "right_knowledge_id": right, "similarity": similarity}
                for left, right, similarity in near_pairs
            ],
        },
        "records": [record.to_dict() for record in records],
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class _UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        root = self.parent[value]
        while root != self.parent[root]:
            root = self.parent[root]
        while value != root:
            parent = self.parent[value]
            self.parent[value] = root
            value = parent
        return root

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def split_records(
    records: list[DatasetRecord],
    *,
    seed: int,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    near_duplicate_threshold: float = 0.92,
) -> dict[str, list[DatasetRecord]]:
    if not 0 < train_ratio < 1 or not 0 < validation_ratio < 1 or train_ratio + validation_ratio >= 1:
        raise ValueError("train_ratio and validation_ratio must be positive and sum to less than 1.")
    issues, near_pairs = validate_records(records, near_duplicate_threshold=near_duplicate_threshold)
    blocking = [issue for issue in issues if issue.code in {"duplicate_question_id", "duplicate_knowledge_id"}]
    if blocking:
        raise ValueError("Cannot split records with non-unique explicit mappings.")
    union_find = _UnionFind(record.knowledge_id for record in records)
    by_hash: dict[str, list[DatasetRecord]] = {}
    for record in records:
        by_hash.setdefault(record.content_hash, []).append(record)
    for group in by_hash.values():
        for record in group[1:]:
            union_find.union(group[0].knowledge_id, record.knowledge_id)
    for left, right, _ in near_pairs:
        union_find.union(left, right)
    groups: dict[str, list[DatasetRecord]] = {}
    for record in records:
        groups.setdefault(union_find.find(record.knowledge_id), []).append(record)

    target = {
        "train": len(records) * train_ratio,
        "validation": len(records) * validation_ratio,
        "test": len(records) * (1 - train_ratio - validation_ratio),
    }
    result = {"train": [], "validation": [], "test": []}
    rng = random.Random(seed)
    ordered_groups = list(groups.values())
    rng.shuffle(ordered_groups)
    ordered_groups.sort(key=lambda group: (-len(group), group[0].knowledge_id))
    for group in ordered_groups:
        destination = min(
            result,
            key=lambda name: (
                len(result[name]) / target[name] if target[name] else float("inf"),
                name,
            ),
        )
        result[destination].extend(sorted(group, key=lambda item: item.question_id))
    return result


def build_workload(
    records: list[DatasetRecord],
    *,
    request_count: int,
    distribution: str,
    seed: int,
    hot_fraction: float = 0.2,
    hot_probability: float = 0.8,
    phase_size: int = 50,
) -> dict[str, Any]:
    if not records:
        raise ValueError("Cannot build a workload from no records.")
    if request_count <= 0:
        raise ValueError("request_count must be positive.")
    if distribution not in {"uniform", "hotspot", "dynamic_hotspot"}:
        raise ValueError("distribution must be one of uniform, hotspot, dynamic_hotspot.")
    if not 0 < hot_fraction <= 1 or not 0 < hot_probability <= 1:
        raise ValueError("hot_fraction and hot_probability must be in (0, 1].")
    if phase_size <= 0:
        raise ValueError("phase_size must be positive.")
    rng = random.Random(seed)
    ordered = sorted(records, key=lambda item: item.question_id)
    hot_count = max(1, min(len(ordered), round(len(ordered) * hot_fraction)))

    def choose_hot() -> set[str]:
        return {record.question_id for record in rng.sample(ordered, hot_count)}

    current_hot = choose_hot() if distribution != "uniform" else set()
    requests: list[dict[str, Any]] = []
    for index in range(request_count):
        phase_index = index // phase_size
        if distribution == "dynamic_hotspot" and index % phase_size == 0:
            current_hot = choose_hot()
        if distribution == "uniform":
            chosen = rng.choice(ordered)
            hotspot_state = "uniform"
        else:
            hot_records = [record for record in ordered if record.question_id in current_hot]
            cold_records = [record for record in ordered if record.question_id not in current_hot]
            choose_from_hot = not cold_records or rng.random() < hot_probability
            population = hot_records if choose_from_hot else cold_records
            chosen = rng.choice(population)
            hotspot_state = "hot" if choose_from_hot else "cold"
        requests.append(
            {
                "request_index": index,
                "question_id": chosen.question_id,
                "knowledge_id": chosen.knowledge_id,
                "phase_index": phase_index,
                "hotspot_state": hotspot_state,
            }
        )
    return {
        "schema_version": 1,
        "metadata": {
            "distribution": distribution,
            "seed": seed,
            "request_count": request_count,
            "hot_fraction": hot_fraction,
            "hot_probability": hot_probability,
            "phase_size": phase_size,
        },
        "requests": requests,
    }
