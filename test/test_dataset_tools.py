from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from types import ModuleType
import zipfile

import openpyxl

from scripts.dataset_tools import (
    DatasetRecord,
    build_workload,
    content_hash,
    load_excel_records,
    load_yaml_records,
    records_without_blocking_issues,
    split_records,
    validate_records,
)


class DatasetToolsTest(unittest.TestCase):
    def _write_fixture(self, root: Path) -> tuple[Path, Path]:
        workbook_path = root / "pairs.xlsx"
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet.append(
            [
                "sample_id",
                "dataset_source",
                "question_len",
                "knowledge_len",
                "Topic",
                "kdn_reference_document_id",
                "actual_doc_len",
                "user_question",
            ]
        )
        sheet.append(["q1", "fixture", 1, 1, "topic", "k1", 1, "Question one?"])
        sheet.append(["q2", "fixture", 1, 1, "topic", "", 1, "Question without mapping?"])
        workbook.save(workbook_path)
        archive_path = root / "knowledge.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr("k1.txt", "The matching knowledge text.")
        return workbook_path, archive_path

    def _record(self, index: int, knowledge: str) -> DatasetRecord:
        return DatasetRecord(
            question_id=f"q{index}",
            knowledge_id=f"k{index}",
            question=f"question {index}",
            knowledge_text=knowledge,
            source="fixture",
            generation_method="fixture",
            mapping_evidence=f"fixture:{index}",
            source_path="fixture",
            content_hash=content_hash(knowledge),
        )

    def test_excel_loader_requires_explicit_mapping_and_knowledge_content(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workbook, archive = self._write_fixture(Path(temp_dir))
            records, issues = load_excel_records(workbook, archive)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].question_id, "q1")
        self.assertEqual(records[0].knowledge_id, "k1")
        self.assertEqual(issues[0].code, "missing_explicit_mapping_field")
        self.assertEqual(len(records_without_blocking_issues(records, issues)), 1)

    def test_validation_reports_exact_and_near_duplicates(self) -> None:
        records = [
            self._record(1, "A document about a cache routing experiment."),
            self._record(2, "A document about a cache routing experiment."),
            self._record(3, "A document about a cache routing experiment!"),
        ]
        issues, pairs = validate_records(records, near_duplicate_threshold=0.8)
        codes = {issue.code for issue in issues}
        self.assertIn("exact_duplicate_knowledge", codes)
        self.assertIn("near_duplicate_knowledge", codes)
        self.assertTrue(pairs)

    def test_split_keeps_near_duplicates_together(self) -> None:
        records = [
            self._record(1, "near duplicate knowledge for a routing test."),
            self._record(2, "near duplicate knowledge for a routing test!"),
            self._record(3, "distinct astronomy knowledge"),
            self._record(4, "distinct chemistry knowledge"),
        ]
        splits = split_records(records, seed=7, near_duplicate_threshold=0.9)
        locations = {
            record.knowledge_id: split_name
            for split_name, group in splits.items()
            for record in group
        }
        self.assertEqual(locations["k1"], locations["k2"])

    def test_workload_is_reproducible_and_has_required_request_fields(self) -> None:
        records = [self._record(index, f"knowledge {index}") for index in range(1, 6)]
        first = build_workload(
            records,
            request_count=25,
            distribution="dynamic_hotspot",
            seed=123,
            phase_size=5,
        )
        second = build_workload(
            records,
            request_count=25,
            distribution="dynamic_hotspot",
            seed=123,
            phase_size=5,
        )
        self.assertEqual(json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True))
        self.assertEqual(first["requests"][0]["request_index"], 0)
        self.assertEqual(
            set(first["requests"][0]),
            {"request_index", "question_id", "knowledge_id", "phase_index", "hotspot_state"},
        )

    def test_yaml_loader_only_uses_explicit_same_record_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "raw.yaml"
            path.write_text("fixture", encoding="utf-8")
            yaml_stub = ModuleType("yaml")
            yaml_stub.safe_load = lambda _: {
                "knowledge_items": [
                    {
                        "id": "k1",
                        "question": "What is the test fact?",
                        "content": "The test fact is explicit.",
                    },
                    {"id": "k2", "content": "This row has no question."},
                ]
            }
            with mock.patch.dict("sys.modules", {"yaml": yaml_stub}):
                records, issues = load_yaml_records(
                    path,
                    items_key="knowledge_items",
                    question_field="question",
                    knowledge_field="content",
                    question_id_field="id",
                    knowledge_id_field="id",
                    source="fixture",
                    generation_method="fixture",
                )
        self.assertEqual([(record.question_id, record.knowledge_id) for record in records], [("k1", "k1")])
        self.assertEqual(issues[0].code, "missing_explicit_mapping_field")


if __name__ == "__main__":
    unittest.main()
