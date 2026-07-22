from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.aggregate_prefill_calibration import aggregate
from scripts.validate_rl4_signal_smoke import validate


RATIOS = [0.6123, 0.7720, 0.9924, 1.0077]


class PrefillAggregateTest(unittest.TestCase):
    def test_aggregate_hashes_inputs_and_passes_repeatability(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dirs = []
            round_values = (
                [0.6301, 0.7961, 0.9924, 1.0077],
                [0.6101, 0.7656, 1.0001, 0.9999],
                [0.6123, 0.7720, 0.9739, 1.0275],
            )
            for index, (seed, scalars) in enumerate(zip((42, 43, 44), round_values), 1):
                run_dir = Path(temp_dir) / f"r{index}"
                run_dir.mkdir()
                summary = {
                    "run_id": f"round-{index}",
                    "git_commit": "5ec94fe",
                    "model": "model",
                    "ports": [18000, 18001, 18002, 18003],
                    "tp_sizes": [4, 2, 1, 1],
                    "length_labels": ["short", "medium", "long"],
                    "prompt_repetitions": [480, 2016, 3959],
                    "warmup_per_instance": 2,
                    "measured_per_instance": 10,
                    "parallel_instances_per_prompt": 4,
                    "seed": seed,
                    "failures": 0,
                    "by_length": {
                        label: {
                            "ports": {
                                str(port): {"ttft_median_ms": 1000.0 / value}
                                for port, value in zip(
                                    (18000, 18001, 18002, 18003), scalars
                                )
                            },
                            "capacity_ratios_relative_to_mean_tp1_median": {
                                str(port): value
                                for port, value in zip(
                                    (18000, 18001, 18002, 18003), scalars
                                )
                            }
                        }
                        for label in ("short", "medium", "long")
                    },
                    "equal_weight_median_capacity_ratios": {
                        str(port): value
                        for port, value in zip((18000, 18001, 18002, 18003), scalars)
                    },
                }
                (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
                records = []
                for prompt_index, label in enumerate(("short", "medium", "long")):
                    for offset, port in enumerate((18000, 18001, 18002, 18003)):
                        records.append({
                            "phase": "measured",
                            "length_label": label,
                            "prompt_index": prompt_index,
                            "port": port,
                            "request_started_at_unix_ns": 1_000_000_000 + offset * 1_000_000,
                        })
                (run_dir / "requests.jsonl").write_text(
                    "".join(json.dumps(record) + "\n" for record in records),
                    encoding="utf-8",
                )
                run_dirs.append(run_dir)

            result = aggregate(run_dirs, max_cv=0.05)

        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["round_count"], 3)
        self.assertEqual(result["diagnostic_speed_factors_csv"], "0.6123,0.7720,0.9924,1.0077")
        self.assertEqual(result["concurrent_http_start_skew_ms"]["p95_ms"], 3.0)
        self.assertTrue(result["concurrent_http_start_skew_ms"]["passed"])
        self.assertEqual(len(result["input_summary_sha256"]), 64)
        self.assertTrue(all(len(row["requests_sha256"]) == 64 for row in result["source_files"]))

    def test_aggregate_rejects_failed_round(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dirs = []
            for index in range(3):
                run_dir = Path(temp_dir) / str(index)
                run_dir.mkdir()
                (run_dir / "summary.json").write_text(
                    json.dumps({"failures": 1}), encoding="utf-8"
                )
                (run_dir / "requests.jsonl").write_text("{}\n", encoding="utf-8")
                run_dirs.append(run_dir)
            with self.assertRaises(ValueError):
                aggregate(run_dirs, max_cv=0.05)


def make_rr_rows(count: int = 60):
    return [
        {
            "success": True,
            "meta_missing": False,
            "trace_warnings": [],
            "trace": {
                "selected_instance_id": f"inst-{index % 4}",
                "applied_instance_strategy": "round_robin",
            },
        }
        for index in range(count)
    ]


def make_linucb_rows(count: int = 60, warmup: int = 30):
    rows = []
    instance_ids = [f"inst-{index}" for index in range(4)]
    for index in range(count):
        phase = "warmup" if index < warmup else "linucb"
        chosen = instance_ids[index % 4] if phase == "warmup" else "inst-3"
        exploits = {instance_id: (0.0 if phase == "warmup" else position * 0.1)
                    for position, instance_id in enumerate(instance_ids)}
        bonuses = {instance_id: (0.0 if phase == "warmup" else 0.2)
                   for instance_id in instance_ids}
        scores = {instance_id: exploits[instance_id] + bonuses[instance_id]
                  for instance_id in instance_ids}
        rows.append({
            "success": True,
            "meta_missing": False,
            "trace_warnings": [],
            "trace": {
                "selected_instance_id": chosen,
                "applied_instance_strategy": "linucb",
                "selection_phase": phase,
                "rl_updated": 1,
                "rl_effective_updates": index,
                "rl_feature_names": ["bias", "compute_delta_norm", "kv_ready_delta_norm"],
                "rl_candidate_features": {
                    instance_id: [1.0, position * 0.1, 0.0]
                    for position, instance_id in enumerate(instance_ids)
                },
                "rl_candidate_costs": {
                    instance_id: {"compute_capacity_ratio": RATIOS[position]}
                    for position, instance_id in enumerate(instance_ids)
                },
                "rl_candidate_scores": scores,
                "rl_candidate_exploit_scores": exploits,
                "rl_candidate_exploration_bonuses": bonuses,
                "linucb_excluded_instances": {},
            },
        })
    return rows


class SignalSmokeValidationTest(unittest.TestCase):
    def test_accepts_complete_signal_trace(self) -> None:
        result = validate(make_rr_rows(), make_linucb_rows(), RATIOS, 30)
        self.assertEqual(result["status"], "passed")
        self.assertTrue(all(result["checks"].values()))
        self.assertEqual(
            result["details"]["kv_delta_interpretation"],
            "zero is valid for shared Redis/global scope",
        )

    def test_rejects_blind_compute_context(self) -> None:
        rows = make_linucb_rows()
        for row in rows:
            for features in row["trace"]["rl_candidate_features"].values():
                features[1] = 0.0
        result = validate(make_rr_rows(), rows, RATIOS, 30)
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["checks"]["compute_context_signal_visible"])


class SignalSmokeLauncherTest(unittest.TestCase):
    def test_launcher_enforces_commit_clean_tree_and_result_bundle(self) -> None:
        root = Path(__file__).resolve().parents[1]
        source = (root / "scripts" / "run_rl4_signal_smoke.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("EXPECTED_COMMIT:?", source)
        self.assertIn("status --porcelain", source)
        self.assertIn("verify_source_sync.sh", source)
        self.assertIn("round_robin linucb", source)
        self.assertIn("validate_rl4_signal_smoke.py", source)
        self.assertIn("SHA256SUMS", source)
        self.assertIn("tar.gz", source)


if __name__ == "__main__":
    unittest.main()
