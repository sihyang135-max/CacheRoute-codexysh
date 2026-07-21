from __future__ import annotations

import unittest
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

from scripts.calibrate_prefill_capacity import (
    build_prompt,
    percentile_nearest_rank,
    run_request_batch,
    summarize,
)


class PrefillCalibrationTest(unittest.TestCase):
    def test_prompt_nonce_is_first_and_unique(self) -> None:
        first = build_prompt("run-1", "short", 1, 3)
        second = build_prompt("run-1", "short", 2, 3)

        self.assertTrue(first.startswith("nonce="))
        self.assertEqual(first.count(" a"), 3)
        self.assertNotEqual(first.splitlines()[0], second.splitlines()[0])

    def test_same_prompt_index_produces_the_same_prompt(self) -> None:
        first = build_prompt("run-1", "medium", 7, 3)
        second = build_prompt("run-1", "medium", 7, 3)

        self.assertEqual(first, second)

    def test_request_batch_starts_all_instances_concurrently(self) -> None:
        barrier = Barrier(4, timeout=1.0)

        def fake_request(**job):
            barrier.wait()
            return job

        jobs = [{"port": port} for port in (18000, 18001, 18002, 18003)]
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = run_request_batch(executor, jobs, request_fn=fake_request)

        self.assertEqual({row["port"] for row in results}, {18000, 18001, 18002, 18003})

    def test_nearest_rank_percentile(self) -> None:
        self.assertEqual(percentile_nearest_rank([1, 2, 3, 4], 0.95), 4.0)

    def test_summary_excludes_warmup_and_uses_tp1_medians(self) -> None:
        records = []
        for port, tp, values in (
            (18000, 4, [400.0, 600.0]),
            (18001, 2, [200.0, 300.0]),
            (18002, 1, [240.0, 260.0]),
            (18003, 1, [250.0, 270.0]),
        ):
            records.append({
                "phase": "warmup",
                "success": True,
                "length_label": "short",
                "port": port,
                "tp": tp,
                "ttft_ms": 9999.0,
                "prompt_tokens": 512,
            })
            records.extend({
                "phase": "measured",
                "success": True,
                "length_label": "short",
                "port": port,
                "tp": tp,
                "ttft_ms": value,
                "prompt_tokens": 512,
            } for value in values)

        summary = summarize(records, [18000, 18001, 18002, 18003])

        self.assertEqual(summary["measured_successes"], 8)
        ratios = summary["by_length"]["short"][
            "capacity_ratios_relative_to_mean_tp1_median"
        ]
        self.assertAlmostEqual(ratios["18000"], 0.51)
        self.assertAlmostEqual(ratios["18001"], 1.02)
        self.assertEqual(
            summary["by_length"]["short"]["ports"]["18000"]["ttft_median_ms"],
            500.0,
        )


if __name__ == "__main__":
    unittest.main()
