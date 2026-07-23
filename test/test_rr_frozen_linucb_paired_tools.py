from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


validation = load(
    "rr_frozen_validation",
    ROOT / "scripts" / "validate_rr_frozen_linucb_paired.py",
)


def row(strategy: str, ttft: float, frozen: bool = False) -> dict:
    return {
        "success": True,
        "client_ttft_ms": ttft,
        "trace": {
            "applied_instance_strategy": strategy,
            "selected_instance_id": "inst-0",
            "rl_model_frozen": 1 if frozen else 0,
            "rl_runtime_mode": "loaded-frozen" if frozen else "disabled",
            "rl_updated": 0,
        },
    }


def inputs() -> tuple:
    commit = "a" * 40
    config = {
        "git_commit": commit,
        "repeats": 2,
        "measure_requests": 2,
        "training_requests": 240,
        "concurrency": 1,
        "kv_residency_scope": "global",
        "kv_link_scope": "global",
    }
    order = [
        {"pair": "1", "strategy": "round_robin", "seed": "10", "raw_file": "rr1"},
        {"pair": "1", "strategy": "linucb", "seed": "10", "raw_file": "rl1"},
        {"pair": "2", "strategy": "linucb", "seed": "11", "raw_file": "rl2"},
        {"pair": "2", "strategy": "round_robin", "seed": "11", "raw_file": "rr2"},
    ]
    rows = {
        "rr1": [row("round_robin", 120.0), row("round_robin", 100.0)],
        "rl1": [row("linucb", 90.0, True), row("linucb", 80.0, True)],
        "rl2": [row("linucb", 95.0, True), row("linucb", 85.0, True)],
        "rr2": [row("round_robin", 125.0), row("round_robin", 105.0)],
    }
    model = {"source_commit": commit, "effective_updates": 240}
    return config, order, rows, model, commit


class PairedValidationTest(unittest.TestCase):
    def test_valid_paired_run_passes_and_reports_each_pair(self) -> None:
        config, order, rows, model, commit = inputs()
        result = validation.validate(
            config, order, rows, model, model.copy(), "same", "same", commit
        )
        self.assertEqual(result["status"], "passed", result)
        self.assertEqual(len(result["paired_ttft"]), 2)
        self.assertGreater(
            result["paired_ttft"][0]["relative_improvement"]["median"], 0
        )

    def test_frozen_update_is_blocking(self) -> None:
        config, order, rows, model, commit = inputs()
        rows["rl2"][0]["trace"]["rl_updated"] = 1
        result = validation.validate(
            config, order, rows, model, model.copy(), "same", "same", commit
        )
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["checks"]["frozen_linucb_never_updates"])


class RunnerContractTest(unittest.TestCase):
    def test_runner_binds_model_and_alternates_policy_order(self) -> None:
        script = (
            ROOT / "scripts" / "run_rr_frozen_linucb_paired.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('PROXY_RL_SOURCE_COMMIT="$EXPECTED_COMMIT"', script)
        self.assertIn('PROXY_RL_MODEL_LOAD_PATH="$load_path"', script)
        self.assertIn('PROXY_RL_FROZEN="$frozen"', script)
        self.assertIn('if [ $((pair % 2)) -eq 1 ]', script)
        self.assertIn('cmp "$host_model"', script)


if __name__ == "__main__":
    unittest.main()
