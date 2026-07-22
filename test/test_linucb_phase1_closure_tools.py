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


analysis_module = load("phase1_analysis", ROOT / "scripts" / "analyze_linucb_convergence.py")
validation_module = load("phase1_validation", ROOT / "scripts" / "validate_rl4_phase1_closure.py")


def row(index: int, frozen: bool = False) -> dict:
    selected = f"inst-{index % 4}"
    return {
        "success": True,
        "client_ttft_ms": 100.0,
        "trace": {
            "selected_instance_id": selected,
            "selection_phase": "linucb" if index >= 30 else "warmup",
            "rl_candidate_exploration_bonuses": {selected: 0.1},
            "rl_reward_milli": -100,
            "rl_updated": 0 if frozen else 1,
            "rl_update_reason": "frozen" if frozen else "updated",
            "rl_model_frozen": 1 if frozen else 0,
            "rl_runtime_mode": "loaded-frozen" if frozen else "fresh-training",
            "outcome_class": "success",
            "reward_source": "observed_ttft",
        },
    }


def model() -> dict:
    return {
        "model_format_version": 1,
        "source_commit": "a" * 40,
        "feature_names": ["bias", "compute_delta_norm", "kv_ready_delta_norm"],
        "dim": 3,
        "alpha": 0.4,
        "lambda": 1.0,
        "warmup_requests": 30,
        "compute_scale_ms": 1000.0,
        "kv_ready_scale_ms": 1000.0,
        "effective_updates": 240,
        "arms": {f"inst-{i}": {"A_inv": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "b": [0, 0, 0], "updates": 60} for i in range(4)},
    }


def snapshots() -> list[dict]:
    points = sorted(set([30, *range(20, 241, 20)]))
    return [{
        "feature_names": ["bias", "compute_delta_norm", "kv_ready_delta_norm"],
        "effective_updates": point,
        "arms": {f"inst-{i}": {"theta": [0.0, 0.0, 0.0], "theta_l2_norm": 0.0, "theta_delta_l2_norm": None if point == 20 else 0.0} for i in range(4)},
    } for point in points]


class ConvergenceAnalysisTest(unittest.TestCase):
    def test_240_requests_never_claim_convergence(self) -> None:
        result = analysis_module.analyze([row(i) for i in range(240)])
        self.assertEqual(result["windows"]["100"]["complete_windows"], 2)
        self.assertEqual(result["windows"]["200"]["complete_windows"], 1)
        self.assertEqual(result["windows"]["100"]["assessment"]["status"], "insufficient_data")
        self.assertEqual(result["windows"]["200"]["assessment"]["status"], "insufficient_data")

    def test_three_stable_windows_are_diagnostic_only(self) -> None:
        result = analysis_module.analyze([row(i) for i in range(600)])
        self.assertEqual(result["windows"]["100"]["assessment"]["status"], "stable_diagnostic")
        self.assertEqual(result["claim_boundary"], "diagnostic only; no convergence or performance conclusion")


class ClosureValidationTest(unittest.TestCase):
    def test_expected_facility_artifacts_pass(self) -> None:
        analysis = analysis_module.analyze([row(i) for i in range(240)])
        config = {"git_commit": "a" * 40, "concurrency": 1, "kv_residency_scope": "global", "kv_link_scope": "global", "training_requests": 240, "frozen_requests": 80, "parameter_snapshot_interval": 20, "warmup_requests": 30}
        result = validation_module.validate([row(i) for i in range(240)], [row(i, frozen=True) for i in range(80)], snapshots(), model(), model(), analysis, "a" * 40, config, "same", "same")
        self.assertEqual(result["status"], "passed", result)

    def test_frozen_update_is_blocking(self) -> None:
        frozen = [row(i, frozen=True) for i in range(80)]
        frozen[5]["trace"]["rl_updated"] = 1
        result = validation_module.validate([row(i) for i in range(240)], frozen, snapshots(), model(), model(), analysis_module.analyze([row(i) for i in range(240)]), "a" * 40, {"git_commit": "a" * 40, "concurrency": 1, "kv_residency_scope": "global", "kv_link_scope": "global", "training_requests": 240, "frozen_requests": 80, "parameter_snapshot_interval": 20, "warmup_requests": 30}, "same", "same")
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["checks"]["frozen_never_updates"])


class RuntimeIsolationTest(unittest.TestCase):
    def test_runtime_writes_are_kept_out_of_tracked_source(self) -> None:
        docker_start = (ROOT / "scripts" / "start_rl_4instance_docker.sh").read_text(encoding="utf-8")
        closure = (ROOT / "scripts" / "run_rl4_phase1_closure.sh").read_text(encoding="utf-8")
        self.assertIn("docker exec -e PYTHONDONTWRITEBYTECODE=1", docker_start)
        self.assertIn("-e KDN_TEXT_DB_DIR=", docker_start)
        self.assertIn('KDN_TEXT_DB_DIR="$container_run_dir/kdn-text-db"', closure)
        self.assertIn('PROXY_RL_SOURCE_COMMIT="$EXPECTED_COMMIT"', closure)


if __name__ == "__main__":
    unittest.main()
