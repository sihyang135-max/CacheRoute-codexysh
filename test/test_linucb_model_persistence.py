from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import dataclass
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType, SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
core_stub = ModuleType("core")
core_stub.config = SimpleNamespace(
    PROXY_RL_ALPHA=0.4,
    PROXY_RL_LAMBDA=1.0,
    PROXY_RL_WARMUP_REQUESTS=0,
    PROXY_RL_COMPUTE_COST_SCALE_MS=1000.0,
    PROXY_RL_KV_READY_COST_SCALE_MS=1000.0,
    PROXY_RL_FROZEN=False,
    PROXY_RL_MODEL_SAVE_PATH="",
    PROXY_RL_PARAMETER_SNAPSHOT_PATH="",
    PROXY_RL_PARAMETER_SNAPSHOT_INTERVAL=20,
)
sys.modules["core"] = core_stub
proxy_pkg = ModuleType("proxy")
proxy_pkg.__path__ = [str(ROOT / "proxy")]
strategy_pkg = ModuleType("proxy.strategy")
strategy_pkg.__path__ = [str(ROOT / "proxy" / "strategy")]
sys.modules["proxy"] = proxy_pkg
sys.modules["proxy.strategy"] = strategy_pkg


def load(name: str, path: Path):
    spec = spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


load("proxy.strategy.base", ROOT / "proxy" / "strategy" / "base.py")
linucb = load("proxy.strategy.linucb", ROOT / "proxy" / "strategy" / "linucb.py")
LinUCBStrategy = linucb.LinUCBStrategy
classify_feedback = linucb.classify_feedback


@dataclass
class FakeInstance:
    instance_id: str
    host: str = "127.0.0.1"
    port: int = 9001


class LinUCBPersistenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.instances = [FakeInstance("inst-0"), FakeInstance("inst-1")]
        self.contexts = {
            "inst-0": {"compute_cost_ms": 100.0, "kv_ready_cost_ms": 0.0},
            "inst-1": {"compute_cost_ms": 900.0, "kv_ready_cost_ms": 0.0},
        }

    def trained(self, **kwargs) -> LinUCBStrategy:
        strategy = LinUCBStrategy(warmup_requests=0, **kwargs)
        for reward in (-0.2, -0.8, -0.3, -0.7):
            decision = strategy.choose(self.instances, self.contexts)
            self.assertTrue(strategy.update(decision.instance_id, decision.features, reward))
        return strategy

    def test_save_load_preserves_state_and_scores(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model = Path(temp_dir) / "model.json"
            trained = self.trained()
            before = trained.choose(self.instances, self.contexts)
            trained.save_model(str(model))

            restored = LinUCBStrategy(warmup_requests=0)
            restored.load_model(str(model))
            after = restored.choose(self.instances, self.contexts)

            self.assertEqual(restored.effective_updates, trained.effective_updates)
            self.assertEqual(restored.arm_updates, trained.arm_updates)
            for instance_id in before.candidate_scores:
                self.assertAlmostEqual(
                    before.candidate_scores[instance_id],
                    after.candidate_scores[instance_id],
                    places=12,
                )

    def test_frozen_model_does_not_update(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model = Path(temp_dir) / "model.json"
            self.trained().save_model(str(model))
            frozen = LinUCBStrategy(warmup_requests=0, frozen=True)
            frozen.load_model(str(model))
            before = frozen.parameter_snapshot()
            selections_before = frozen.effective_selections
            decision = frozen.choose(self.instances, self.contexts)
            self.assertFalse(frozen.update(decision.instance_id, decision.features, -1.0))
            after = frozen.parameter_snapshot()
            self.assertEqual(before["effective_updates"], after["effective_updates"])
            self.assertEqual(before["arms"], after["arms"])
            self.assertEqual(frozen.effective_selections, selections_before + 1)

    def test_checkpoint_writes_theta_at_fixed_interval(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            snapshots = Path(temp_dir) / "theta.jsonl"
            strategy = LinUCBStrategy(
                warmup_requests=0,
                parameter_snapshot_path=str(snapshots),
                parameter_snapshot_interval=2,
            )
            for _ in range(2):
                decision = strategy.choose(self.instances, self.contexts)
                strategy.update(decision.instance_id, decision.features, -0.5)
                strategy.checkpoint_if_due()
            rows = [json.loads(line) for line in snapshots.read_text().splitlines()]
            self.assertEqual([row["effective_updates"] for row in rows], [2])
            self.assertEqual(rows[0]["feature_names"], [
                "bias", "compute_delta_norm", "kv_ready_delta_norm"
            ])

    def test_corrupt_or_mismatched_model_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model = Path(temp_dir) / "model.json"
            model.write_text("not-json", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid LinUCB model JSON"):
                LinUCBStrategy(warmup_requests=0).load_model(str(model))

            self.trained().save_model(str(model))
            with self.assertRaisesRegex(ValueError, "metadata mismatch for alpha"):
                LinUCBStrategy(alpha=0.2, warmup_requests=0).load_model(str(model))

    def test_loaded_model_rejects_instance_set_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model = Path(temp_dir) / "model.json"
            self.trained().save_model(str(model))
            restored = LinUCBStrategy(warmup_requests=0)
            restored.load_model(str(model))
            with self.assertRaisesRegex(RuntimeError, "instance set mismatch"):
                restored.choose(self.instances[:1], {"inst-0": self.contexts["inst-0"]})

    def test_feedback_classification_avoids_unattributable_penalties(self) -> None:
        self.assertEqual(classify_feedback(20, 10), ("success", "observed_ttft", True))
        self.assertEqual(
            classify_feedback(None, 10, "ready_failed: read timeout"),
            ("timeout", "instance_failure_penalty", True),
        )
        self.assertEqual(
            classify_feedback(None, 10, "ready_failed: HTTP 500"),
            ("backend_failure", "instance_failure_penalty", True),
        )
        self.assertFalse(classify_feedback(None, 10, "client_cancelled")[2])
        self.assertFalse(classify_feedback(None, 10, "")[2])


if __name__ == "__main__":
    unittest.main()
