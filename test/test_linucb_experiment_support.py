from __future__ import annotations

import ast
import asyncio
import unittest
from dataclasses import dataclass
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, patch
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from util.openai_stream import OpenAIStreamObserver
from kdn_server.text_db import TextDatabase


def load_module(name: str, path: Path):
    spec = spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Load the narrow units directly so tests do not initialize the full model,
# embedding, and knowledge-store stack.
sys.modules.setdefault("httpx", ModuleType("httpx"))
perf_client = load_module("_perf_client_under_test", ROOT / "client" / "perf_client.py")
observe_stream_payload = perf_client.observe_stream_payload

core_stub = ModuleType("core")
core_stub.config = SimpleNamespace(
    PROXY_RL_ALPHA=0.4,
    PROXY_RL_LAMBDA=1.0,
    PROXY_RL_WARMUP_REQUESTS=30,
    PROXY_RL_COMPUTE_COST_SCALE_MS=1000.0,
    PROXY_RL_KV_READY_COST_SCALE_MS=1000.0,
    PROXY_RL_FROZEN=False,
    PROXY_RL_MODEL_SAVE_PATH="",
    PROXY_RL_SOURCE_COMMIT="",
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
load_module("proxy.strategy.base", ROOT / "proxy" / "strategy" / "base.py")
linucb = load_module("proxy.strategy.linucb", ROOT / "proxy" / "strategy" / "linucb.py")
LinUCBStrategy = linucb.LinUCBStrategy
ttft_reward = linucb.ttft_reward
least_inflight = load_module(
    "proxy.strategy.least_inflight", ROOT / "proxy" / "strategy" / "least_inflight.py"
)
LeastInflightStrategy = least_inflight.LeastInflightStrategy
round_robin = load_module(
    "proxy.strategy.round_robin", ROOT / "proxy" / "strategy" / "round_robin.py"
)
RoundRobinStrategy = round_robin.RoundRobinStrategy
candidate_filter = load_module(
    "proxy.strategy.candidate_filter", ROOT / "proxy" / "strategy" / "candidate_filter.py"
)
filter_safe_instances = candidate_filter.filter_safe_instances
ProxyTask = load_module("_proxy_task_under_test", ROOT / "proxy" / "queue" / "task.py").ProxyTask

redis_stub = ModuleType("redis")
redis_stub.Redis = object
sys.modules.setdefault("redis", redis_stub)
kv_injector = load_module(
    "_kv_injector_under_test",
    ROOT / "kdn_server" / "kv_injector.py",
)
prometheus_cache = load_module(
    "_prometheus_cache_under_test",
    ROOT / "proxy" / "metrics" / "prometheus_cache.py",
)
PrometheusCache = prometheus_cache.PrometheusCache


@dataclass
class FakeInstance:
    instance_id: str
    host: str = "127.0.0.1"
    port: int = 9001
    weight: float = 1.0


@dataclass
class FakeMetric:
    fresh: bool = True
    failures: int = 0
    kv_usage: float = 0.0

    def is_fresh(self, stale_s: float) -> bool:
        del stale_s
        return self.fresh


def context(compute_ms: float, kv_ready_ms: float = 0.0) -> dict:
    return {
        "compute_cost_ms": compute_ms,
        "kv_ready_cost_ms": kv_ready_ms,
    }


class LinUCBObservabilityTest(unittest.TestCase):
    def test_ttft_reward_has_an_experiment_scale_not_client_slo(self) -> None:
        self.assertAlmostEqual(ttft_reward(300, scale_ms=1000, clip=5), -0.3)
        self.assertAlmostEqual(ttft_reward(900, scale_ms=1000, clip=5), -0.9)
        self.assertEqual(ttft_reward(9000, scale_ms=1000, clip=5), -5.0)

    def test_decision_exposes_phase_candidates_and_update_count(self) -> None:
        instances = [FakeInstance("inst-0"), FakeInstance("inst-1")]
        contexts = {"inst-0": context(100), "inst-1": context(200)}
        strategy = LinUCBStrategy(alpha=0.1, ridge=1.0, warmup_requests=1)

        warmup = strategy.choose(instances, contexts)
        self.assertEqual(warmup.phase, "warmup")
        self.assertEqual(warmup.instance_id, "inst-0")
        self.assertEqual(warmup.effective_updates, 0)
        self.assertEqual(set(warmup.candidate_features), {"inst-0", "inst-1"})

        strategy.update(warmup.instance_id, warmup.features, reward=-0.5)
        decision = strategy.choose(instances, contexts)
        self.assertEqual(decision.phase, "linucb")
        self.assertEqual(decision.effective_updates, 1)
        self.assertEqual(set(decision.candidate_scores), {"inst-0", "inst-1"})

    def test_warmup_balances_updates_across_arms(self) -> None:
        instances = [FakeInstance("inst-0"), FakeInstance("inst-1")]
        contexts = {"inst-0": context(100), "inst-1": context(100)}
        strategy = LinUCBStrategy(alpha=0.1, ridge=1.0, warmup_requests=4)

        for _ in range(4):
            decision = strategy.choose(instances, contexts)
            strategy.update(decision.instance_id, decision.features, reward=-0.5)

        self.assertEqual(strategy.arm_updates, {"inst-0": 2, "inst-1": 2})

    def test_concurrent_warmup_balances_selections_before_feedback(self) -> None:
        instances = [FakeInstance("inst-0"), FakeInstance("inst-1")]
        contexts = {"inst-0": context(100), "inst-1": context(100)}
        strategy = LinUCBStrategy(alpha=0.1, ridge=1.0, warmup_requests=4)

        decisions = [strategy.choose(instances, contexts) for _ in range(4)]

        self.assertEqual([item.phase for item in decisions], ["warmup"] * 4)
        self.assertEqual(strategy.arm_selections, {"inst-0": 2, "inst-1": 2})
        self.assertEqual(strategy.effective_updates, 0)

    def test_standard_ucb_bonus_uses_context_uncertainty(self) -> None:
        instances = [FakeInstance("inst-0"), FakeInstance("inst-1")]
        strategy = LinUCBStrategy(alpha=0.2, ridge=1.0, warmup_requests=0)
        decision = strategy.choose(
            instances,
            {"inst-0": context(100), "inst-1": context(900)},
        )

        self.assertAlmostEqual(
            decision.candidate_exploration_bonuses["inst-0"],
            0.2,
        )
        self.assertAlmostEqual(
            decision.candidate_exploration_bonuses["inst-1"],
            0.2 * (1.0 + 0.8**2) ** 0.5,
        )

    def test_disjoint_models_update_only_the_selected_arm(self) -> None:
        instances = [FakeInstance("inst-0"), FakeInstance("inst-1")]
        contexts = {"inst-0": context(100), "inst-1": context(100)}
        strategy = LinUCBStrategy(alpha=0.0, ridge=1.0, warmup_requests=0)
        decision = strategy.choose(instances, contexts)
        strategy.update("inst-0", decision.candidate_features["inst-0"], reward=-1.0)
        after = strategy.choose(instances, contexts)

        self.assertLess(after.candidate_exploit_scores["inst-0"], 0.0)
        self.assertEqual(after.candidate_exploit_scores["inst-1"], 0.0)
        self.assertEqual(strategy.arm_updates, {"inst-0": 1, "inst-1": 0})

    def test_standard_disjoint_update_matches_closed_form(self) -> None:
        instances = [FakeInstance("inst-0"), FakeInstance("inst-1")]
        contexts = {"inst-0": context(100), "inst-1": context(900)}
        strategy = LinUCBStrategy(alpha=0.3, ridge=1.0, warmup_requests=0)
        decision = strategy.choose(instances, contexts)
        x = np.asarray(decision.candidate_features["inst-1"], dtype=float)
        reward = -0.75
        strategy.update("inst-1", x, reward=reward)

        expected_A_inv = np.linalg.inv(np.eye(3) + np.outer(x, x))
        expected_theta = expected_A_inv @ (x * reward)
        expected_exploit = float(expected_theta @ x)
        expected_bonus = 0.3 * float(x @ expected_A_inv @ x) ** 0.5
        after = strategy.choose(instances, contexts)

        self.assertAlmostEqual(
            after.candidate_exploit_scores["inst-1"],
            expected_exploit,
        )
        self.assertAlmostEqual(
            after.candidate_exploration_bonuses["inst-1"],
            expected_bonus,
        )

    def test_common_kv_cost_is_removed_from_every_candidate(self) -> None:
        instances = [FakeInstance("inst-0"), FakeInstance("inst-1")]
        strategy = LinUCBStrategy(alpha=0.0, ridge=1.0, warmup_requests=0)
        decision = strategy.choose(
            instances,
            {
                "inst-0": context(100, kv_ready_ms=700),
                "inst-1": context(200, kv_ready_ms=700),
            },
        )

        self.assertEqual(decision.candidate_features["inst-0"][2], 0.0)
        self.assertEqual(decision.candidate_features["inst-1"][2], 0.0)

    def test_kv_ready_cost_remains_part_of_compact_context(self) -> None:
        instances = [FakeInstance("inst-0"), FakeInstance("inst-1")]
        contexts = {
            "inst-0": context(100, kv_ready_ms=100),
            "inst-1": context(100, kv_ready_ms=900),
        }
        strategy = LinUCBStrategy(alpha=0.0, ridge=1.0, warmup_requests=0)
        decision = strategy.choose(instances, contexts)

        self.assertEqual(decision.candidate_features["inst-0"][2], 0.0)
        self.assertAlmostEqual(decision.candidate_features["inst-1"][2], 0.8)

    def test_least_inflight_prefers_proxy_local_load_snapshot(self) -> None:
        instances = [FakeInstance("inst-0"), FakeInstance("inst-1")]
        chosen = LeastInflightStrategy().select(
            instances,
            hint={"instance_inflight": {"inst-0": 3, "inst-1": 0}},
        )
        self.assertEqual(chosen.instance_id, "inst-1")

    def test_least_inflight_rotates_equal_load_ties(self) -> None:
        instances = [FakeInstance("inst-0"), FakeInstance("inst-1")]
        strategy = LeastInflightStrategy()
        hint = {"inflight_by_instance": {"inst-0": 0, "inst-1": 0}}

        self.assertEqual(strategy.select(instances, hint).instance_id, "inst-0")
        self.assertEqual(strategy.select(instances, hint).instance_id, "inst-1")


class SharedCandidateFilterTest(unittest.TestCase):
    @staticmethod
    def queue_snapshot(prepare: int = 0, ready: int = 0) -> dict:
        return {
            "prepare_queue_size": prepare,
            "ready_queue_size": ready,
        }

    def test_filters_every_safety_reason_and_sorts_candidates(self) -> None:
        instances = [
            FakeInstance("inst-safe-b"),
            FakeInstance("inst-queue"),
            FakeInstance("inst-missing"),
            FakeInstance("inst-stale"),
            FakeInstance("inst-failures"),
            FakeInstance("inst-kv"),
            FakeInstance("inst-safe-a"),
        ]
        metrics = {
            "inst-safe-a": FakeMetric(),
            "inst-safe-b": FakeMetric(),
            "inst-queue": FakeMetric(),
            "inst-stale": FakeMetric(fresh=False),
            "inst-failures": FakeMetric(failures=3),
            "inst-kv": FakeMetric(kv_usage=0.9),
        }

        safe, excluded = filter_safe_instances(
            instances,
            metric_for=metrics.get,
            queue_snapshot_for=lambda instance_id: self.queue_snapshot(
                prepare=256 if instance_id == "inst-queue" else 0
            ),
            metric_stale_s=3.0,
            metric_failure_limit=3,
            kv_usage_limit=0.9,
        )

        self.assertEqual(
            [item.instance_id for item in safe],
            ["inst-safe-a", "inst-safe-b"],
        )
        self.assertEqual(excluded, {
            "inst-failures": "metrics_failures",
            "inst-kv": "kv_usage_limit",
            "inst-missing": "metrics_missing",
            "inst-queue": "proxy_queue_limit",
            "inst-stale": "metrics_stale",
        })
        contexts = {
            item.instance_id: context(100)
            for item in safe
        }
        self.assertIn(RoundRobinStrategy().select(safe).instance_id, contexts)
        self.assertIn(LeastInflightStrategy().select(safe).instance_id, contexts)
        self.assertIn(
            LinUCBStrategy(warmup_requests=0).choose(safe, contexts).instance_id,
            contexts,
        )

    def test_single_safe_candidate_is_not_replaced_by_an_excluded_one(self) -> None:
        instances = [FakeInstance("inst-excluded"), FakeInstance("inst-safe")]
        metrics = {
            "inst-excluded": FakeMetric(fresh=False),
            "inst-safe": FakeMetric(),
        }
        safe, excluded = filter_safe_instances(
            instances,
            metric_for=metrics.get,
            queue_snapshot_for=lambda _: self.queue_snapshot(),
            metric_stale_s=3.0,
            metric_failure_limit=3,
            kv_usage_limit=0.9,
        )

        self.assertEqual([item.instance_id for item in safe], ["inst-safe"])
        self.assertEqual(excluded, {"inst-excluded": "metrics_stale"})
        self.assertEqual(LeastInflightStrategy().select(safe).instance_id, "inst-safe")
        decision = LinUCBStrategy(warmup_requests=1).choose(
            safe, {"inst-safe": context(100)}
        )
        self.assertEqual(decision.instance_id, "inst-safe")

    def test_no_safe_candidates_returns_empty_without_fallback(self) -> None:
        instances = [FakeInstance("inst-stale"), FakeInstance("inst-missing")]
        metrics = {"inst-stale": FakeMetric(fresh=False)}

        safe, excluded = filter_safe_instances(
            instances,
            metric_for=metrics.get,
            queue_snapshot_for=lambda _: self.queue_snapshot(),
            metric_stale_s=3.0,
            metric_failure_limit=3,
            kv_usage_limit=0.9,
        )

        self.assertEqual(safe, [])
        self.assertEqual(excluded, {
            "inst-missing": "metrics_missing",
            "inst-stale": "metrics_stale",
        })


class KDNRefreshStabilityTest(unittest.IsolatedAsyncioTestCase):
    async def test_unchanged_refresh_preserves_indexed_table(self) -> None:
        class FakeKnowledgeTable:
            def __init__(self, dim: int = 2, index=None):
                self.dim = dim
                self._units = {"kid-1": object(), "kid-2": object()}
                self._faiss_index = index
                self.clone_calls = 0

            def clone_without_index(self):
                self.clone_calls += 1
                return FakeKnowledgeTable(dim=self.dim, index=None)

        class FakeKnowledgeUnit:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        core_pkg = ModuleType("core")
        core_pkg.__path__ = []
        core_config = ModuleType("core.config")
        core_config.SCHEDULER_KDN_REFRESH_INTERVAL_S_DEFAULT = 30
        store_pkg = ModuleType("store")
        store_pkg.__path__ = []
        knowledge_base = ModuleType("store.knowledge_base")
        knowledge_base.KnowledgeTable = FakeKnowledgeTable
        knowledge_base.KnowledgeUnit = FakeKnowledgeUnit
        scheduler_pkg = ModuleType("scheduler")
        scheduler_pkg.__path__ = []
        knowledge_pkg = ModuleType("scheduler.knowledge")
        knowledge_pkg.__path__ = [str(ROOT / "scheduler" / "knowledge")]
        resource_pkg = ModuleType("scheduler.resource")
        resource_pkg.__path__ = []
        control_plane = ModuleType("scheduler.resource.control_plane")
        control_plane._hb_agg = None
        kdn_client = ModuleType("scheduler.knowledge.kdn_client")
        kdn_client.fetch_kdn_snapshot = AsyncMock(return_value=[])
        kdn_client.fetch_kdn_items_by_kids = AsyncMock(return_value=[])

        module_stubs = {
            "core": core_pkg,
            "core.config": core_config,
            "store": store_pkg,
            "store.knowledge_base": knowledge_base,
            "scheduler": scheduler_pkg,
            "scheduler.knowledge": knowledge_pkg,
            "scheduler.resource": resource_pkg,
            "scheduler.resource.control_plane": control_plane,
            "scheduler.knowledge.kdn_client": kdn_client,
        }
        with patch.dict(sys.modules, module_stubs):
            kdn_sync = load_module(
                "scheduler.knowledge._kdn_sync_under_test",
                ROOT / "scheduler" / "knowledge" / "kdn_sync.py",
            )

        old_index = object()
        old_table = FakeKnowledgeTable(index=old_index)
        app = SimpleNamespace(
            state=SimpleNamespace(
                _kdn_refresh_lock=asyncio.Lock(),
                knowledge_table=old_table,
            )
        )
        kdn_sync.refresh_kdn_knowledge_index = AsyncMock(return_value={})
        kdn_sync.select_kdn_base_url = AsyncMock(
            return_value=("http://127.0.0.1:9101", ["kdn://127.0.0.1:9101"])
        )
        kdn_sync.fetch_kdn_snapshot = AsyncMock(return_value=[])
        kdn_sync.diff_kdn_meta = lambda _: ([], [], [])

        result = await kdn_sync.kdn_refresh_once(app)

        self.assertTrue(result["ok"])
        self.assertEqual(result["entries"], 2)
        self.assertEqual(old_table.clone_calls, 1)
        self.assertIs(app.state.knowledge_table, old_table)
        self.assertIs(app.state.knowledge_table._faiss_index, old_index)


class TextDatabaseEmbeddingRepairTest(unittest.TestCase):
    class FakeEmbedder:
        def __init__(self) -> None:
            self.calls = 0

        def encode_vector(self, content: str):
            self.calls += 1
            return [np.asarray([1.0, 2.0, 3.0], dtype=np.float32)]

    def test_existing_row_without_embedding_is_backfilled(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            plain_db = TextDatabase(temp_dir)
            kid, status, _ = plain_db.register_text("repair me")
            self.assertEqual(status, "created")
            self.assertIsNone(plain_db.get_many([kid])[0][0].embedding)

            embedder = self.FakeEmbedder()
            repaired_db = TextDatabase(temp_dir, embedder=embedder)
            _, status, _ = repaired_db.register_text("repair me")
            item = repaired_db.get_many([kid])[0][0]

            self.assertEqual(status, "exists")
            self.assertEqual(embedder.calls, 1)
            self.assertEqual(item.embed_dim, 3)
            self.assertEqual(item.embedding, [1.0, 2.0, 3.0])

            repaired_db.register_text("repair me")
            self.assertEqual(embedder.calls, 1)


class ExperimentLauncherSafetyTest(unittest.TestCase):
    def test_launcher_tracks_process_groups_and_cleans_partial_startup(self) -> None:
        source = (ROOT / "scripts" / "start_rl_4instance_in_container.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn('nohup setsid "$@"', source)
        self.assertIn('kill -TERM -- "-$pid"', source)
        self.assertIn("trap cleanup_on_exit EXIT", source)
        self.assertIn("startup failed; stopping partial stack", source)
        self.assertIn("stale vLLM processes remain after cleanup", source)


class KVResidencyTest(unittest.TestCase):
    class FakePipeline:
        def __init__(self, values: dict[bytes, bytes]) -> None:
            self.values = values
            self.keys: list[bytes] = []

        def exists(self, key: bytes) -> None:
            self.keys.append(key)

        def execute(self) -> list[int]:
            return [int(key in self.values) for key in self.keys]

    class FakeRedis:
        def __init__(self) -> None:
            self.values: dict[bytes, bytes] = {}

        def pipeline(self, transaction: bool = False):
            return KVResidencyTest.FakePipeline(self.values)

        def set(self, key: bytes, value: bytes, nx: bool = False) -> bool:
            if nx and key in self.values:
                return False
            self.values[key] = value
            return True

    def test_cold_injection_then_resident_hit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "blocks").mkdir()
            (root / "blocks" / "value.dump").write_bytes(b"payload")
            (root / "manifest.jsonl").write_text(
                '{"key_b64url":"azE","file":"blocks/value.dump"}\n',
                encoding="utf-8",
            )
            injector = kv_injector.KVCacheInjector.__new__(
                kv_injector.KVCacheInjector
            )
            injector.rds = self.FakeRedis()

            cold = injector.inject_kv_dir(str(root))
            warm = injector.inject_kv_dir(str(root))

            self.assertEqual(cold.injected, 1)
            self.assertEqual(cold.payload_bytes, len(b"payload"))
            self.assertFalse(cold.cache_hit)
            self.assertEqual(warm.injected, 0)
            self.assertEqual(warm.existing, 1)
            self.assertEqual(warm.payload_bytes, 0)
            self.assertTrue(warm.cache_hit)


class PrometheusMetricTest(unittest.TestCase):
    def test_current_vllm_kv_cache_metric_is_parsed(self) -> None:
        metrics = (
            '# HELP vllm:kv_cache_usage_perc KV-cache usage.\n'
            'vllm:kv_cache_usage_perc{engine="0",model_name="model"} 0.25\n'
        )
        self.assertEqual(PrometheusCache._metric_value(metrics), 0.25)


class StreamObservationTest(unittest.TestCase):
    def test_role_only_chunk_is_not_counted_as_first_token(self) -> None:
        observation = observe_stream_payload(
            '{"choices":[{"delta":{"role":"assistant","content":""}}]}'
        )
        self.assertFalse(observation["has_token"])
        self.assertEqual(observation["output_chars"], 0)

    def test_reasoning_content_and_usage_are_observed(self) -> None:
        observation = observe_stream_payload(
            '{"choices":[{"delta":{"reasoning_content":"abc"}}],'
            '"usage":{"completion_tokens":7}}'
        )
        self.assertTrue(observation["has_token"])
        self.assertEqual(observation["output_chars"], 3)
        self.assertEqual(observation["completion_tokens"], 7)

    def test_token_is_detected_across_transport_chunks(self) -> None:
        observer = OpenAIStreamObserver()
        first = observer.feed(b'data: {"choices":[{"delta":{"cont')
        second = observer.feed(b'ent":"x"}}]}\n\n')
        self.assertFalse(first["has_token"])
        self.assertTrue(second["has_token"])


class ClientTTFTTest(unittest.IsolatedAsyncioTestCase):
    async def test_client_ttft_ignores_role_only_chunk_and_keeps_meta(self) -> None:
        class FakeResponse:
            status_code = 200

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def aiter_lines(self):
                for line in (
                    'data: {"choices":[{"delta":{"role":"assistant","content":""}}]}',
                    "",
                    'data: {"choices":[{"delta":{"content":"hello"}}]}',
                    "",
                    "event: cacheroute_meta",
                    'data: {"trace":{"selected_instance_id":"inst-1"}}',
                    "",
                    "data: [DONE]",
                ):
                    yield line

        class FakeClient:
            def stream(self, *args, **kwargs):
                return FakeResponse()

        status, meta, metrics = await perf_client.read_chat_stream_meta(
            FakeClient(),
            "http://example.test/v1/chat/completions",
            {},
            {},
            request_start_ts=time.time() - 0.1,
        )
        self.assertEqual(status, 200)
        self.assertEqual(meta["trace"]["selected_instance_id"], "inst-1")
        self.assertGreaterEqual(metrics["client_first_chunk_ms"], 90)
        self.assertGreaterEqual(metrics["client_ttft_ms"], metrics["client_first_chunk_ms"])


class ControlPortTest(unittest.TestCase):
    def test_instance_registration_advertises_control_port(self) -> None:
        tree = ast.parse(
            (ROOT / "instance" / "instance_api.py").read_text(encoding="utf-8")
        )
        registration_meta = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr != "register":
                continue
            meta_keyword = next(
                (keyword for keyword in node.keywords if keyword.arg == "meta"),
                None,
            )
            if meta_keyword is not None and isinstance(meta_keyword.value, ast.Dict):
                registration_meta.append(meta_keyword.value)

        self.assertEqual(len(registration_meta), 1)
        meta = registration_meta[0]
        entries = {
            key.value: value
            for key, value in zip(meta.keys, meta.values)
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        self.assertIn("control_port", entries)
        self.assertIsInstance(entries["control_port"], ast.Name)
        self.assertEqual(entries["control_port"].id, "cp_port")

    def test_task_uses_registered_control_port(self) -> None:
        task = ProxyTask(
            request_id=1,
            req_obj=object(),
            instance_body={},
            instance_id="inst-0",
            instance_host="127.0.0.1",
            instance_port=19001,
            url_path="/v1/chat/completions",
            instance_control_port=19101,
        )
        self.assertEqual(task.resolve_instance_control_port(9002), 19101)

    def test_task_falls_back_for_invalid_control_port(self) -> None:
        task = ProxyTask(
            request_id=1,
            req_obj=object(),
            instance_body={},
            instance_id="inst-0",
            instance_host="127.0.0.1",
            instance_port=19001,
            url_path="/v1/chat/completions",
            instance_control_port=-1,
        )
        self.assertEqual(task.resolve_instance_control_port(9002), 9002)


if __name__ == "__main__":
    unittest.main()
