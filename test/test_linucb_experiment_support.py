from __future__ import annotations

import unittest
from dataclasses import dataclass
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import tempfile
import time

from util.openai_stream import OpenAIStreamObserver


ROOT = Path(__file__).resolve().parents[1]


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


def context(prefill: int) -> dict:
    return {
        "prompt_norm": 0.2,
        "kv_norm": 0.8,
        "prefill_norm": prefill / 8.0,
        "decode_norm": 0.0,
        "kv_usage": 0.1,
        "net_norm": 0.4,
        "prefill_raw": prefill,
        "decode_raw": 0,
    }


class LinUCBObservabilityTest(unittest.TestCase):
    def test_ttft_reward_has_an_experiment_scale_not_client_slo(self) -> None:
        self.assertAlmostEqual(ttft_reward(300, scale_ms=1000, clip=5), -0.3)
        self.assertAlmostEqual(ttft_reward(900, scale_ms=1000, clip=5), -0.9)
        self.assertEqual(ttft_reward(9000, scale_ms=1000, clip=5), -5.0)

    def test_decision_exposes_phase_candidates_and_update_count(self) -> None:
        instances = [FakeInstance("inst-0"), FakeInstance("inst-1")]
        contexts = {"inst-0": context(0), "inst-1": context(1)}
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
        contexts = {"inst-0": context(0), "inst-1": context(0)}
        strategy = LinUCBStrategy(alpha=0.1, ridge=1.0, warmup_requests=4)

        for _ in range(4):
            decision = strategy.choose(instances, contexts)
            strategy.update(decision.instance_id, decision.features, reward=-0.5)

        self.assertEqual(strategy.arm_updates, {"inst-0": 2, "inst-1": 2})

    def test_disjoint_models_learn_instance_specific_reward(self) -> None:
        instances = [FakeInstance("inst-0"), FakeInstance("inst-1")]
        contexts = {"inst-0": context(0), "inst-1": context(0)}
        strategy = LinUCBStrategy(alpha=0.0, ridge=1.0, warmup_requests=0)
        features = strategy.choose(instances, contexts).features

        strategy.update("inst-0", features, reward=-1.0)
        strategy.update("inst-1", features, reward=-0.1)

        self.assertEqual(
            strategy.choose(instances, contexts).instance_id,
            "inst-1",
        )

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
