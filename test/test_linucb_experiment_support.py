from __future__ import annotations

import unittest
from dataclasses import dataclass
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
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

        strategy.update(warmup.features, reward=-0.5)
        decision = strategy.choose(instances, contexts)
        self.assertEqual(decision.phase, "linucb")
        self.assertEqual(decision.effective_updates, 1)
        self.assertEqual(set(decision.candidate_scores), {"inst-0", "inst-1"})

    def test_least_inflight_prefers_proxy_local_load_snapshot(self) -> None:
        instances = [FakeInstance("inst-0"), FakeInstance("inst-1")]
        chosen = LeastInflightStrategy().select(
            instances,
            hint={"instance_inflight": {"inst-0": 3, "inst-1": 0}},
        )
        self.assertEqual(chosen.instance_id, "inst-1")


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
