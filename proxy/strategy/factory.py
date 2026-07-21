# proxy/strategy/factory.py
from __future__ import annotations

from .base import BaseInstanceStrategy
from .round_robin import RoundRobinStrategy
from .least_inflight import LeastInflightStrategy
from .linucb import LinUCBStrategy


def build_instance_strategy(name: str) -> BaseInstanceStrategy:
    n = (name or "").strip().lower()
    if n in ("rr", "round_robin", "round-robin"):
        return RoundRobinStrategy()
    if n in ("least_inflight", "least-inflight", "li"):
        return LeastInflightStrategy()
    if n in ("linucb", "rl", "bandit"):
        return LinUCBStrategy()
    raise ValueError(f"unknown instance strategy: {name}")
