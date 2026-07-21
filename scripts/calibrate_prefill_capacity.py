#!/usr/bin/env python3
"""Measure isolated Prefill TTFT directly against multiple vLLM engines."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from util.openai_stream import observe_stream_payload


def parse_csv_ints(raw: str, name: str) -> List[int]:
    try:
        values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"{name} must be a comma-separated integer list") from exc
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"{name} values must be positive")
    return values


def build_prompt(run_id: str, length_label: str, request_index: int, repetitions: int) -> str:
    key = f"{run_id}:{length_label}:{request_index}"
    nonce = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return f"nonce={nonce}\n" + (" a" * repetitions)


def percentile_nearest_rank(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of an empty sequence")
    ordered = sorted(float(value) for value in values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def check_vllm(port: int, timeout_s: float) -> None:
    with urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=timeout_s) as response:
        if int(response.status) != 200:
            raise RuntimeError(f"vLLM port {port} health status={response.status}")
        response.read()


def request_once(
    *,
    run_id: str,
    phase: str,
    length_label: str,
    prompt_repetitions: int,
    port: int,
    tp: int,
    request_index: int,
    prompt_index: int,
    model: str,
    timeout_s: float,
) -> Dict[str, Any]:
    prompt = build_prompt(run_id, length_label, prompt_index, prompt_repetitions)
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode("utf-8")
    request = Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    started = time.perf_counter()
    ttft_ms = None
    first_chunk_ms = None
    prompt_tokens = None
    completion_tokens = None
    status_code = None
    error = None
    try:
        with urlopen(request, timeout=timeout_s) as response:
            status_code = int(response.status)
            for raw_line in response:
                now = time.perf_counter()
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if not data or data == "[DONE]":
                    continue
                if first_chunk_ms is None:
                    first_chunk_ms = (now - started) * 1000.0
                observation = observe_stream_payload(data)
                if observation["has_token"] and ttft_ms is None:
                    ttft_ms = (now - started) * 1000.0
                if observation["completion_tokens"] is not None:
                    completion_tokens = int(observation["completion_tokens"])
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                usage = obj.get("usage") or {}
                if isinstance(usage, dict) and isinstance(usage.get("prompt_tokens"), int):
                    prompt_tokens = int(usage["prompt_tokens"])
    except HTTPError as exc:
        status_code = int(exc.code)
        error = f"HTTPError: {exc.reason}"
    except (URLError, TimeoutError, OSError) as exc:
        error = f"{type(exc).__name__}: {exc}"

    wall_ms = (time.perf_counter() - started) * 1000.0
    success = bool(status_code == 200 and ttft_ms is not None and prompt_tokens is not None)
    if not success and error is None:
        error = "missing first token or prompt-token usage"
    return {
        "run_id": run_id,
        "phase": phase,
        "length_label": length_label,
        "prompt_repetitions": prompt_repetitions,
        "port": port,
        "tp": tp,
        "request_index": request_index,
        "prompt_index": prompt_index,
        "status_code": status_code,
        "success": success,
        "ttft_ms": round(ttft_ms, 3) if ttft_ms is not None else None,
        "first_chunk_ms": round(first_chunk_ms, 3) if first_chunk_ms is not None else None,
        "wall_ms": round(wall_ms, 3),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "error": error,
    }


def summarize(records: Iterable[Dict[str, Any]], ports: Sequence[int]) -> Dict[str, Any]:
    measured = [
        record
        for record in records
        if record.get("phase") == "measured" and record.get("success") is True
    ]
    labels = sorted({str(record["length_label"]) for record in measured})
    by_length: Dict[str, Any] = {}
    ratio_samples: Dict[int, List[float]] = {port: [] for port in ports}

    for label in labels:
        rows = [record for record in measured if record["length_label"] == label]
        port_summary: Dict[str, Any] = {}
        medians: Dict[int, float] = {}
        tp_by_port: Dict[int, int] = {}
        for port in ports:
            port_rows = [record for record in rows if int(record["port"]) == port]
            ttfts = [float(record["ttft_ms"]) for record in port_rows]
            prompt_tokens = [int(record["prompt_tokens"]) for record in port_rows]
            if not ttfts:
                continue
            median_ms = statistics.median(ttfts)
            medians[port] = median_ms
            tp_by_port[port] = int(port_rows[0]["tp"])
            port_summary[str(port)] = {
                "tp": tp_by_port[port],
                "requests": len(ttfts),
                "ttft_mean_ms": round(statistics.mean(ttfts), 3),
                "ttft_median_ms": round(median_ms, 3),
                "ttft_p95_ms": round(percentile_nearest_rank(ttfts, 0.95), 3),
                "prompt_tokens_min": min(prompt_tokens),
                "prompt_tokens_max": max(prompt_tokens),
            }

        tp1_medians = [medians[port] for port in medians if tp_by_port[port] == 1]
        if not tp1_medians:
            raise RuntimeError(f"length={label} has no successful TP1 reference")
        tp1_reference_ms = statistics.mean(tp1_medians)
        ratios = {
            str(port): round(tp1_reference_ms / medians[port], 4)
            for port in medians
        }
        for port in medians:
            ratio_samples[port].append(tp1_reference_ms / medians[port])
        by_length[label] = {
            "ports": port_summary,
            "tp1_reference_median_ms": round(tp1_reference_ms, 3),
            "capacity_ratios_relative_to_mean_tp1_median": ratios,
        }

    aggregate_ratios = {
        str(port): round(statistics.median(values), 4)
        for port, values in ratio_samples.items()
        if values
    }
    return {
        "measured_successes": len(measured),
        "by_length": by_length,
        "equal_weight_median_capacity_ratios": aggregate_ratios,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Direct single-start Prefill calibration for four vLLM engines."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--ports", default="18000,18001,18002,18003")
    parser.add_argument("--tp-sizes", default="4,2,1,1")
    parser.add_argument("--length-labels", default="short,medium,long")
    parser.add_argument("--prompt-repetitions", default="480,2016,3959")
    parser.add_argument("--warmup-per-instance", type=int, default=2)
    parser.add_argument("--measured-per-instance", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    ports = parse_csv_ints(args.ports, "--ports")
    tp_sizes = parse_csv_ints(args.tp_sizes, "--tp-sizes")
    repetitions = parse_csv_ints(args.prompt_repetitions, "--prompt-repetitions")
    labels = [item.strip() for item in args.length_labels.split(",") if item.strip()]
    if len(ports) != len(tp_sizes):
        raise ValueError("--ports and --tp-sizes must have the same length")
    if len(labels) != len(repetitions):
        raise ValueError("--length-labels and --prompt-repetitions must have the same length")
    if args.warmup_per_instance < 0 or args.measured_per_instance < 1:
        raise ValueError("warm-up must be non-negative and measured count must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    records_path = args.output_dir / "requests.jsonl"
    summary_path = args.output_dir / "summary.json"
    started_at = datetime.now(timezone.utc).isoformat()
    for port in ports:
        check_vllm(port, args.timeout_s)

    rng = random.Random(args.seed)
    records: List[Dict[str, Any]] = []
    request_index = 0
    prompt_index = 0
    with records_path.open("w", encoding="utf-8") as output:
        for label, prompt_repetitions in zip(labels, repetitions):
            for phase, count in (
                ("warmup", args.warmup_per_instance),
                ("measured", args.measured_per_instance),
            ):
                for _ in range(count):
                    order = list(range(len(ports)))
                    rng.shuffle(order)
                    for instance_index in order:
                        record = request_once(
                            run_id=args.run_id,
                            phase=phase,
                            length_label=label,
                            prompt_repetitions=prompt_repetitions,
                            port=ports[instance_index],
                            tp=tp_sizes[instance_index],
                            request_index=request_index,
                            prompt_index=prompt_index,
                            model=args.model,
                            timeout_s=args.timeout_s,
                        )
                        records.append(record)
                        output.write(json.dumps(record, ensure_ascii=False) + "\n")
                        output.flush()
                        print(json.dumps(record, ensure_ascii=False), flush=True)
                        request_index += 1
                    prompt_index += 1

    for port in ports:
        check_vllm(port, args.timeout_s)

    failures = [record for record in records if record.get("success") is not True]
    summary = {
        "run_id": args.run_id,
        "git_commit": args.expected_commit,
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "ports": ports,
        "tp_sizes": tp_sizes,
        "length_labels": labels,
        "prompt_repetitions": repetitions,
        "warmup_per_instance": args.warmup_per_instance,
        "measured_per_instance": args.measured_per_instance,
        "seed": args.seed,
        "total_requests": len(records),
        "failures": len(failures),
        **summarize(records, ports),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("=== SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
