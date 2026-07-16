#!/usr/bin/env python3
"""Warm every vLLM engine directly without updating routing policies."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from urllib.request import Request, urlopen
from typing import Any, Dict, List


def warm_one(
    base_port: int,
    instance_index: int,
    request_index: int,
    model: str,
    max_tokens: int,
    timeout_s: float,
) -> Dict[str, Any]:
    port = base_port + instance_index
    payload = json.dumps(
        {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": f"Reply with OK. Warm-up request {request_index}.",
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": False,
        }
    ).encode("utf-8")
    request = Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout_s) as response:
        response.read()
        status_code = int(response.status)
    return {
        "instance_index": instance_index,
        "port": port,
        "request_index": request_index,
        "status_code": status_code,
    }


async def run(args: argparse.Namespace) -> List[Dict[str, Any]]:
    tasks = [
        asyncio.to_thread(
            warm_one,
            base_port=args.base_port,
            instance_index=instance_index,
            request_index=request_index,
            model=args.model,
            max_tokens=args.max_tokens,
            timeout_s=args.timeout_s,
        )
        for instance_index in range(args.instance_count)
        for request_index in range(args.requests_per_instance)
    ]
    return await asyncio.gather(*tasks)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send direct, concurrent warm-up requests to every vLLM engine."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--instance-count", type=int, required=True)
    parser.add_argument("--requests-per-instance", type=int, default=2)
    parser.add_argument("--base-port", type=int, default=18000)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.instance_count < 1:
        raise ValueError("--instance-count must be positive")
    if args.requests_per_instance < 0:
        raise ValueError("--requests-per-instance must be non-negative")
    if args.max_tokens < 1:
        raise ValueError("--max-tokens must be positive")

    results = asyncio.run(run(args)) if args.requests_per_instance else []
    output = json.dumps({"requests": len(results), "results": results}, indent=2)
    print(output)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
