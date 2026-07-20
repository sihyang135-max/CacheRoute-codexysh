#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import requests


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill missing KDN embeddings without deleting or rebuilding KV data."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:9101")
    parser.add_argument(
        "--blocks-dir",
        default="/workspace/llm-stack/CacheRoute/kdn_server/text_database/blocks",
    )
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()

    files = sorted(Path(args.blocks_dir).glob("*.txt"))
    if not files:
        raise SystemExit(f"no txt blocks found under {args.blocks_dir}")

    counts: Counter[str] = Counter()
    print(f"blocks={len(files)}", flush=True)
    for index, path in enumerate(files, start=1):
        response = requests.post(
            f"{args.base_url.rstrip('/')}/knowledge/register_text",
            json={"content": path.read_text(encoding="utf-8")},
            timeout=args.timeout,
        )
        response.raise_for_status()
        counts[str(response.json().get("status"))] += 1
        if index % 10 == 0 or index == len(files):
            print(f"processed={index}/{len(files)}", flush=True)

    status_response = requests.post(
        f"{args.base_url.rstrip('/')}/knowledge/pool_status",
        json={"sample_limit": 0},
        timeout=args.timeout,
    )
    status_response.raise_for_status()
    status = status_response.json()
    total = int(status.get("total_blocks") or 0)
    ready = int(status.get("embedding_ready_blocks") or 0)
    print(f"register_status={dict(counts)}", flush=True)
    print(f"total_blocks={total} embedding_ready_blocks={ready}", flush=True)
    if total == 0 or ready != total:
        print("ERROR: embedding backfill is incomplete", flush=True)
        return 1
    print("OK: all KDN embeddings are ready; existing KV data was not rebuilt", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
