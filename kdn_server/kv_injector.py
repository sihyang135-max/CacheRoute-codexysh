# kdn_server/kv_inject.py
from __future__ import annotations

import argparse
import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import redis


def b64url_decode(s: str) -> bytes:
    # urlsafe base64 without '=' padding -> restore padding then decode
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


@dataclass
class InjectResult:
    injected: int
    existing: int
    total_entries: int
    cache_hit: bool
    missing_files: int
    keys_b64url: List[str]
    payload_bytes: int                  # 本次实际成功注入到 Redis 的 KV dump 总字节数
    payload_files: int


class KVCacheInjector:
    """
    从 KV_database/<kid>/manifest.jsonl 读取 key -> dump_file 映射，
    将 dump 的 value bytes 原样注入到目标 Redis。
    """

    def __init__(
        self,
        redis_host: str = "127.0.0.1",
        redis_port: int = 6379,
        redis_db: int = 0,
        redis_password: Optional[str] = None,
        socket_timeout_s: int = 30,
    ):
        self.rds = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            password=redis_password,
            decode_responses=False,  # 必须 False：保证 key/value 都是 bytes
            socket_timeout=socket_timeout_s,
        )

    @staticmethod
    def _read_manifest(kv_dir: str) -> List[Tuple[str, bytes, Path]]:
        kv_path = Path(kv_dir).resolve()
        manifest_path = kv_path / "manifest.jsonl"
        if not manifest_path.exists():
            raise FileNotFoundError(f"manifest.jsonl not found: {manifest_path}")

        records: List[Tuple[str, bytes, Path]] = []
        with manifest_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception as exc:
                    raise ValueError(
                        f"invalid json at {manifest_path}:{line_no}: {exc}"
                    ) from exc
                key_b64 = rec.get("key_b64url")
                rel_file = rec.get("file")
                if not isinstance(key_b64, str) or not isinstance(rel_file, str):
                    raise ValueError(
                        f"manifest record missing fields at line {line_no}: {rec}"
                    )
                records.append(
                    (key_b64, b64url_decode(key_b64), kv_path / rel_file)
                )
        return records

    def probe_kv_dir(self, kv_dir: str) -> Tuple[bool, int, int]:
        """Return whether every manifest key is already resident in Redis."""
        records = self._read_manifest(kv_dir)
        if not records:
            return False, 0, 0
        pipe = self.rds.pipeline(transaction=False)
        for _, key, _ in records:
            pipe.exists(key)
        states = pipe.execute()
        existing = sum(1 for state in states if int(state) > 0)
        return existing == len(records), existing, len(records)

    def inject_kv_dir(self, kv_dir: str, return_keys: bool = True) -> InjectResult:
        """Inject only missing content-addressed keys using atomic SET NX."""
        records = self._read_manifest(kv_dir)
        injected = 0
        existing = 0
        missing_files = 0
        keys_b64url: List[str] = []
        payload_bytes = 0
        payload_files = 0

        pipe = self.rds.pipeline(transaction=False)
        for _, key, _ in records:
            pipe.exists(key)
        states = pipe.execute() if records else []

        for (key_b64, key, dump_path), state in zip(records, states):
            if int(state) > 0:
                existing += 1
                if return_keys:
                    keys_b64url.append(key_b64)
                continue
            if not dump_path.exists():
                missing_files += 1
                continue

            value = dump_path.read_bytes()
            if self.rds.set(key, value, nx=True):
                injected += 1
                payload_bytes += len(value)
                payload_files += 1
            else:
                existing += 1
            if return_keys:
                keys_b64url.append(key_b64)

        total_entries = len(records)
        return InjectResult(
            injected=injected,
            existing=existing,
            total_entries=total_entries,
            cache_hit=(
                total_entries > 0
                and existing == total_entries
                and injected == 0
                and missing_files == 0
            ),
            missing_files=missing_files,
            keys_b64url=keys_b64url,
            payload_bytes=payload_bytes,
            payload_files=payload_files,
        )


def main():
    ap = argparse.ArgumentParser(description="Inject KVCache dumps into target Redis (no flushdb).")
    ap.add_argument("--kv-dir", required=True, help="KV_database/<kid> directory containing manifest.jsonl")
    ap.add_argument("--redis-host", default="127.0.0.1")
    ap.add_argument("--redis-port", type=int, default=6379)
    ap.add_argument("--redis-db", type=int, default=0)
    ap.add_argument("--redis-password", default=None)
    ap.add_argument("--no-return-keys", action="store_true", help="Do not print key list (faster, less output)")
    args = ap.parse_args()

    injector = KVCacheInjector(
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        redis_db=args.redis_db,
        redis_password=args.redis_password,
    )
    res = injector.inject_kv_dir(args.kv_dir, return_keys=not args.no_return_keys)

    # 打印结果：测试阶段你就能直接看到 keys 是否符合预期
    out: Dict = {
        "kv_dir": str(Path(args.kv_dir).resolve()),
        "injected": res.injected,
        "existing": res.existing,
        "total_entries": res.total_entries,
        "cache_hit": res.cache_hit,
        "missing_files": res.missing_files,
        "payload_bytes": res.payload_bytes,
        "payload_files": res.payload_files,
    }
    if not args.no_return_keys:
        out["keys_b64url"] = res.keys_b64url

    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
