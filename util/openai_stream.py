from __future__ import annotations

import json
from typing import Any, Dict, Optional, Union


def observe_stream_payload(data: str) -> Dict[str, Any]:
    """Extract token text and usage from one OpenAI-compatible SSE payload."""
    try:
        obj = json.loads(data)
    except Exception:
        return {"has_token": False, "output_chars": 0, "completion_tokens": None}

    parts = []
    for choice in obj.get("choices", []) or []:
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta") or {}
        if not isinstance(delta, dict):
            continue
        for key in ("reasoning_content", "content"):
            value = delta.get(key)
            if isinstance(value, str) and value:
                parts.append(value)

    usage = obj.get("usage") or {}
    completion_tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
    if not isinstance(completion_tokens, int):
        completion_tokens = None
    return {
        "has_token": bool(parts),
        "output_chars": sum(len(part) for part in parts),
        "completion_tokens": completion_tokens,
    }


class OpenAIStreamObserver:
    """Incrementally observes token-bearing data lines across arbitrary chunks."""

    def __init__(self) -> None:
        self._buffer = ""
        self.completion_tokens: Optional[int] = None
        self.output_chars = 0

    def feed(self, chunk: Union[bytes, str]) -> Dict[str, Any]:
        text = chunk.decode("utf-8", errors="replace") if isinstance(chunk, bytes) else str(chunk)
        self._buffer += text
        has_token = False
        while "\n" in self._buffer:
            raw_line, self._buffer = self._buffer.split("\n", 1)
            line = raw_line.strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if not data or data == "[DONE]":
                continue
            observation = observe_stream_payload(data)
            has_token = has_token or bool(observation["has_token"])
            self.output_chars += int(observation["output_chars"])
            if observation["completion_tokens"] is not None:
                self.completion_tokens = int(observation["completion_tokens"])
        return {
            "has_token": has_token,
            "output_chars": self.output_chars,
            "completion_tokens": self.completion_tokens,
        }
