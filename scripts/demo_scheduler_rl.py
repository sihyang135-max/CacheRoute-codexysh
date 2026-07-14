"""Scheduler launcher with model/tokenizer paths supplied by the Docker launcher."""
from __future__ import annotations

import json
import os
import uvicorn

from scheduler import scheduler
from core import config


def main() -> None:
    model_path = os.environ.get("SCHEDULER_MODEL_PATH", config.DEFAULT_MODEL)
    model_name = os.environ.get("SCHEDULER_MODEL_NAME", config.DEFAULT_MODEL_SHORTNAME)
    tokenizer_path = os.environ.get("SCHEDULER_TOKENIZER_PATH", model_path)
    os.environ["SCHEDULER_MODEL_PATH"] = model_path
    os.environ["SCHEDULER_TOKENIZER_MAP"] = json.dumps({model_name: tokenizer_path})
    os.environ["SCHEDULER_KNOWLEDGE_YAML"] = str(config.KNOWLEDGE_YAML_PATH)
    os.environ["SCHEDULER_KDN_BASE_URL"] = config.KDN_BASE_URL.rstrip("/")
    os.environ["SCHEDULER_EMBEDDING_MODEL"] = config.EMBEDDING_MODEL
    os.environ["SCHEDULER_STRATEGY"] = "cacheroute"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    uvicorn.run(scheduler, host=config.SCHEDULER_DP_HOST, port=config.SCHEDULER_DP_PORT, reload=False)


if __name__ == "__main__":
    main()
