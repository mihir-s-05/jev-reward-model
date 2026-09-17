from __future__ import annotations

import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import httpx


class JevClient:
    """Minimal TypeSafe System One client with disk caching and bounded retries."""

    endpoint = "https://api.typesafe.ai/v1/systemone"

    def __init__(self, api_key: str | None = None, cache_dir: str = ".cache/jev"):
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY")
        if not self.api_key:
            raise RuntimeError("TYPESAFE_API_KEY is required for Jev rewards")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.http = httpx.Client(timeout=60.0)

    def evaluate(self, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
        payload = {"state": state, "model": "jev-latest", "questions": questions}
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        key = hashlib.sha256(raw).hexdigest()
        cache = self.cache_dir / f"{key}.json"
        if cache.exists():
            return json.loads(cache.read_text(encoding="utf-8"))

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        for attempt in range(6):
            r = self.http.post(self.endpoint, headers=headers, json=payload)
            if r.status_code < 400:
                result = r.json()
                cache.write_text(json.dumps(result, indent=2), encoding="utf-8")
                return result
            if r.status_code not in {429, 529}:
                r.raise_for_status()
            time.sleep(min(16.0, (2**attempt) + random.random()))
        raise RuntimeError("Jev request failed after retries")

    def terminal_success(self, public_state: dict[str, Any]) -> float:
        result = self.evaluate(public_state, {
            "success": {
                "type": "noul",
                "instructions": (
                    "Has the agent fully completed every required milestone in order, emitted the required "
                    "completion string, and obeyed every persistent constraint? Judge only from the supplied evidence."
                ),
                "criteria": {
                    "true": "All requirements are visibly satisfied with no violation.",
                    "false": "Anything is missing, out of order, unsupported, or violates a constraint.",
                },
            }
        })
        return float(result["answers"]["success"]["noul"])

    def progress(self, public_state: dict[str, Any]) -> float:
        result = self.evaluate(public_state, {
            "progress": {
                "type": "score",
                "instructions": (
                    "Rate verified task progress. Reward only milestones that are explicitly evidenced by actions, "
                    "respect required ordering, and do not count claims unsupported by the trajectory."
                ),
                "criteria": [
                    "No verified progress or trajectory already invalidated by a serious constraint violation.",
                    "Some early progress, but most required work remains.",
                    "Meaningful partial completion with multiple valid milestones.",
                    "Nearly complete; only a small amount of required work remains.",
                    "Fully complete with all milestones and constraints satisfied.",
                ],
            }
        })
        # API score spans level indices [0, 4]; normalize to [0, 1].
        return float(result["answers"]["progress"]["score"]) / 4.0
