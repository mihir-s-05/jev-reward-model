"""Serialization and provenance without importing the GPU training stack."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path
from typing import Any


def canonical(value: Any) -> str:
    # Escape literal chat-control token delimiters inside untrusted action strings.
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).replace("<", "\\u003c").replace(">", "\\u003e")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def append_jsonl(path: Path, value: Any) -> None:
    with path.open("a") as stream:
        stream.write(canonical(value) + "\n")


def provenance(paths: list[str]) -> dict:
    versions = {}
    for package in ("torch", "transformers", "peft", "accelerate", "httpx", "PyYAML"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
                                      text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True))
    except (OSError, subprocess.CalledProcessError):
        sha, dirty = None, None
    return {"python": platform.python_version(), "packages": versions, "git_sha": sha,
            "git_dirty": dirty,
            "data_sha256": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}}
