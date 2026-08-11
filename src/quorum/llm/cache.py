"""Content-addressed disk cache for LLM calls.

The point is not latency, it is quota. An eval sweep over 80 meetings makes
thousands of calls; without caching, a single re-run after a scoring bug would
burn a full day of free-tier requests and stall the project for 24 hours. With
it, re-runs are free and deterministic, which is also what makes the reported
metrics reproducible.

Keys hash every input that can change the output (provider, model, system
prompt, user prompt, temperature, max tokens, response schema), so a prompt edit
correctly misses the cache instead of silently serving a stale answer.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    writes: int = 0
    errors: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
            "errors": self.errors,
            "hit_rate": round(self.hit_rate, 4),
        }


class LLMCache:
    """Filesystem cache. One JSON file per call, sharded by key prefix.

    Sharding into 256 subdirectories keeps any single directory small; Windows
    gets noticeably slow listing directories with tens of thousands of entries.
    """

    def __init__(self, root: Path, enabled: bool = True) -> None:
        self.root = root
        self.enabled = enabled
        self.stats = CacheStats()

    @staticmethod
    def make_key(**parts: Any) -> str:
        """Stable hash over the call's semantic inputs."""
        canonical = json.dumps(parts, sort_keys=True, default=str, ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        path = self._path(key)
        if not path.exists():
            self.stats.misses += 1
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.stats.hits += 1
            return payload
        except (json.JSONDecodeError, OSError) as exc:
            # A corrupt entry must never break a run - drop it and treat as miss.
            log.warning("Discarding unreadable cache entry %s: %s", key[:12], exc)
            self.stats.errors += 1
            self.stats.misses += 1
            path.unlink(missing_ok=True)
            return None

    def set(self, key: str, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        path = self._path(key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            tmp.replace(path)
            self.stats.writes += 1
        except OSError as exc:
            log.warning("Could not write cache entry %s: %s", key[:12], exc)
            self.stats.errors += 1

    def clear(self) -> int:
        """Delete every entry. Returns how many files were removed."""
        removed = 0
        if not self.root.exists():
            return 0
        for path in self.root.rglob("*.json"):
            path.unlink(missing_ok=True)
            removed += 1
        return removed

    def size(self) -> tuple[int, int]:
        """(entry count, total bytes)."""
        if not self.root.exists():
            return 0, 0
        files = list(self.root.rglob("*.json"))
        return len(files), sum(f.stat().st_size for f in files)
