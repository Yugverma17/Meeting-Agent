"""Quota accounting for free-tier providers.

Free tiers impose three simultaneous limits - requests/minute, requests/day and
tokens/minute - and the binding one differs by provider. Groq allows 30 RPM but
only 6,000 TPM, so a single long prompt can exhaust a minute's budget in one
call; Gemini 2.5 Flash allows a comfortable TPM but only 250 requests per day.

Usage is persisted to disk so limits survive process restarts. Without that, an
overnight eval run that crashes and resumes would silently blow the daily quota
and start failing halfway through.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

_MINUTE = 60.0
_DAY = 24 * 60 * 60.0


@dataclass
class QuotaVerdict:
    """Whether a call can proceed, and if not, how long until it can."""

    allowed: bool
    wait_s: float = 0.0
    reason: str = ""


class QuotaTracker:
    """Sliding-window accounting of requests and tokens per model.

    Events are (timestamp, tokens) pairs. Windows are recomputed on read and
    pruned to the longest window we care about (24h), which keeps the state file
    small without a background sweeper.
    """

    def __init__(self, state_path: Path, autosave: bool = True) -> None:
        self.state_path = state_path
        self.autosave = autosave
        self._lock = threading.Lock()
        self._events: dict[str, list[tuple[float, int]]] = {}
        self._load()

    # -- persistence ------------------------------------------------------

    def _load(self) -> None:
        if not self.state_path.exists():
            return
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
            self._events = {k: [tuple(e) for e in v] for k, v in raw.items()}  # type: ignore[misc]
        except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
            log.warning("Could not read quota state (%s); starting fresh", exc)
            self._events = {}

    def _save(self) -> None:
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._events), encoding="utf-8")
            tmp.replace(self.state_path)
        except OSError as exc:
            log.warning("Could not persist quota state: %s", exc)

    # -- internals --------------------------------------------------------

    def _prune(self, key: str, now: float) -> list[tuple[float, int]]:
        events = [e for e in self._events.get(key, []) if now - e[0] < _DAY]
        self._events[key] = events
        return events

    @staticmethod
    def _within(events: list[tuple[float, int]], now: float, window: float):
        return [e for e in events if now - e[0] < window]

    # -- public API -------------------------------------------------------

    def check(
        self,
        key: str,
        *,
        rpm: int | None,
        rpd: int | None,
        tpm: int | None,
        est_tokens: int,
    ) -> QuotaVerdict:
        """Can `key` serve a call costing roughly `est_tokens` right now?"""
        with self._lock:
            now = time.time()
            events = self._prune(key, now)
            minute = self._within(events, now, _MINUTE)

            if rpm is not None and len(minute) >= rpm:
                oldest = min(e[0] for e in minute)
                return QuotaVerdict(False, _MINUTE - (now - oldest) + 0.1, f"{key}: RPM {rpm} reached")

            if tpm is not None:
                used = sum(e[1] for e in minute)
                if used + est_tokens > tpm:
                    if minute:
                        oldest = min(e[0] for e in minute)
                        wait = _MINUTE - (now - oldest) + 0.1
                    else:
                        # Single call exceeds the whole per-minute budget.
                        return QuotaVerdict(
                            False, 0.0, f"{key}: prompt of ~{est_tokens} tok exceeds TPM {tpm}"
                        )
                    return QuotaVerdict(
                        False, wait, f"{key}: TPM {tpm} would be exceeded ({used}+{est_tokens})"
                    )

            if rpd is not None and len(events) >= rpd:
                oldest = min(e[0] for e in events)
                return QuotaVerdict(False, _DAY - (now - oldest) + 1.0, f"{key}: RPD {rpd} exhausted")

            return QuotaVerdict(True)

    def record(self, key: str, tokens: int) -> None:
        with self._lock:
            now = time.time()
            self._prune(key, now)
            self._events.setdefault(key, []).append((now, max(0, tokens)))
            if self.autosave:
                self._save()

    def usage(self, key: str) -> dict[str, int]:
        """Current consumption, for the metrics dashboard."""
        with self._lock:
            now = time.time()
            events = self._prune(key, now)
            minute = self._within(events, now, _MINUTE)
            return {
                "requests_last_minute": len(minute),
                "tokens_last_minute": sum(e[1] for e in minute),
                "requests_last_day": len(events),
                "tokens_last_day": sum(e[1] for e in events),
            }

    def snapshot(self) -> dict[str, dict[str, int]]:
        return {key: self.usage(key) for key in sorted(self._events)}

    def reset(self, key: str | None = None) -> None:
        with self._lock:
            if key is None:
                self._events = {}
            else:
                self._events.pop(key, None)
            if self.autosave:
                self._save()


def estimate_tokens(text: str) -> int:
    """Rough token estimate.

    Deliberately crude (~4 chars/token) and biased slightly high. Exact counts
    need a per-provider tokenizer round-trip, which costs more than the safety
    margin is worth; over-estimating just means we throttle marginally early.
    """
    return max(1, int(len(text) / 3.6) + 16)
