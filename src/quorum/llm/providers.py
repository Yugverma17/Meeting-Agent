"""Model registry with free-tier quota limits.

IMPORTANT: the numbers below are free-tier limits as published in August 2026.
Providers change them without notice and they vary by account age. Run

    python -m quorum.cli models --probe

to check each entry against your live account before trusting a long eval run.

Tiers exist so callers ask for a capability level, not a model name. The router
then picks whichever model in that tier still has quota, which is what lets the
system keep working after Gemini's 250 requests/day run out.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ModelTier(str, Enum):
    """What a call needs, rather than which model serves it."""

    FAST = "fast"
    """High-volume, low-stakes: segmentation, classification, filtering."""

    BALANCED = "balanced"
    """The workhorse: extraction, assignee/deadline resolution, summarisation."""

    DEEP = "deep"
    """Planning, critique, contradiction detection. Scarcest quota - use sparingly."""


@dataclass(frozen=True)
class ModelSpec:
    name: str
    provider: str
    tier: ModelTier
    rpm: int | None
    rpd: int | None
    tpm: int | None
    context_tokens: int
    supports_json_schema: bool = False
    """True if the provider enforces a JSON schema server-side. When False the
    router falls back to prompt-level JSON instructions plus parse-retry."""

    notes: str = ""

    @property
    def key(self) -> str:
        return f"{self.provider}:{self.name}"


# --- Gemini free tier -------------------------------------------------------
# Every entry below was verified against a live free-tier account on 2026-08-12.
# The whole gemini-2.5-* family is still *listed* by models.list() but returns
# 404 "no longer available to new users" on generateContent, and
# gemini-3.1-pro-preview returns 429 on the very first request - preview models
# carry no meaningful free quota. Neither is included.
#
# Request/day limits for the 3.x family are not published for free accounts, so
# the numbers here are deliberate under-estimates. That is safe: the router
# treats a provider 429 as authoritative and backs off regardless, so local
# limits only ever save a wasted call.
GEMINI_MODELS = [
    ModelSpec(
        name="gemini-3.5-flash-lite",
        provider="gemini",
        tier=ModelTier.FAST,
        rpm=15,
        rpd=1_000,
        tpm=250_000,
        context_tokens=1_000_000,
        supports_json_schema=True,
        notes="Best value measured: ~0.95s, 96 output tokens, no reasoning overhead.",
    ),
    ModelSpec(
        name="gemini-3.5-flash",
        provider="gemini",
        tier=ModelTier.BALANCED,
        rpm=10,
        rpd=250,
        tpm=250_000,
        context_tokens=1_000_000,
        supports_json_schema=True,
    ),
    ModelSpec(
        name="gemini-3.6-flash",
        provider="gemini",
        tier=ModelTier.DEEP,
        rpm=10,
        rpd=250,
        tpm=250_000,
        context_tokens=1_000_000,
        supports_json_schema=True,
        notes="Strongest Gemini available here. Spends ~474 reasoning tokens per "
        "call unless thinking is suppressed; see THINKING_LEVEL_OFF.",
    ),
]

# --- Groq free tier ---------------------------------------------------------
# Huge daily allowance but a punishing 6k tokens/minute. Groq is the reason the
# pipeline segments transcripts instead of sending them whole.
#
# Re-verified 2026-08-20. `llama-3.1-8b-instant` and `llama-3.3-70b-versatile`
# were both removed and now 404 with "does not exist or you do not have access
# to it" - they are gone from the account entirely, not merely rate-limited.
# That took the only Groq model out of the FAST tier; FAST now starts on Gemini
# and degrades into BALANCED, which `fallback_chain` already does.
#
# Still excluded, re-tested the same day: `qwen/qwen3.6-27b` returns
# 400 json_validate_failed in JSON mode, exactly as it did in August, and spends
# its output allowance on `<think>` before answering.
#
# `groq/compound-mini` and `openai/gpt-oss-safeguard-20b` are alive and handle
# JSON mode, but neither is listed here: this project's own rule is to benchmark
# a model on the real task before trusting it, and neither has been. They are
# candidates, not entries.
GROQ_MODELS = [
    ModelSpec(
        name="openai/gpt-oss-20b",
        provider="groq",
        tier=ModelTier.BALANCED,
        rpm=30,
        rpd=1_000,
        tpm=8_000,
        context_tokens=131_072,
        supports_json_schema=False,
    ),
    ModelSpec(
        name="openai/gpt-oss-120b",
        provider="groq",
        tier=ModelTier.DEEP,
        rpm=30,
        rpd=1_000,
        tpm=8_000,
        context_tokens=131_072,
        supports_json_schema=False,
        notes="Best recall observed on commitment extraction - the only model "
        "that surfaced tentative/musing items rather than firm ones only.",
    ),
]

# --- Gemini reasoning control ----------------------------------------------
# Measured on 2026-08-12 across the three Gemini models above:
#
#   gemini-3.6-flash   default: 474 thinking + 41 output, 4.02s
#                      minimal:   0 thinking + 85 output, 1.64s
#
# Same answer, ~6x fewer output tokens, ~2.4x faster. Reasoning is a resource to
# spend deliberately, not a default to absorb.
#
# NOTE: the obvious knob is the wrong one. `thinking_budget=0` is the 2.5-era
# API and returns 400 INVALID_ARGUMENT on gemini-3.6-flash and
# gemini-3.5-flash-lite. `thinking_level` is the portable one.
THINKING_LEVEL_OFF = "minimal"


# --- Speech-to-text ---------------------------------------------------------
# 28,800 audio seconds/day free - eight hours of meetings, at no cost. This is
# what makes live capture viable on a machine that cannot host a local model.
WHISPER_MODEL = ModelSpec(
    name="whisper-large-v3-turbo",
    provider="groq",
    tier=ModelTier.FAST,
    rpm=20,
    rpd=2_000,
    tpm=None,
    context_tokens=0,
    notes="Limits are audio seconds: 7,200/hour and 28,800/day.",
)

# --- Prompt-injection guard -------------------------------------------------
# A purpose-built injection classifier, free on Groq. Used as the first line of
# defence in the speech-injection suite: a dedicated detector is far harder to
# talk out of its job than a general model asked to police its own input.
# It is a classifier, not a chat model - never route normal traffic here.
GUARD_MODEL = ModelSpec(
    name="meta-llama/llama-prompt-guard-2-86m",
    provider="groq",
    tier=ModelTier.FAST,
    rpm=30,
    rpd=14_400,
    tpm=6_000,
    context_tokens=512,
    notes="512-token context: long utterances must be windowed before scanning.",
)


class ModelRegistry:
    def __init__(self, specs: list[ModelSpec]) -> None:
        self._specs = specs

    def all(self) -> list[ModelSpec]:
        return list(self._specs)

    def by_name(self, name: str) -> ModelSpec | None:
        return next((s for s in self._specs if s.name == name), None)

    def for_tier(self, tier: ModelTier, providers: list[str] | None = None) -> list[ModelSpec]:
        """Candidates for a tier, ordered by daily headroom (most first).

        Preferring the model with the largest daily allowance keeps the scarce
        ones in reserve, so a long eval sweep degrades gracefully instead of
        burning Gemini's 250/day in the first ten minutes.
        """
        candidates = [s for s in self._specs if s.tier is tier]
        if providers is not None:
            candidates = [s for s in candidates if s.provider in providers]
        return sorted(candidates, key=lambda s: (s.rpd or 0), reverse=True)

    def fallback_chain(
        self, tier: ModelTier, providers: list[str] | None = None
    ) -> list[ModelSpec]:
        """Preferred tier first, then progressively cheaper tiers.

        A DEEP request that exhausts Pro's 25/day is better served by a BALANCED
        model than by failing outright - degraded output beats no output, and the
        response records which model actually answered so metrics stay honest.
        """
        order = {
            ModelTier.DEEP: [ModelTier.DEEP, ModelTier.BALANCED, ModelTier.FAST],
            ModelTier.BALANCED: [ModelTier.BALANCED, ModelTier.FAST],
            ModelTier.FAST: [ModelTier.FAST, ModelTier.BALANCED],
        }[tier]
        chain: list[ModelSpec] = []
        for level in order:
            chain.extend(self.for_tier(level, providers))
        return chain


registry = ModelRegistry([*GEMINI_MODELS, *GROQ_MODELS])
