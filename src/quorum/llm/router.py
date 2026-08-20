"""Quota-aware router over free-tier LLM providers.

Responsibilities, in order of importance to the project:

1. Never lose work to a quota wall. Every request walks a fallback chain across
   models and providers; only when the whole chain is exhausted does it fail.
2. Make re-runs free. Identical calls are served from disk, so re-scoring an
   eval sweep costs zero quota and produces identical numbers.
3. Stay honest. Every response records which model actually answered and whether
   it was a degraded fallback, so metrics can be broken down by model rather
   than silently averaging over a mix.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from quorum.config import LLM_CACHE_DIR, CACHE_DIR, get_settings
from quorum.llm.cache import LLMCache
from quorum.llm.providers import (
    THINKING_LEVEL_OFF,
    ModelSpec,
    ModelTier,
    registry as default_registry,
)
from quorum.llm.ratelimit import QuotaTracker, estimate_tokens
from quorum.llm.tracing import add_metadata, traced

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class QuotaExhausted(RuntimeError):
    """Every model in the fallback chain is rate-limited or out of daily quota."""


class NoProvidersConfigured(RuntimeError):
    """No API keys present. Set GEMINI_API_KEY and/or GROQ_API_KEY in .env."""


@dataclass
class LLMResponse:
    text: str
    model: str
    provider: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_s: float = 0.0
    cached: bool = False
    degraded: bool = False
    """True when the request was served by a lower tier than asked for, because
    the preferred tier was out of quota. Worth surfacing in metrics: a run where
    30% of extractions were degraded is not comparable to one where none were."""

    attempts: list[str] = field(default_factory=list)
    parse_retries: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "provider": self.provider,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "latency_s": round(self.latency_s, 3),
            "cached": self.cached,
            "degraded": self.degraded,
            "attempts": self.attempts,
            "parse_retries": self.parse_retries,
        }


class Router:
    def __init__(
        self,
        *,
        settings=None,
        model_registry=None,
        cache: LLMCache | None = None,
        quota: QuotaTracker | None = None,
        max_wait_s: float = 65.0,
    ) -> None:
        self.settings = settings or get_settings()
        self.registry = model_registry or default_registry
        self.cache = cache or LLMCache(LLM_CACHE_DIR, enabled=self.settings.cache_enabled)
        self.quota = quota or QuotaTracker(CACHE_DIR / "quota_state.json")
        self.max_wait_s = max_wait_s
        """How long a call may block waiting for a rate-limit window to roll over.
        Slightly over a minute by default so a TPM stall resolves rather than
        failing; set to 0 for interactive paths that must not hang."""

        self._clients: dict[str, Any] = {}
        self._call_counts: dict[str, int] = {}
        self._retired: set[str] = set()
        """Models the provider has withdrawn. Discovered at run time and skipped
        thereafter, because a 404 will not become a 200 by waiting."""

    # -- provider clients (lazy so importing needs no credentials) ---------

    def _gemini(self):
        if "gemini" not in self._clients:
            from google import genai

            if not self.settings.gemini_api_key:
                raise NoProvidersConfigured("GEMINI_API_KEY is not set")
            self._clients["gemini"] = genai.Client(api_key=self.settings.gemini_api_key)
        return self._clients["gemini"]

    def _groq(self):
        if "groq" not in self._clients:
            from groq import Groq

            if not self.settings.groq_api_key:
                raise NoProvidersConfigured("GROQ_API_KEY is not set")
            self._clients["groq"] = Groq(api_key=self.settings.groq_api_key)
        return self._clients["groq"]

    # -- provider calls ---------------------------------------------------

    def _call_gemini(
        self,
        spec: ModelSpec,
        prompt: str,
        system: str | None,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
        response_schema: type[BaseModel] | None,
        thinking: bool = False,
    ) -> tuple[str, int, int]:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        if not thinking:
            # Measured 6x fewer output tokens and 2.4x lower latency for
            # identical answers on extraction work. Left on, reasoning also
            # silently eats max_output_tokens and truncates the JSON mid-object.
            try:
                config.thinking_config = types.ThinkingConfig(
                    thinking_level=THINKING_LEVEL_OFF
                )
            except (TypeError, ValueError):  # pragma: no cover - SDK too old
                log.debug("thinking_level unsupported by this SDK; leaving default")

        if json_mode:
            config.response_mime_type = "application/json"
            if response_schema is not None:
                config.response_schema = response_schema

        resp = self._gemini().models.generate_content(
            model=spec.name, contents=prompt, config=config
        )
        usage = getattr(resp, "usage_metadata", None)
        prompt_tok = getattr(usage, "prompt_token_count", 0) or 0
        out_tok = getattr(usage, "candidates_token_count", 0) or 0
        # Reasoning tokens are billed but reported separately; fold them in so
        # quota accounting reflects what was actually consumed.
        out_tok += getattr(usage, "thoughts_token_count", 0) or 0
        return (resp.text or ""), prompt_tok, out_tok

    def _call_groq(
        self,
        spec: ModelSpec,
        prompt: str,
        system: str | None,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
        response_schema: type[BaseModel] | None,
        thinking: bool = False,
    ) -> tuple[str, int, int]:
        # `thinking` is accepted for signature parity; Groq's reasoning models
        # expose no equivalent switch, so their overhead is absorbed instead.
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict[str, Any] = {
            "model": spec.name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            # Groq has no server-side schema enforcement for these models, so we
            # ask for valid JSON and rely on parse-retry to fix the rest.
            kwargs["response_format"] = {"type": "json_object"}

        resp = self._groq().chat.completions.create(**kwargs)
        text = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        return (
            text,
            getattr(usage, "prompt_tokens", 0) or 0,
            getattr(usage, "completion_tokens", 0) or 0,
        )

    @staticmethod
    def _is_rate_limit_error(exc: Exception) -> bool:
        text = f"{type(exc).__name__} {exc}".lower()
        return any(
            marker in text
            for marker in ("429", "rate limit", "rate_limit", "quota", "resource_exhausted")
        )

    @staticmethod
    def _is_model_gone(exc: Exception) -> bool:
        """Whether the model has been withdrawn rather than merely refused.

        Providers retire free-tier models without warning - Groq removed both
        llama entries in this registry between one week and the next. A rate
        limit is worth retrying in a minute; a withdrawal is not worth retrying
        ever, and the two are worth telling apart because a retired model that
        stays in the chain costs a failed round trip on every single call.
        """
        text = f"{type(exc).__name__} {exc}".lower()
        return "model_not_found" in text or (
            "404" in text and "does not exist" in text
        )

    # -- core -------------------------------------------------------------

    @traced("router.complete", run_type="llm")
    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        tier: ModelTier = ModelTier.BALANCED,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        json_mode: bool = False,
        response_schema: type[BaseModel] | None = None,
        thinking: bool = False,
        purpose: str = "",
    ) -> LLMResponse:
        """Run one completion, walking the fallback chain until something works.

        `thinking` opts into provider reasoning. It defaults to off because
        reasoning tokens dominate the free-tier budget without improving
        extraction; turn it on deliberately for the planner and the critic,
        where multi-step reasoning is the point.

        `purpose` is a free-text label ("extract", "resolve_assignee") used only
        for per-stage cost attribution in the metrics report.
        """
        providers = self.settings.configured_providers()
        if not providers:
            raise NoProvidersConfigured(
                "No API keys found. Copy .env.example to .env and set "
                "GEMINI_API_KEY and/or GROQ_API_KEY."
            )

        chain = self.registry.fallback_chain(tier, providers)
        if not chain:
            raise NoProvidersConfigured(f"No models available for tier {tier} with {providers}")

        est = estimate_tokens((system or "") + prompt) + max_tokens
        attempts: list[str] = []
        blocked: list[str] = []
        deadline = time.time() + self.max_wait_s

        while True:
            for spec in chain:
                if spec.key in self._retired:
                    # Withdrawn earlier in this run. Recorded is not the same as
                    # skipped: the first version of this logged the retirement
                    # and then tried the model again on the very next call,
                    # which is the cost it exists to avoid.
                    continue

                cache_key = LLMCache.make_key(
                    provider=spec.provider,
                    model=spec.name,
                    system=system,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    json_mode=json_mode,
                    schema=response_schema.__name__ if response_schema else None,
                    thinking=thinking,
                )
                hit = self.cache.get(cache_key)
                if hit is not None:
                    add_metadata(
                        model=spec.name, provider=spec.provider, purpose=purpose,
                        cached=True, degraded=spec.tier is not tier,
                    )
                    return LLMResponse(
                        text=hit["text"],
                        model=spec.name,
                        provider=spec.provider,
                        prompt_tokens=hit.get("prompt_tokens", 0),
                        completion_tokens=hit.get("completion_tokens", 0),
                        total_tokens=hit.get("total_tokens", 0),
                        cached=True,
                        degraded=spec.tier is not tier,
                        attempts=[*attempts, f"{spec.key}(cache)"],
                    )

                verdict = self.quota.check(
                    spec.key, rpm=spec.rpm, rpd=spec.rpd, tpm=spec.tpm, est_tokens=est
                )
                if not verdict.allowed:
                    blocked.append(f"{spec.key}: {verdict.reason}")
                    continue

                attempts.append(spec.key)
                started = time.time()
                try:
                    caller = self._call_gemini if spec.provider == "gemini" else self._call_groq
                    text, prompt_tok, out_tok = caller(
                        spec, prompt, system, temperature, max_tokens, json_mode,
                        response_schema, thinking,
                    )
                except Exception as exc:  # noqa: BLE001 - provider SDKs raise varied types
                    if self._is_model_gone(exc):
                        # Permanent. Drop it for the life of the process so the
                        # next call does not pay for the same discovery, and say
                        # so loudly once - the registry needs editing, and
                        # `models --probe` is what confirms the rest.
                        if spec.key not in self._retired:
                            log.error(
                                "%s no longer exists on this account; skipping it "
                                "for the rest of this run. Check `quorum models "
                                "--probe` and update the registry.", spec.key,
                            )
                            self._retired.add(spec.key)
                        blocked.append(f"{spec.key}: withdrawn by the provider")
                        continue
                    if self._is_rate_limit_error(exc):
                        # The provider disagrees with our accounting. Trust the
                        # provider: burn the local budget so we stop trying it.
                        log.warning("%s reported a rate limit; backing off", spec.key)
                        self.quota.record(spec.key, spec.tpm or 0)
                        blocked.append(f"{spec.key}: provider 429")
                        continue
                    log.warning("%s failed (%s); trying next model", spec.key, exc)
                    blocked.append(f"{spec.key}: {type(exc).__name__}")
                    continue

                total = (prompt_tok + out_tok) or estimate_tokens(prompt + text)
                self.quota.record(spec.key, total)
                self._call_counts[spec.key] = self._call_counts.get(spec.key, 0) + 1

                self.cache.set(
                    cache_key,
                    {
                        "text": text,
                        "prompt_tokens": prompt_tok,
                        "completion_tokens": out_tok,
                        "total_tokens": total,
                        "model": spec.name,
                        "provider": spec.provider,
                        "purpose": purpose,
                    },
                )
                # Which model actually answered is decided here, not at the call
                # site, so it has to be attached rather than passed in. It is
                # also the first thing worth filtering a trace list on when
                # numbers move between runs.
                add_metadata(
                    model=spec.name, provider=spec.provider, purpose=purpose,
                    cached=False, degraded=spec.tier is not tier,
                    total_tokens=total, attempts=list(attempts),
                )
                return LLMResponse(
                    text=text,
                    model=spec.name,
                    provider=spec.provider,
                    prompt_tokens=prompt_tok,
                    completion_tokens=out_tok,
                    total_tokens=total,
                    latency_s=time.time() - started,
                    degraded=spec.tier is not tier,
                    attempts=attempts,
                )

            # Whole chain unavailable. Wait for the soonest window to reopen.
            waits = [
                self.quota.check(
                    s.key, rpm=s.rpm, rpd=s.rpd, tpm=s.tpm, est_tokens=est
                ).wait_s
                for s in chain
            ]
            waits = [w for w in waits if w > 0]
            remaining = deadline - time.time()
            if not waits or remaining <= 0:
                raise QuotaExhausted(
                    "All models exhausted for tier "
                    f"{tier.value}. Tried: {attempts or 'none'}. Blocked: {blocked}"
                )
            sleep_for = min(min(waits), remaining)
            log.info("All models rate-limited; waiting %.1fs for quota", sleep_for)
            time.sleep(sleep_for)

    # -- structured output -------------------------------------------------

    @traced("router.structured", run_type="chain")
    def structured(
        self,
        prompt: str,
        schema: type[T],
        *,
        system: str | None = None,
        tier: ModelTier = ModelTier.BALANCED,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        max_parse_retries: int = 2,
        thinking: bool = False,
        purpose: str = "",
    ) -> tuple[T, LLMResponse]:
        """Get a validated Pydantic object back, repairing malformed JSON.

        Only Gemini enforces the schema server-side; Groq returns free-form JSON.
        Rather than special-casing, we validate locally either way and feed the
        validation error back to the model as a repair instruction. Repairs are
        counted so the eval report can show which models needed hand-holding.
        """
        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        instruction = (
            f"{prompt}\n\nRespond with JSON only - no prose, no markdown fences - "
            f"conforming exactly to this JSON Schema:\n{schema_json}"
        )

        attempt_prompt = instruction
        last_error = ""
        response: LLMResponse | None = None

        for attempt in range(max_parse_retries + 1):
            response = self.complete(
                attempt_prompt,
                system=system,
                tier=tier,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=True,
                response_schema=schema,
                thinking=thinking,
                purpose=purpose,
            )
            raw = _strip_code_fences(response.text)
            try:
                parsed = schema.model_validate_json(raw)
                response.parse_retries = attempt
                return parsed, response
            except (ValidationError, ValueError) as exc:
                last_error = str(exc)[:800]
                log.debug("Parse failure on attempt %d: %s", attempt + 1, last_error)
                attempt_prompt = (
                    f"{instruction}\n\nYour previous reply could not be parsed:\n"
                    f"{raw[:1500]}\n\nThe validation error was:\n{last_error}\n\n"
                    "Return corrected JSON only."
                )

        raise ValueError(
            f"Could not obtain valid {schema.__name__} after "
            f"{max_parse_retries + 1} attempts. Last error: {last_error}"
        )

    # -- reporting --------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        return {
            "cache": self.cache.stats.as_dict(),
            "calls_by_model": dict(self._call_counts),
            "quota": self.quota.snapshot(),
        }


def _strip_code_fences(text: str) -> str:
    """Models wrap JSON in ```json fences despite instructions. Unwrap it."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


_router: Router | None = None


def get_router(**kwargs: Any) -> Router:
    """Process-wide router. Sharing one instance keeps quota accounting correct."""
    global _router
    if _router is None or kwargs:
        _router = Router(**kwargs)
    return _router
