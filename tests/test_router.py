from __future__ import annotations

import pytest
from pydantic import BaseModel

from quorum.config import Settings
from quorum.llm.providers import ModelRegistry, ModelSpec, ModelTier
from quorum.llm.router import (
    NoProvidersConfigured,
    QuotaExhausted,
    Router,
    _strip_code_fences,
)

GEMINI_BALANCED = ModelSpec(
    name="fake-gemini-balanced", provider="gemini", tier=ModelTier.BALANCED,
    rpm=2, rpd=10, tpm=10_000, context_tokens=100_000, supports_json_schema=True,
)
GROQ_BALANCED = ModelSpec(
    name="fake-groq-balanced", provider="groq", tier=ModelTier.BALANCED,
    rpm=5, rpd=50, tpm=6_000, context_tokens=100_000,
)
GROQ_FAST = ModelSpec(
    name="fake-groq-fast", provider="groq", tier=ModelTier.FAST,
    rpm=30, rpd=1_000, tpm=6_000, context_tokens=100_000,
)


@pytest.fixture
def test_registry() -> ModelRegistry:
    return ModelRegistry([GEMINI_BALANCED, GROQ_BALANCED, GROQ_FAST])


@pytest.fixture
def router(settings, test_registry, cache, quota) -> Router:
    return Router(
        settings=settings, model_registry=test_registry, cache=cache, quota=quota, max_wait_s=0.0
    )


def stub_providers(router: Router, monkeypatch, *, gemini=None, groq=None):
    """Replace real network calls. Returns the list of (model, prompt) calls made."""
    calls: list[tuple[str, str]] = []

    def make(handler, default_text):
        def _call(spec, prompt, system, temperature, max_tokens, json_mode,
                  response_schema, thinking=False):
            calls.append((spec.name, prompt))
            if handler is not None:
                return handler(spec, prompt, len(calls))
            return default_text, 10, 5

        return _call

    monkeypatch.setattr(router, "_call_gemini", make(gemini, "gemini says hi"))
    monkeypatch.setattr(router, "_call_groq", make(groq, "groq says hi"))
    return calls


# --- routing ---------------------------------------------------------------


def test_prefers_model_with_largest_daily_headroom(router, monkeypatch):
    """Scarce quota must be held in reserve, so the 50/day model is tried before
    the 10/day one even though both are BALANCED."""
    calls = stub_providers(router, monkeypatch)
    resp = router.complete("hello")

    assert calls[0][0] == GROQ_BALANCED.name
    assert resp.model == GROQ_BALANCED.name
    assert not resp.degraded


def test_falls_back_to_next_model_when_quota_blocked(router, monkeypatch, quota):
    calls = stub_providers(router, monkeypatch)
    for _ in range(GROQ_BALANCED.rpd):
        quota.record(GROQ_BALANCED.key, 1)

    resp = router.complete("hello")
    assert resp.model == GEMINI_BALANCED.name
    assert calls[0][0] == GEMINI_BALANCED.name


def test_degrades_to_lower_tier_when_whole_tier_exhausted(router, monkeypatch, quota):
    """Degraded output beats no output - but the response must say so."""
    stub_providers(router, monkeypatch)
    for spec in (GROQ_BALANCED, GEMINI_BALANCED):
        for _ in range(spec.rpd):
            quota.record(spec.key, 1)

    resp = router.complete("hello", tier=ModelTier.BALANCED)
    assert resp.model == GROQ_FAST.name
    assert resp.degraded, "a fallback to a cheaper tier must be visible in metrics"


def test_provider_rate_limit_error_moves_to_next_model(router, monkeypatch):
    def groq_429(spec, prompt, call_no):
        raise RuntimeError("429 Too Many Requests: rate limit exceeded")

    calls = stub_providers(router, monkeypatch, groq=groq_429)
    resp = router.complete("hello")

    assert resp.model == GEMINI_BALANCED.name
    assert [c[0] for c in calls][:2] == [GROQ_BALANCED.name, GEMINI_BALANCED.name]


def test_transient_error_moves_to_next_model(router, monkeypatch):
    def groq_boom(spec, prompt, call_no):
        raise RuntimeError("connection reset")

    stub_providers(router, monkeypatch, groq=groq_boom)
    assert router.complete("hello").model == GEMINI_BALANCED.name


def test_raises_when_every_model_is_exhausted(router, monkeypatch, quota):
    stub_providers(router, monkeypatch)
    for spec in (GROQ_BALANCED, GEMINI_BALANCED, GROQ_FAST):
        for _ in range(spec.rpd):
            quota.record(spec.key, 1)

    with pytest.raises(QuotaExhausted) as exc:
        router.complete("hello")
    assert "exhausted" in str(exc.value).lower()


def test_no_api_keys_raises_clear_error(test_registry, cache, quota):
    bare = Settings(GEMINI_API_KEY=None, GROQ_API_KEY=None)
    router = Router(settings=bare, model_registry=test_registry, cache=cache, quota=quota)
    with pytest.raises(NoProvidersConfigured):
        router.complete("hello")


# --- caching ---------------------------------------------------------------


def test_identical_call_is_served_from_cache(router, monkeypatch):
    calls = stub_providers(router, monkeypatch)

    first = router.complete("same prompt")
    second = router.complete("same prompt")

    assert len(calls) == 1, "the second call must not hit a provider"
    assert second.cached and not first.cached
    assert second.text == first.text


def test_cached_call_consumes_no_quota(router, monkeypatch, quota):
    stub_providers(router, monkeypatch)
    router.complete("p")
    after_first = quota.usage(GROQ_BALANCED.key)["requests_last_day"]
    router.complete("p")
    assert quota.usage(GROQ_BALANCED.key)["requests_last_day"] == after_first


def test_cache_hit_works_even_when_quota_exhausted(router, monkeypatch, quota):
    """This is what makes re-running an eval sweep free."""
    stub_providers(router, monkeypatch)
    router.complete("p")
    for _ in range(GROQ_BALANCED.rpd):
        quota.record(GROQ_BALANCED.key, 1)

    resp = router.complete("p")
    assert resp.cached
    assert resp.model == GROQ_BALANCED.name


def test_different_temperature_misses_cache(router, monkeypatch):
    calls = stub_providers(router, monkeypatch)
    router.complete("p", temperature=0.0)
    router.complete("p", temperature=0.9)
    assert len(calls) == 2


# --- structured output -----------------------------------------------------


class Extraction(BaseModel):
    description: str
    owner: str


def test_structured_parses_valid_json(router, monkeypatch):
    def ok(spec, prompt, call_no):
        return '{"description": "send the spec", "owner": "Yug"}', 20, 10

    stub_providers(router, monkeypatch, groq=ok)
    parsed, resp = router.structured("extract", Extraction)

    assert parsed.description == "send the spec"
    assert parsed.owner == "Yug"
    assert resp.parse_retries == 0


def test_structured_strips_markdown_fences(router, monkeypatch):
    def fenced(spec, prompt, call_no):
        return '```json\n{"description": "d", "owner": "o"}\n```', 20, 10

    stub_providers(router, monkeypatch, groq=fenced)
    parsed, _ = router.structured("extract", Extraction)
    assert parsed.owner == "o"


def test_structured_repairs_malformed_json(router, monkeypatch):
    """Groq has no server-side schema enforcement, so the repair loop is the
    only thing standing between a bad reply and a crashed pipeline."""

    def flaky(spec, prompt, call_no):
        if call_no == 1:
            return '{"description": "missing owner"}', 20, 10
        return '{"description": "fixed", "owner": "Yug"}', 20, 10

    calls = stub_providers(router, monkeypatch, groq=flaky)
    parsed, resp = router.structured("extract", Extraction)

    assert parsed.owner == "Yug"
    assert resp.parse_retries == 1
    assert len(calls) == 2
    assert "could not be parsed" in calls[1][1], "the repair prompt must include the error"


def test_structured_gives_up_after_retries(router, monkeypatch):
    def always_bad(spec, prompt, call_no):
        return "{not json at all", 5, 5

    stub_providers(router, monkeypatch, groq=always_bad)
    with pytest.raises(ValueError, match="Could not obtain valid Extraction"):
        router.structured("extract", Extraction, max_parse_retries=1)


def test_schema_is_included_in_the_prompt(router, monkeypatch):
    def ok(spec, prompt, call_no):
        return '{"description": "d", "owner": "o"}', 5, 5

    calls = stub_providers(router, monkeypatch, groq=ok)
    router.structured("extract commitments", Extraction)
    assert "JSON Schema" in calls[0][1]
    assert "description" in calls[0][1]


# --- reporting -------------------------------------------------------------


def test_stats_report_cache_and_quota(router, monkeypatch):
    stub_providers(router, monkeypatch)
    router.complete("a")
    router.complete("a")

    stats = router.stats()
    assert stats["cache"]["hits"] == 1
    assert stats["calls_by_model"][GROQ_BALANCED.key] == 1
    assert GROQ_BALANCED.key in stats["quota"]


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("{}", "{}"),
        ("```json\n{}\n```", "{}"),
        ("```\n{\"a\": 1}\n```", '{"a": 1}'),
        ("  {\"a\": 1}  ", '{"a": 1}'),
    ],
)
def test_strip_code_fences(raw, expected):
    assert _strip_code_fences(raw) == expected


# --- a withdrawn model is not a rate limit ----------------------------------


class GoneModel(Exception):
    """Shaped like Groq's NotFoundError for a retired model."""

    def __str__(self) -> str:
        return ("Error code: 404 - {'error': {'message': 'The model "
                "`llama-3.3-70b-versatile` does not exist or you do not have "
                "access to it.', 'code': 'model_not_found'}}")


def test_a_withdrawn_model_is_told_apart_from_a_rate_limit():
    """Groq removed both llama models in this registry between one week and the
    next. A rate limit is worth retrying in a minute; a withdrawal never is."""
    from quorum.llm.router import Router

    assert Router._is_model_gone(GoneModel())
    assert not Router._is_rate_limit_error(GoneModel())
    assert not Router._is_model_gone(RuntimeError("429 rate limit exceeded"))


def test_a_withdrawn_model_is_dropped_for_the_rest_of_the_run(settings, cache, quota):
    """Left in the chain, a retired model costs a failed round trip on every
    single call - which is exactly what a real run looked like."""
    from quorum.llm.providers import ModelTier
    from quorum.llm.router import Router

    router = Router(settings=settings, cache=cache, quota=quota)
    chain = router.registry.fallback_chain(ModelTier.BALANCED, ["gemini", "groq"])
    dead, alive = chain[0], chain[1]
    attempts: list[str] = []

    def fake_call(spec, *args, **kwargs):
        attempts.append(spec.key)
        if spec.key == dead.key:
            raise GoneModel()
        return "answered", 5, 5

    router._call_gemini = fake_call
    router._call_groq = fake_call

    first = router.complete("hello", tier=ModelTier.BALANCED)
    assert first.text == "answered"
    assert attempts == [dead.key, alive.key]

    router.cache.enabled = False
    router.complete("a different prompt", tier=ModelTier.BALANCED)

    assert attempts.count(dead.key) == 1, "the retired model was tried again"
    assert dead.key in router._retired
