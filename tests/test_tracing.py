"""LangSmith tracing.

Two properties matter more than the traces themselves. Tracing must be free
when it is off - not "cheap", but literally the undecorated function - and it
must never be the reason a run fails, because a free-tier observability backend
is exactly the kind of thing that rate-limits at the worst moment.

The third test here is about neither: an inherited `LANGSMITH_TRACING=true` from
some other project's shell must not start uploading meeting transcripts.
"""

from __future__ import annotations

import os

import pytest

from quorum.config import get_settings
from quorum.llm import tracing


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    """Settings are memoised and tracing decisions are memoised on top of them."""
    for name in (
        "LANGSMITH_API_KEY", "LANGSMITH_TRACING", "LANGSMITH_PROJECT",
        "LANGSMITH_ENDPOINT", "LANGCHAIN_TRACING_V2",
    ):
        monkeypatch.delenv(name, raising=False)
    import langsmith.utils

    get_settings.cache_clear()
    tracing.reset_for_tests()
    langsmith.utils.get_env_var.cache_clear()
    yield
    get_settings.cache_clear()
    tracing.reset_for_tests()
    langsmith.utils.get_env_var.cache_clear()


def enable(monkeypatch, key: str = "ls-test-key") -> None:
    """Turn tracing on. The upload itself is severed session-wide in conftest,
    so the real `traceable` wrapping runs and nothing leaves the machine."""
    monkeypatch.setenv("LANGSMITH_API_KEY", key)
    get_settings.cache_clear()
    tracing.reset_for_tests()

    # LangSmith memoises its own environment reads, so a previous test that
    # left LANGSMITH_TRACING=false keeps `traceable` a pass-through no matter
    # what the environment says now.
    import langsmith.utils

    langsmith.utils.get_env_var.cache_clear()


# --- off by default ---------------------------------------------------------


def test_tracing_is_off_without_a_key():
    assert not tracing.tracing_enabled()
    assert not tracing.configure_tracing()


def test_an_untraced_function_is_not_wrapped_at_all():
    """Not a wrapper that checks a flag - the function itself. A decorator that
    re-decides on every call costs something on the hot path forever."""
    def extract(x):
        return x * 2

    decorated = tracing.traced("test.extract")(extract)

    assert decorated is extract


def test_metadata_on_an_untraced_run_is_a_no_op():
    tracing.add_metadata(model="gemini-3.6-flash", cached=False)  # must not raise


def test_trace_url_is_empty_when_untraced():
    assert tracing.trace_url() == ""


# --- not leaking ------------------------------------------------------------


def test_inherited_tracing_from_another_project_is_switched_off(monkeypatch):
    """Meeting transcripts are the last thing to start uploading by accident
    because a shell exported LANGSMITH_TRACING for something else."""
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    get_settings.cache_clear()
    tracing.reset_for_tests()

    assert not tracing.configure_tracing()
    assert os.environ["LANGSMITH_TRACING"] == "false"
    assert os.environ["LANGCHAIN_TRACING_V2"] == "false"


def test_a_key_alone_is_not_enough_when_tracing_is_disabled(monkeypatch):
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-test-key")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    get_settings.cache_clear()
    tracing.reset_for_tests()

    assert not tracing.tracing_enabled()


# --- on ---------------------------------------------------------------------


def test_configuring_exports_what_langsmith_and_langgraph_read(monkeypatch):
    """Both libraries pick tracing up from the environment rather than an API,
    which is why keys live in .env and get exported here instead of by hand."""
    enable(monkeypatch)

    assert tracing.configure_tracing()
    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert os.environ["LANGCHAIN_TRACING_V2"] == "true", "older LangChain reads this name"
    assert os.environ["LANGSMITH_API_KEY"] == "ls-test-key"
    assert os.environ["LANGSMITH_PROJECT"] == "quorum"


def test_a_traced_function_still_returns_its_own_result(monkeypatch):
    enable(monkeypatch)

    @tracing.traced("test.stage")
    def stage(value):
        return {"doubled": value * 2}

    assert stage(21) == {"doubled": 42}


def test_a_traced_function_still_raises_its_own_errors(monkeypatch):
    enable(monkeypatch)

    @tracing.traced("test.failing")
    def failing():
        raise ValueError("the real error")

    with pytest.raises(ValueError, match="the real error"):
        failing()


def test_a_broken_tracing_backend_does_not_break_the_run(monkeypatch):
    """Tracing is a nice-to-have. If instrumenting the function fails, the
    function still has to run."""
    enable(monkeypatch)
    monkeypatch.setattr(
        "langsmith.traceable",
        lambda **kw: (_ for _ in ()).throw(RuntimeError("langsmith is down")),
    )

    @tracing.traced("test.stage")
    def stage():
        return "worked anyway"

    assert stage() == "worked anyway"


def test_metadata_survives_being_called_outside_a_span(monkeypatch):
    enable(monkeypatch)
    tracing.add_metadata(model="llama-3.3-70b-versatile")  # no active run; must not raise


def test_a_span_really_opens_and_metadata_lands_on_it(monkeypatch):
    """Which model answered is decided inside the call, so it has to be
    attached rather than passed in - and it is the first thing worth filtering
    a trace list on when numbers move between runs."""
    from langsmith.run_helpers import get_current_run_tree

    enable(monkeypatch)
    seen = {}

    @tracing.traced("test.router")
    def call():
        tracing.add_metadata(model="gemini-3.6-flash", degraded=True)
        run = get_current_run_tree()
        seen.update(run.extra.get("metadata", {}))
        seen["name"] = run.name

    call()

    assert seen["name"] == "test.router"
    assert seen["model"] == "gemini-3.6-flash" and seen["degraded"] is True


# --- the router is still the router -----------------------------------------


def test_the_router_serves_from_cache_with_tracing_on(monkeypatch, settings, cache, quota):
    """The decorator sits on `complete`, so a mistake there would break every
    model call in the project rather than just the tracing."""
    enable(monkeypatch)
    from quorum.llm.providers import ModelTier
    from quorum.llm.router import Router

    router = Router(settings=settings, cache=cache, quota=quota)
    chain = router.registry.fallback_chain(ModelTier.FAST, settings.configured_providers())
    spec = chain[0]
    cache.set(
        __import__("quorum.llm.cache", fromlist=["LLMCache"]).LLMCache.make_key(
            provider=spec.provider, model=spec.name, system=None, prompt="hello",
            temperature=0.0, max_tokens=2048, json_mode=False, schema=None, thinking=False,
        ),
        {"text": "hi", "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    )

    response = router.complete("hello", tier=ModelTier.FAST)

    assert response.text == "hi" and response.cached
