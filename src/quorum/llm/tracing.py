"""LangSmith tracing, wired at the one place every model call passes through.

The router is the single choke point for LLM traffic, so instrumenting it once
gives a complete picture: which model answered, whether the answer came from
cache, whether the tier was degraded by a quota wall, how many parse retries the
JSON needed. Those four facts explain nearly every surprising number this
project produces, and before tracing they were only visible by reading logs.

Two things this module is careful about.

**Off is the default and off is free.** With no `LANGSMITH_API_KEY` set,
`traced` returns the function unchanged - not a wrapper that checks a flag on
every call. There is no import of `langsmith` and no per-call overhead on a
machine that has not opted in.

**Tracing must never break a run.** A tracing backend is a nice-to-have on a
free tier that can rate-limit or go down. Every failure path here degrades to
"no trace" rather than raising into the caller's stack.
"""

from __future__ import annotations

import functools
import logging
import os
from typing import Any, Callable, TypeVar

from quorum.config import get_settings

log = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

_configured: bool | None = None


def tracing_enabled() -> bool:
    """True when a key is present and tracing is not explicitly switched off."""
    settings = get_settings()
    return bool(settings.langsmith_api_key) and settings.langsmith_tracing


def configure_tracing(force: bool = False) -> bool:
    """Publish credentials into the environment LangSmith and LangGraph read.

    Both libraries pick tracing up from environment variables rather than an
    API, which is why this exports rather than returning a client. Doing it here
    means the keys live in `.env` alongside everything else instead of having to
    be exported by hand before every run - and it is what makes LangGraph trace
    each node with no per-node code.
    """
    global _configured
    if _configured is not None and not force:
        return _configured

    settings = get_settings()
    if not tracing_enabled():
        # Explicitly off, so an inherited LANGSMITH_TRACING from some other
        # project's shell cannot silently start uploading this project's
        # transcripts. Meeting content is the last thing to leak by accident.
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        _configured = False
        return False

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"  # older LangChain reads this name
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key or ""
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
    _configured = True
    log.info("LangSmith tracing on, project %r", settings.langsmith_project)
    return True


def traced(name: str, run_type: str = "chain") -> Callable[[F], F]:
    """Decorate a function so it appears as a span, or leave it exactly as is.

    The decision is made once at import time. That is deliberate: a decorator
    that re-checks a setting on every call costs something on the hot path and
    buys a flexibility nobody needs, since enabling tracing means restarting the
    process anyway.
    """

    def decorate(func: F) -> F:
        if not tracing_enabled():
            return func

        configure_tracing()
        try:
            from langsmith import traceable
        except ImportError:  # pragma: no cover - langsmith ships with langchain-core
            log.debug("langsmith not installed; %s will not be traced", name)
            return func

        try:
            return traceable(name=name, run_type=run_type)(func)
        except Exception as exc:  # noqa: BLE001 - never break a run to add a trace
            log.warning("Could not trace %s (%s); continuing untraced", name, exc)
            return func

    return decorate


def add_metadata(**fields: Any) -> None:
    """Attach facts to the span currently running, if there is one.

    Used to record which model actually answered, since that is decided inside
    the call rather than known at its start - and it is the single most useful
    thing to filter a trace list on when results move between runs.
    """
    if not tracing_enabled():
        return
    try:
        from langsmith.run_helpers import get_current_run_tree

        run = get_current_run_tree()
        if run is not None:
            run.extra.setdefault("metadata", {}).update(fields)
    except Exception:  # noqa: BLE001 - decorative, never fatal
        pass


def trace_url() -> str:
    """Where to go and look. Printed after commands that made model calls."""
    settings = get_settings()
    if not tracing_enabled():
        return ""
    return f"https://smith.langchain.com/o/-/projects/p/{settings.langsmith_project}"


def reset_for_tests() -> None:
    """Clear the memoised decision. Tests change settings between cases."""
    global _configured
    _configured = None
