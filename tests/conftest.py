from __future__ import annotations

from datetime import date

import pytest

from quorum.config import Settings
from quorum.llm.cache import LLMCache
from quorum.llm.ratelimit import QuotaTracker
from quorum.models import Speaker, Transcript, Utterance


@pytest.fixture(scope="session", autouse=True)
def never_upload_traces():
    """Cut LangSmith's HTTP layer for the whole session.

    Tracing tests set a fake key to exercise the real `traceable` wrapping.
    LangSmith batches uploads on a background thread that flushes at process
    exit - after any per-test patching has been torn down - so a function-scoped
    patch still let the suite POST to the real API and collect 403s. Severing it
    once, for the session, and deliberately never restoring it is what keeps the
    suite offline.
    """
    try:
        from langsmith.client import Client
    except ImportError:  # pragma: no cover - langsmith ships with langchain-core
        yield
        return

    for name in ("request_with_retries", "_send_multipart_req",
                 "_send_compressed_multipart_req"):
        if hasattr(Client, name):
            setattr(Client, name, lambda self, *args, **kwargs: None)
    yield


@pytest.fixture(autouse=True)
def no_real_quota_and_no_sleeping(tmp_path, monkeypatch):
    """Keep the suite off the real quota file, and stop it ever sleeping.

    `get_router()` memoises a process-wide Router whose default `QuotaTracker`
    reads `.cache/quota_state.json` - the real one. Anything that reaches it
    after a day of real API use sees exhausted windows and *waits*, up to 65
    seconds per call, silently. One run of this suite took 22 minutes instead of
    28 seconds for exactly that reason, right after the AMI evaluation had burned
    through the day's Groq allowance.

    Nothing in the suite is supposed to reach the global router. This makes that
    true rather than assumed, and makes a violation fail fast instead of slowly.
    """
    import quorum.llm.router as router_module

    monkeypatch.setattr(router_module, "_router", None)
    original = router_module.Router.__init__

    def no_waiting(self, *args, **kwargs):
        kwargs.setdefault("max_wait_s", 0.0)
        kwargs.setdefault("quota", QuotaTracker(tmp_path / "quota_state.json"))
        original(self, *args, **kwargs)

    monkeypatch.setattr(router_module.Router, "__init__", no_waiting)
    yield
    router_module._router = None


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Settings with both providers 'configured' but no real network access."""
    return Settings(
        GEMINI_API_KEY="test-gemini-key",
        GROQ_API_KEY="test-groq-key",
        QUORUM_CACHE_ENABLED=True,
        QUORUM_REQUIRE_APPROVAL=True,
    )


@pytest.fixture
def cache(tmp_path) -> LLMCache:
    return LLMCache(tmp_path / "llm_cache", enabled=True)


@pytest.fixture
def quota(tmp_path) -> QuotaTracker:
    return QuotaTracker(tmp_path / "quota.json")


@pytest.fixture
def transcript() -> Transcript:
    """A tiny meeting containing one firm commitment, one musing, and a nickname."""
    priya = Speaker(id="spk_priya", display_name="Priya Raghavan", aliases=["Priya"],
                    email="priya@example.com")
    yug = Speaker(id="spk_yug", display_name="Yug Verma", aliases=["Yug"],
                  email="yug@example.com", github_login="yugverma")
    sam = Speaker(id="spk_sam", display_name="Sam Okafor", aliases=["Sam"],
                  email="sam@example.com")

    lines = [
        (priya.id, "Okay, where are we on the ingestion API?"),
        (yug.id, "Mostly done. I'll have the spec document to you by Friday."),
        (priya.id, "Great. Sam, can you review it once it lands?"),
        (sam.id, "Sure, I'll take a look over the weekend."),
        (priya.id, "We should probably think about rate limiting at some point too."),
        (yug.id, "Yeah, someone should look into that eventually."),
    ]
    utterances = [
        Utterance(id=f"utt_{i}", index=i, speaker_id=spk, text=txt, start_s=i * 15.0,
                  end_s=i * 15.0 + 12.0)
        for i, (spk, txt) in enumerate(lines)
    ]
    return Transcript(
        meeting_id="mtg_test",
        title="Weekly sync",
        meeting_date=date(2026, 3, 9),
        speakers=[priya, yug, sam],
        utterances=utterances,
        source="fixture",
        project_id="proj_test",
    )
