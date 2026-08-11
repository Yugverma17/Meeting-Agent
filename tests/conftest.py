from __future__ import annotations

from datetime import date

import pytest

from quorum.config import Settings
from quorum.llm.cache import LLMCache
from quorum.llm.ratelimit import QuotaTracker
from quorum.models import Speaker, Transcript, Utterance


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
