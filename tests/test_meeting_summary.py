"""The meeting summary.

`MeetingRecord.summary` existed unpopulated for the whole life of the project,
because obligations were the interesting part to build first. The rules it has
to hold to are the ones lecture notes already established: two passes because of
the token ceiling, and nothing added that nobody said, because this is read
months later as a record.
"""

from __future__ import annotations

from datetime import date

import pytest

from quorum.analysis.meeting import (
    MeetingSummariser,
    MeetingSummary,
    MeetingSynthesis,
    SegmentDigest,
    _dedupe,
)
from quorum.models import (
    Assignee,
    Commitment,
    Deadline,
    Decision,
    Evidence,
    MeetingRecord,
    Segment,
)


class StubRouter:
    """Scripted structured output, queued by schema."""

    def __init__(self, digests=None, synthesis=None, fail_on=None) -> None:
        self.digests = list(digests or [])
        self.synthesis = synthesis
        self.fail_on = fail_on
        self.calls: list[str] = []

    def structured(self, prompt, schema, **kwargs):
        from quorum.llm.router import LLMResponse

        self.calls.append(schema.__name__)
        if self.fail_on == schema.__name__:
            raise RuntimeError("quota exhausted")

        if schema is SegmentDigest:
            result = self.digests.pop(0) if self.digests else SegmentDigest()
        else:
            result = self.synthesis or MeetingSynthesis(summary="", key_points=[])
        return result, LLMResponse(text="", model="stub", provider="stub", total_tokens=100)


def segments(count: int) -> list[Segment]:
    return [
        Segment(meeting_id="mtg_1", start_index=i * 2, end_index=i * 2 + 1)
        for i in range(count)
    ]


def test_points_are_gathered_per_segment_then_synthesised(transcript):
    """Two passes, not one: a 40-minute meeting is ~30k tokens and cannot be
    summarised in a single call under a 6,000 tokens/minute ceiling."""
    router = StubRouter(
        digests=[
            SegmentDigest(topic="ingestion", points=["Sam raised a cost concern"]),
            SegmentDigest(topic="schema", points=["Priya wanted the freeze first"]),
        ],
        synthesis=MeetingSynthesis(
            summary="They disagreed about sequencing.",
            key_points=["Cost versus schedule was the disagreement"],
        ),
    )

    result = MeetingSummariser(router=router).summarise(transcript, segments(2))

    assert router.calls == ["SegmentDigest", "SegmentDigest", "MeetingSynthesis"]
    assert result.summary == "They disagreed about sequencing."
    assert result.llm_calls == 3


def test_a_failed_segment_does_not_lose_the_meeting(transcript):
    router = StubRouter(fail_on="SegmentDigest")

    result = MeetingSummariser(router=router).summarise(transcript, segments(2))

    assert result.failed_segments == 2
    assert result.summary == ""


def test_a_failed_synthesis_keeps_the_points(transcript):
    """Minutes without a summary still have value; losing both is the outcome
    worth avoiding."""
    router = StubRouter(
        digests=[SegmentDigest(topic="x", points=["Sam raised a cost concern"])],
        fail_on="MeetingSynthesis",
    )

    result = MeetingSummariser(router=router).summarise(transcript, segments(1))

    assert result.key_points == ["Sam raised a cost concern"]
    assert result.summary == ""


def test_nothing_is_synthesised_from_no_points(transcript):
    """A meeting of pure small talk should not be given a summary invented to
    fill the space."""
    router = StubRouter(digests=[SegmentDigest(topic="", points=[])])

    result = MeetingSummariser(router=router).summarise(transcript, segments(1))

    assert "MeetingSynthesis" not in router.calls
    assert result.summary == ""


def test_repeated_points_are_collapsed():
    """People circle back, and consecutive segments overlap in subject."""
    points = [
        "Sam raised a cost concern about the migration",
        "Sam raised a cost concern about the migration timing",
        "Priya wanted the schema freeze first",
    ]

    assert len(_dedupe(points)) == 2


def test_blank_points_are_dropped():
    assert _dedupe(["", "   ", "a real point"]) == ["a real point"]


# --- rendering ---------------------------------------------------------------


def record_with(commitments=None, decisions=None) -> MeetingRecord:
    return MeetingRecord(
        meeting_id="mtg_1", meeting_date=date(2026, 8, 20), title="Weekly sync",
        commitments=commitments or [], decisions=decisions or [],
    )


def test_the_markdown_carries_the_commitments_with_their_quotes():
    """The commitment list is the part someone will dispute six weeks later."""
    commitment = Commitment(
        description="send the ingestion spec", meeting_id="mtg_1",
        assignee=Assignee(display_name="Priya Raghavan", confidence=0.9),
        deadline=Deadline(resolved=date(2026, 8, 28)),
        evidence=[Evidence(utterance_id="u1", quote="I'll send the ingestion spec")],
    )
    markdown = MeetingSummary(summary="A short meeting.").as_markdown(
        record_with([commitment])
    )

    assert "## Summary" in markdown
    assert "Priya Raghavan" in markdown
    assert "2026-08-28" in markdown
    assert "I'll send the ingestion spec" in markdown


def test_an_undated_commitment_says_so_rather_than_omitting_the_date():
    commitment = Commitment(
        description="look into rate limiting", meeting_id="mtg_1",
        assignee=Assignee(display_name="Sam Okafor", confidence=0.8),
        evidence=[Evidence(utterance_id="u1", quote="I'll look into rate limiting")],
    )

    markdown = MeetingSummary().as_markdown(record_with([commitment]))

    assert "no deadline" in markdown


def test_decisions_are_listed():
    decision = Decision(
        statement="Use Postgres rather than Mongo", meeting_id="mtg_1",
        evidence=[Evidence(utterance_id="u1", quote="let's go with Postgres")],
    )

    assert "Use Postgres" in MeetingSummary().as_markdown(record_with(decisions=[decision]))


def test_an_empty_summary_renders_without_error():
    assert MeetingSummary().as_markdown() .startswith("# Meeting")
