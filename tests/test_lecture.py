from __future__ import annotations

from datetime import date

import pytest

from quorum.analysis import LectureAnalyser
from quorum.analysis.lecture import (
    Concept,
    KeyPoint,
    LectureNotes,
    SegmentNotes,
    Synthesis,
    WorkedExample,
)
from quorum.llm.router import LLMResponse
from quorum.models import Segment, Speaker, Transcript, Utterance


class FakeRouter:
    """Returns queued objects. Segment calls and the synthesis call are
    distinguished by the schema requested."""

    def __init__(self, segment_results, synthesis=None, tokens: int = 300) -> None:
        self.segment_results = list(segment_results)
        self.synthesis = synthesis
        self.tokens = tokens
        self.prompts: list[str] = []
        self.systems: list[str] = []
        self.thinking: list[bool] = []

    def structured(self, prompt, schema, *, system=None, thinking=False, **kwargs):
        self.prompts.append(prompt)
        self.systems.append(system or "")
        self.thinking.append(thinking)
        response = LLMResponse(
            text="{}", model="fake", provider="fake", total_tokens=self.tokens
        )
        if schema is Synthesis:
            if self.synthesis is None:
                raise RuntimeError("synthesis unavailable")
            return self.synthesis, response
        nxt = self.segment_results.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt, response


@pytest.fixture
def transcript() -> Transcript:
    speaker = Speaker(id="spk_remote", display_name="Remote participant")
    lines = [
        "Today we're looking at how quicksort actually behaves in practice.",
        "The key idea is that you pick a pivot and partition around it.",
        "Now the catch is that on already-sorted input the pivot choice degrades badly.",
        "So a random pivot gives you expected n log n regardless of the input order.",
        "Let's work through an example with the array five three eight one.",
        "That's the core of it. Any questions can wait until the end.",
    ]
    utterances = [
        Utterance(id=f"u{i}", index=i, speaker_id=speaker.id, text=text, start_s=i * 30.0)
        for i, text in enumerate(lines)
    ]
    return Transcript(
        meeting_id="lec_1", title="Quicksort", meeting_date=date(2026, 8, 13),
        speakers=[speaker], utterances=utterances, source="live",
    )


def whole(transcript) -> list[Segment]:
    return [Segment(meeting_id=transcript.meeting_id, start_index=0,
                    end_index=len(transcript.utterances) - 1)]


def notes_with(**kw) -> SegmentNotes:
    return SegmentNotes(**kw)


SYNTHESIS = Synthesis(
    title="Quicksort in practice",
    summary="Covers partitioning and why sorted input degrades naive pivot choice.",
    takeaways=["Random pivots give expected n log n"],
    prerequisites=["Big-O notation", "Basic recursion"],
)


# --- extraction -------------------------------------------------------------


def test_collects_points_concepts_and_examples(transcript):
    router = FakeRouter([
        notes_with(
            topic="Partitioning",
            key_points=[KeyPoint(point="Pick a pivot and partition around it",
                                 timestamp="00:30")],
            concepts=[Concept(term="pivot",
                              plain_explanation="the value you split the list around",
                              why_it_matters="it decides how even the split is")],
            examples=[WorkedExample(description="sorting five three eight one")],
            open_questions=["how does it compare to mergesort"],
        )
    ], SYNTHESIS)

    notes = LectureAnalyser(router=router).analyse(transcript, whole(transcript))

    assert notes.key_points[0].timestamp == "00:30"
    assert notes.concepts[0].term == "pivot"
    assert notes.examples and notes.open_questions
    assert notes.topics == ["Partitioning"]


def test_synthesis_runs_over_extracted_points_not_the_transcript(transcript):
    """A summary written from distilled points is cheaper and more coherent
    than one written from an hour of speech."""
    router = FakeRouter([
        notes_with(key_points=[KeyPoint(point="Random pivots avoid the worst case")])
    ], SYNTHESIS)

    LectureAnalyser(router=router).analyse(transcript, whole(transcript))

    synthesis_prompt = router.prompts[-1]
    assert "Random pivots avoid the worst case" in synthesis_prompt
    assert "Today we're looking at" not in synthesis_prompt, "raw transcript must not be resent"


def test_synthesis_uses_reasoning(transcript):
    """One of only two places reasoning is worth its token cost."""
    router = FakeRouter([notes_with(key_points=[KeyPoint(point="a point")])], SYNTHESIS)
    LectureAnalyser(router=router).analyse(transcript, whole(transcript))

    assert router.thinking[0] is False, "per-segment extraction should not reason"
    assert router.thinking[-1] is True, "synthesis should"


def test_notes_survive_a_failed_synthesis(transcript):
    """Losing the summary should not lose the notes."""
    router = FakeRouter([notes_with(key_points=[KeyPoint(point="a real point")])], None)
    notes = LectureAnalyser(router=router).analyse(transcript, whole(transcript))

    assert notes.key_points and notes.summary == ""


def test_one_bad_segment_does_not_lose_the_lecture(transcript):
    segments = [
        Segment(meeting_id="lec_1", start_index=0, end_index=2),
        Segment(meeting_id="lec_1", start_index=3, end_index=5),
    ]
    router = FakeRouter(
        [RuntimeError("boom"), notes_with(key_points=[KeyPoint(point="survived")])], SYNTHESIS
    )
    notes = LectureAnalyser(router=router).analyse(transcript, segments)

    assert notes.failed_segments == 1
    assert [p.point for p in notes.key_points] == ["survived"]


def test_empty_lecture_skips_synthesis(transcript):
    """Admin and introductions contain no teaching; there is nothing to summarise."""
    router = FakeRouter([notes_with()], SYNTHESIS)
    notes = LectureAnalyser(router=router).analyse(transcript, whole(transcript))

    assert notes.summary == ""
    assert router.prompts and len(router.prompts) == 1, "no synthesis call should be made"


# --- deduplication ----------------------------------------------------------


def test_repeated_points_across_segments_are_merged(transcript):
    """Lecturers repeat themselves and adjacent segments overlap. Without this
    the notes restate one idea several times and read like a transcript."""
    segments = [
        Segment(meeting_id="lec_1", start_index=0, end_index=2),
        Segment(meeting_id="lec_1", start_index=3, end_index=5),
    ]
    router = FakeRouter([
        notes_with(key_points=[KeyPoint(point="A random pivot gives expected n log n")]),
        notes_with(key_points=[KeyPoint(point="Random pivot gives you expected n log n")]),
    ], SYNTHESIS)

    notes = LectureAnalyser(router=router).analyse(transcript, segments)
    assert len(notes.key_points) == 1


def test_distinct_points_are_both_kept(transcript):
    segments = [
        Segment(meeting_id="lec_1", start_index=0, end_index=2),
        Segment(meeting_id="lec_1", start_index=3, end_index=5),
    ]
    router = FakeRouter([
        notes_with(key_points=[KeyPoint(point="Partitioning splits around a pivot")]),
        notes_with(key_points=[KeyPoint(point="Sorted input degrades naive pivot choice")]),
    ], SYNTHESIS)

    notes = LectureAnalyser(router=router).analyse(transcript, segments)
    assert len(notes.key_points) == 2


def test_duplicate_concepts_are_merged_by_term(transcript):
    segments = [
        Segment(meeting_id="lec_1", start_index=0, end_index=2),
        Segment(meeting_id="lec_1", start_index=3, end_index=5),
    ]
    router = FakeRouter([
        notes_with(concepts=[Concept(term="pivot", plain_explanation="the split value")]),
        notes_with(concepts=[Concept(term="pivot", plain_explanation="the chosen element")]),
    ], SYNTHESIS)

    notes = LectureAnalyser(router=router).analyse(transcript, segments)
    assert len(notes.concepts) == 1


# --- prompting --------------------------------------------------------------


def test_prompt_demands_explanation_not_restatement(transcript):
    router = FakeRouter([notes_with()], SYNTHESIS)
    LectureAnalyser(router=router).analyse(transcript, whole(transcript))

    system = router.systems[0].lower()
    assert "explain" in system and "restate" in system


def test_prompt_marks_the_transcript_untrusted(transcript):
    router = FakeRouter([notes_with()], SYNTHESIS)
    LectureAnalyser(router=router).analyse(transcript, whole(transcript))
    assert "untrusted" in router.prompts[0].lower()


# --- rendering --------------------------------------------------------------


def test_markdown_contains_every_section():
    notes = LectureNotes(
        title="Quicksort", summary="A summary.", takeaways=["Use random pivots"],
        prerequisites=["Big-O"],
        key_points=[KeyPoint(point="Partition around a pivot", timestamp="00:30")],
        concepts=[Concept(term="pivot", plain_explanation="the split value",
                          why_it_matters="decides the balance")],
        examples=[WorkedExample(description="five three eight one",
                                what_it_demonstrates="partitioning")],
        open_questions=["how does mergesort compare"],
    )
    markdown = notes.as_markdown()

    for expected in [
        "# Quicksort", "## Summary", "## Takeaways", "## Assumed knowledge",
        "## Concepts, in plain English", "**pivot**", "`00:30`",
        "## Worked examples", "## Left unanswered",
    ]:
        assert expected in markdown


def test_concepts_section_is_labelled_as_possibly_adding_background():
    """The summary is held to what was said; concept explanations deliberately
    are not, since explaining an undefined term is the point. A student
    revising from these needs to know which is which."""
    notes = LectureNotes(
        concepts=[Concept(term="stack", plain_explanation="last in, first out")]
    )
    assert "may add background" in notes.as_markdown()


def test_markdown_omits_empty_sections():
    markdown = LectureNotes(title="T", summary="S").as_markdown()
    assert "## Takeaways" not in markdown
    assert "## Worked examples" not in markdown


def test_timestamps_make_notes_navigable():
    """A key point you cannot find in the video is close to useless."""
    notes = LectureNotes(key_points=[KeyPoint(point="the important bit", timestamp="12:34")])
    assert "`12:34`" in notes.as_markdown()
