from __future__ import annotations

from datetime import date

import pytest

from quorum.agents.embedding import LexicalEmbedder
from quorum.agents.segmenter import Segmenter, SegmenterConfig
from quorum.llm.ratelimit import estimate_tokens
from quorum.models import Speaker, Transcript, Utterance

TOPICS = {
    "billing": [
        "The invoice generator is still producing duplicate billing rows.",
        "Right, the billing reconciliation job double counts refunds.",
        "We should reconcile invoices against the payment ledger nightly.",
        "Agreed, nightly invoice reconciliation would catch the billing drift.",
        "I'll patch the billing invoice duplication this week.",
        "Good, duplicate invoices are the worst billing complaint we get.",
    ],
    "hiring": [
        "Moving on to hiring, we have four candidate interviews scheduled.",
        "The frontend candidate interview went well, strong portfolio.",
        "Let's schedule the second round interview with that candidate.",
        "Hiring for the backend role is slower, fewer candidates applying.",
        "I'll refresh the job posting to attract more backend candidates.",
        "Interview panels need two engineers per candidate minimum.",
    ],
    "infra": [
        "Last item, the Kubernetes cluster keeps evicting pods overnight.",
        "Pod eviction correlates with the nightly cluster autoscaling window.",
        "Cluster memory limits are too tight for the ingestion pods.",
        "I'll raise the pod memory limits on the ingestion cluster tomorrow.",
        "Autoscaling should also be capped so the cluster stops thrashing.",
        "Let's monitor cluster pod evictions for a week after that change.",
    ],
}


def build_transcript(blocks: list[list[str]]) -> Transcript:
    speakers = [Speaker(id=f"spk_{i}", display_name=f"Speaker {i}") for i in range(3)]
    utterances, index = [], 0
    for block in blocks:
        for line in block:
            utterances.append(
                Utterance(
                    id=f"utt_{index}", index=index,
                    speaker_id=speakers[index % 3].id, text=line,
                    start_s=index * 10.0, end_s=index * 10.0 + 8.0,
                )
            )
            index += 1
    return Transcript(
        meeting_id="mtg_seg", meeting_date=date(2026, 3, 9),
        speakers=speakers, utterances=utterances, source="fixture",
    )


@pytest.fixture
def segmenter() -> Segmenter:
    # Lexical embeddings keep this test deterministic and offline.
    return Segmenter(embedder=LexicalEmbedder())


# --- invariants ------------------------------------------------------------


def test_empty_transcript_yields_no_segments(segmenter):
    empty = Transcript(meeting_id="m", meeting_date=date(2026, 3, 9))
    assert segmenter.segment(empty) == []


def test_short_transcript_is_a_single_segment(segmenter):
    transcript = build_transcript([TOPICS["billing"][:3]])
    segments = segmenter.segment(transcript)
    assert len(segments) == 1
    assert (segments[0].start_index, segments[0].end_index) == (0, 2)


def test_segments_exactly_partition_the_transcript(segmenter):
    """The invariant that matters: every utterance lands in exactly one segment,
    with no gaps and no overlaps. A dropped utterance is a silently lost
    commitment."""
    transcript = build_transcript(list(TOPICS.values()))
    segments = segmenter.segment(transcript)

    covered = []
    for seg in segments:
        covered.extend(range(seg.start_index, seg.end_index + 1))

    assert covered == sorted(covered), "segments must be in order"
    assert covered == list(range(len(transcript.utterances))), "exact cover required"


def test_no_segment_exceeds_the_token_ceiling(segmenter):
    """A segment over budget cannot be sent at all under a 6k TPM limit."""
    long_block = ["This is a fairly wordy sentence about quarterly planning. " * 12] * 20
    transcript = build_transcript([long_block])
    config = SegmenterConfig(max_tokens=400)
    segments = Segmenter(config=config, embedder=LexicalEmbedder()).segment(transcript)

    assert len(segments) > 1, "an oversized transcript must be split"
    for seg in segments:
        text = " ".join(
            u.text for u in transcript.utterances[seg.start_index : seg.end_index + 1]
        )
        assert estimate_tokens(text) <= config.max_tokens


def test_single_oversized_utterance_is_emitted_rather_than_dropped(segmenter):
    """Nothing splits a single utterance, so it must pass through, not vanish."""
    transcript = build_transcript([["word " * 5000]])
    segments = Segmenter(
        config=SegmenterConfig(max_tokens=100), embedder=LexicalEmbedder()
    ).segment(transcript)
    assert len(segments) == 1
    assert segments[0].start_index == segments[0].end_index == 0


# --- behaviour -------------------------------------------------------------


def test_detects_boundaries_between_distinct_topics(segmenter):
    transcript = build_transcript(list(TOPICS.values()))
    segments = segmenter.segment(transcript)

    assert len(segments) >= 2, "three unrelated topics should not be one segment"

    # Each 6-line topic block starts at index 0, 6, 12. At least one real
    # boundary should land on one of those, within a tolerance of one turn.
    starts = {seg.start_index for seg in segments}
    assert any(abs(start - expected) <= 1 for start in starts for expected in (6, 12))


def test_uniform_text_produces_no_topic_boundaries(segmenter):
    """With nothing to distinguish the turns, there is no honest place to cut."""
    transcript = build_transcript([["Exactly the same sentence every time."] * 14])
    segments = segmenter.segment(transcript)
    assert len(segments) == 1


def test_min_utterances_spacing_is_respected():
    transcript = build_transcript(list(TOPICS.values()))
    config = SegmenterConfig(min_utterances=5, max_tokens=10_000)
    segments = Segmenter(config=config, embedder=LexicalEmbedder()).segment(transcript)

    starts = sorted(seg.start_index for seg in segments)
    gaps = [b - a for a, b in zip(starts, starts[1:])]
    assert all(gap >= config.min_utterances for gap in gaps)


def test_segment_carries_meeting_id(segmenter):
    transcript = build_transcript(list(TOPICS.values()))
    for seg in segmenter.segment(transcript):
        assert seg.meeting_id == transcript.meeting_id
        assert seg.n_utterances >= 1


# --- embedding fallback ----------------------------------------------------


def test_lexical_embedder_is_normalised_and_discriminative():
    embedder = LexicalEmbedder()
    vectors = embedder.embed(
        ["billing invoice reconciliation", "kubernetes pod eviction cluster"]
    )
    norms = (vectors**2).sum(axis=1) ** 0.5
    assert all(abs(n - 1.0) < 1e-5 for n in norms)

    similarity = float(vectors[0] @ vectors[1])
    assert similarity < 0.3, "unrelated sentences should not look similar"


def test_lexical_embedder_handles_empty_input():
    assert LexicalEmbedder().embed([]).shape[0] == 0
