from __future__ import annotations

import pytest
from pydantic import ValidationError

from quorum.models import (
    Assignee,
    Commitment,
    CommitmentStrength,
    Evidence,
    Speaker,
)


def test_evidence_rejects_empty_quote():
    """An uncited item must not be representable at all."""
    with pytest.raises(ValidationError):
        Evidence(utterance_id="utt_1", quote="   ")


def test_grounded_requires_at_least_one_evidence():
    with pytest.raises(ValidationError):
        Commitment(description="do the thing", evidence=[])


def test_speaker_matches_aliases_and_email_local_part():
    spk = Speaker(display_name="Yug Verma", aliases=["Yug"], email="yugverma@example.com")
    assert spk.matches("Yug")
    assert spk.matches("  yug  ")
    assert spk.matches("Yug Verma")
    assert spk.matches("yugverma")
    assert not spk.matches("Priya")
    assert not spk.matches("")


def test_commitment_is_actionable_requires_firm_and_resolved_owner():
    evidence = [Evidence(utterance_id="utt_1", quote="I'll send it Friday")]

    firm_resolved = Commitment(
        description="Send the spec",
        evidence=evidence,
        strength=CommitmentStrength.FIRM,
        assignee=Assignee(raw_mention="I", speaker_id="spk_yug", confidence=0.9),
    )
    assert firm_resolved.is_actionable

    firm_unowned = Commitment(
        description="Send the spec",
        evidence=evidence,
        strength=CommitmentStrength.FIRM,
        assignee=Assignee(raw_mention="someone"),
    )
    assert not firm_unowned.is_actionable, "no resolved owner means we cannot chase it"

    musing = Commitment(
        description="Look into rate limiting",
        evidence=evidence,
        strength=CommitmentStrength.MUSING,
        assignee=Assignee(speaker_id="spk_yug"),
    )
    assert not musing.is_actionable, "musings must never become tasks"


def test_is_grounded_reflects_verifier_output():
    unverified = Commitment(
        description="x", evidence=[Evidence(utterance_id="u1", quote="q")]
    )
    assert not unverified.is_grounded

    verified = Commitment(
        description="x",
        evidence=[Evidence(utterance_id="u1", quote="q", verified=True, match_score=1.0)],
    )
    assert verified.is_grounded


def test_utterance_timestamp_formatting(transcript):
    assert transcript.utterances[0].timestamp == "00:00"
    assert transcript.utterances[4].timestamp == "01:00"


def test_as_dialogue_includes_speaker_names_and_indices(transcript):
    dialogue = transcript.as_dialogue(0, 2)
    assert "[0] Priya Raghavan (00:00): Okay, where are we on the ingestion API?" in dialogue
    assert "Yug Verma" in dialogue
    assert "Sam Okafor" not in dialogue, "slicing must be respected"


def test_resolve_mention_uses_aliases(transcript):
    assert transcript.resolve_mention("Yug").display_name == "Yug Verma"
    assert transcript.resolve_mention("nobody") is None
