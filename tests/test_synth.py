from __future__ import annotations

import json
from collections import Counter

import pytest

from quorum.synth import CommitmentFate, ProjectGenerator


@pytest.fixture(scope="module")
def project():
    return ProjectGenerator(seed=1, weeks=6).generate()


@pytest.fixture(scope="module")
def many():
    """A spread of seeds, so distribution claims are not one lucky draw."""
    return [ProjectGenerator(seed=s, weeks=8).generate() for s in range(6)]


# --- the core invariant ----------------------------------------------------


def test_every_manifest_entry_matches_the_rendered_transcript(many):
    """The manifest is not an annotation of generated text - the text is a
    rendering of the manifest. If these ever diverge, every metric built on the
    benchmark is silently wrong."""
    for transcripts, manifest in many:
        for commitment in manifest.commitments:
            transcript = transcripts[commitment.meeting_index]
            assert 0 <= commitment.utterance_index < len(transcript.utterances)
            actual = transcript.utterances[commitment.utterance_index].text
            assert actual == commitment.spoken_quote

        for musing in manifest.musings:
            actual = transcripts[musing.meeting_index].utterances[musing.utterance_index].text
            assert actual == musing.spoken_quote

        for decision in manifest.decisions:
            actual = transcripts[decision.meeting_index].utterances[decision.utterance_index].text
            assert actual == decision.spoken_quote


def test_first_person_commitments_are_spoken_by_their_owner(many):
    for transcripts, manifest in many:
        for commitment in manifest.commitments:
            if commitment.assignee_mention != "I":
                continue
            transcript = transcripts[commitment.meeting_index]
            utterance = transcript.utterances[commitment.utterance_index]
            speaker = transcript.speaker(utterance.speaker_id)
            assert speaker.display_name == commitment.owner_name


def test_owner_is_always_a_meeting_participant(many):
    for transcripts, manifest in many:
        names = {p.name for p in manifest.people}
        assert all(c.owner_name in names for c in manifest.commitments)


# --- determinism ------------------------------------------------------------


def test_same_seed_reproduces_everything():
    a_transcripts, a_manifest = ProjectGenerator(seed=42, weeks=5).generate()
    b_transcripts, b_manifest = ProjectGenerator(seed=42, weeks=5).generate()

    assert [t.as_dialogue() for t in a_transcripts] == [t.as_dialogue() for t in b_transcripts]
    assert a_manifest.to_json() == b_manifest.to_json()


def test_different_seeds_produce_different_projects():
    a, _ = ProjectGenerator(seed=1, weeks=5).generate()
    b, _ = ProjectGenerator(seed=2, weeks=5).generate()
    assert a[0].as_dialogue() != b[0].as_dialogue()


# --- fate coverage ----------------------------------------------------------


def test_all_fates_appear_across_seeds(many):
    """Every fate must be exercised or the benchmark cannot measure that case."""
    seen = Counter()
    for _, manifest in many:
        seen.update(c.fate for c in manifest.commitments)
    for fate in CommitmentFate:
        assert seen[fate] > 0, f"{fate.value} never generated"


def test_silent_deliveries_are_never_mentioned_again(many):
    """The whole point of this fate: conversation alone cannot resolve it, so an
    agent must consult external evidence or it will wrongly nag."""
    for transcripts, manifest in many:
        for commitment in manifest.commitments:
            if commitment.fate is not CommitmentFate.DELIVERED_SILENTLY:
                continue
            later = " ".join(
                t.as_dialogue() for t in transcripts[commitment.meeting_index + 1 :]
            ).lower()
            assert commitment.description.lower() not in later
            assert commitment.github_evidence, "external proof must exist instead"


def test_dropped_commitments_leave_no_trace_and_no_evidence(many):
    for transcripts, manifest in many:
        for commitment in manifest.commitments:
            if commitment.fate is not CommitmentFate.DROPPED:
                continue
            later = " ".join(
                t.as_dialogue() for t in transcripts[commitment.meeting_index + 1 :]
            ).lower()
            assert commitment.description.lower() not in later
            assert commitment.github_evidence is None
            assert commitment.delivered_on is None


def test_slipped_commitments_keep_their_identity(many):
    """A slip moves the date on the same obligation. An agent that creates a
    second commitment instead has failed to track it."""
    for _, manifest in many:
        slipped = [c for c in manifest.commitments if c.fate is CommitmentFate.SLIPPED]
        ids = [c.id for c in slipped]
        assert len(ids) == len(set(ids))


def test_blocked_commitments_name_an_upstream_dependency(many):
    for _, manifest in many:
        for commitment in manifest.commitments:
            if commitment.fate is CommitmentFate.BLOCKED and commitment.depends_on:
                assert manifest.commitment(commitment.depends_on) is not None


def test_delivered_commitments_carry_external_evidence(many):
    for _, manifest in many:
        for commitment in manifest.commitments:
            if commitment.fate in (
                CommitmentFate.DELIVERED, CommitmentFate.DELIVERED_SILENTLY
            ):
                assert commitment.github_evidence is not None


# --- the scoring surface ----------------------------------------------------


def test_chase_set_and_nag_set_are_disjoint(many):
    for _, manifest in many:
        for commitment in manifest.commitments:
            assert not (commitment.should_be_chased and commitment.is_false_nag_target)


def test_tentative_items_are_never_in_the_chase_set(many):
    """"I might get to it if there's time" is not an obligation."""
    for _, manifest in many:
        for commitment in manifest.commitments:
            if commitment.strength == "tentative":
                assert not commitment.should_be_chased


def test_musings_are_recorded_as_negatives(project):
    _, manifest = project
    assert manifest.musings, "precision needs negatives to score against"
    quotes = {m.spoken_quote for m in manifest.musings}
    firm_quotes = {c.spoken_quote for c in manifest.firm()}
    assert not (quotes & firm_quotes), "a musing must never also be a firm commitment"


def test_some_decisions_get_reversed(many):
    """Contradiction detection needs at least some real contradictions."""
    reversals = sum(
        1 for _, m in many for d in m.decisions if d.reversed_by is not None
    )
    assert reversals > 0


def test_reversal_ids_point_at_real_decisions(many):
    for _, manifest in many:
        by_id = {d.id: d for d in manifest.decisions}
        for decision in manifest.decisions:
            if decision.reversed_by:
                assert decision.reversed_by in by_id
                assert by_id[decision.reversed_by].meeting_index > decision.meeting_index


# --- shape and serialisation ------------------------------------------------


def test_meetings_are_weekly_and_ordered(project):
    transcripts, _ = project
    dates = [t.meeting_date for t in transcripts]
    assert dates == sorted(dates)
    assert all((b - a).days == 7 for a, b in zip(dates, dates[1:]))


def test_transcripts_share_a_project_id(project):
    transcripts, manifest = project
    assert all(t.project_id == manifest.project_id for t in transcripts)
    assert all(t.source == "synthetic" for t in transcripts)


def test_deadlines_are_after_the_meeting_they_were_made_in(project):
    transcripts, manifest = project
    for commitment in manifest.commitments:
        if commitment.deadline_date:
            assert commitment.deadline_date >= transcripts[commitment.meeting_index].meeting_date


def test_manifest_serialises_to_json(project):
    _, manifest = project
    payload = json.loads(manifest.to_json())
    assert payload["seed"] == 1
    assert len(payload["commitments"]) == len(manifest.commitments)
    assert all(isinstance(c["fate"], str) for c in payload["commitments"])
