from __future__ import annotations

from datetime import date

import pytest

from quorum.models import (
    Assignee,
    Commitment,
    CommitmentStatus,
    CommitmentStrength,
    Deadline,
    Decision,
    Evidence,
    MeetingRecord,
    StatusKind,
    StatusUpdate,
)
from quorum.workspace import Workspace, slugify


@pytest.fixture
def workspace(tmp_path) -> Workspace:
    return Workspace(tmp_path / "workspace")


def commitment(description: str, due: date | None = date(2026, 3, 13)) -> Commitment:
    return Commitment(
        description=description,
        assignee=Assignee(speaker_id="spk_yug", display_name="Yug Verma",
                          email="yug@example.com", confidence=0.9),
        deadline=Deadline(resolved=due),
        strength=CommitmentStrength.FIRM,
        evidence=[Evidence(utterance_id="u1", quote="a sufficiently long verbatim quote")],
    )


def record(meeting_id: str, when: date, commitments=None, updates=None) -> MeetingRecord:
    return MeetingRecord(
        meeting_id=meeting_id, meeting_date=when,
        commitments=commitments or [], status_updates=updates or [],
    )


# --- registry ---------------------------------------------------------------


def test_create_and_fetch(workspace):
    workspace.create("Ingestion Revamp", description="rebuild the pipeline")
    found = workspace.get("ingestion-revamp")

    assert found is not None
    assert found.meta.name == "Ingestion Revamp"
    assert found.meta.description == "rebuild the pipeline"


def test_lookup_is_name_or_slug(workspace):
    workspace.create("Ingestion Revamp")
    assert workspace.get("Ingestion Revamp") is not None
    assert workspace.get("ingestion-revamp") is not None
    assert workspace.get("INGESTION revamp") is not None


def test_duplicate_names_are_rejected(workspace):
    workspace.create("Ingestion Revamp")
    with pytest.raises(ValueError, match="already exists"):
        workspace.create("ingestion revamp")


def test_unknown_project_is_none(workspace):
    assert workspace.get("nope") is None


def test_delete_keeps_the_files(workspace):
    """A mistyped name must not destroy weeks of meeting history."""
    project = workspace.create("Ingestion Revamp")
    project.add_meeting(record("m1", date(2026, 3, 9), [commitment("the spec")]))
    root = project.root

    assert workspace.delete("ingestion-revamp") is True
    assert workspace.get("ingestion-revamp") is None
    assert (root / "ledger.json").exists(), "history should survive de-registration"


def test_members_seed_the_roster(workspace):
    project = workspace.create(
        "P", members={"Priya Raghavan": "priya@x.com", "Sam Okafor": "sam@x.com"}
    )
    roster = project.roster_string()
    assert "Priya Raghavan:priya@x.com" in roster
    assert "Sam Okafor:sam@x.com" in roster


@pytest.mark.parametrize(
    "name,expected",
    [("Ingestion Revamp", "ingestion-revamp"), ("  A/B  test!", "a-b-test"), ("...", "project")],
)
def test_slugify(name, expected):
    assert slugify(name) == expected


# --- persistence ------------------------------------------------------------


def test_ledger_survives_a_restart(workspace):
    """The whole point: week 7 has to know what week 2 promised."""
    project = workspace.create("P")
    project.add_meeting(record("m1", date(2026, 3, 9), [commitment("the ingestion spec")]))
    workspace.save(project)

    reopened = Workspace(workspace.root).get("P")
    assert len(reopened.ledger.commitments) == 1
    assert reopened.ledger.commitments[0].description == "the ingestion spec"


def test_meeting_count_and_last_date_are_tracked(workspace):
    project = workspace.create("P")
    project.add_meeting(record("m1", date(2026, 3, 9)))
    project.add_meeting(record("m2", date(2026, 3, 16)))
    workspace.save(project)

    meta = Workspace(workspace.root).get("P").meta
    assert meta.meeting_count == 2
    assert meta.last_meeting_on == "2026-03-16"


def test_meetings_are_returned_in_date_order(workspace):
    project = workspace.create("P")
    project.add_meeting(record("m2", date(2026, 3, 16)))
    project.add_meeting(record("m1", date(2026, 3, 9)))

    assert [m.meeting_id for m in project.meetings()] == ["m1", "m2"]


def test_a_status_update_closes_an_earlier_commitment(workspace):
    """The cross-meeting behaviour, through the persistence layer."""
    project = workspace.create("P")
    project.add_meeting(record("m1", date(2026, 3, 9), [commitment("the ingestion spec")]))

    update = StatusUpdate(
        about="the ingestion spec", kind=StatusKind.DELIVERED,
        evidence=[Evidence(utterance_id="u9", quote="I sent the ingestion spec on Tuesday")],
    )
    project.add_meeting(record("m2", date(2026, 3, 16), updates=[update]))

    tracked = project.ledger.commitments[0]
    assert tracked.status is CommitmentStatus.CLAIMED_DONE
    # Still open on purpose: someone saying it is done is a report, not proof.
    # Only external evidence may set VERIFIED_DONE. The planner stops nagging it
    # either way, so the user sees no difference until verification lands.
    assert len(project.ledger.open_commitments()) == 1
    assert tracked.resolution_note and "claimed" in tracked.resolution_note


def test_empty_project_has_an_empty_ledger(workspace):
    project = workspace.create("P")
    assert project.ledger.commitments == []
    assert project.meetings() == []


def test_corrupt_registry_does_not_crash(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "projects.json").write_text("{ not json", encoding="utf-8")
    assert Workspace(root).list() == []


def test_decisions_persist_for_contradiction_checks(workspace):
    project = workspace.create("P")
    decision = Decision(
        statement="Use Postgres",
        evidence=[Evidence(utterance_id="u1", quote="Final call: we go with Postgres")],
    )
    project.add_meeting(
        MeetingRecord(meeting_id="m1", meeting_date=date(2026, 3, 9), decisions=[decision])
    )
    reopened = Workspace(workspace.root).get("P")
    assert len(reopened.ledger.decisions) == 1
