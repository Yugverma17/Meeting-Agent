from __future__ import annotations

from datetime import date

import pytest

from quorum.agents.embedding import LexicalEmbedder
from quorum.memory import MemoryKind, ProjectMemory
from quorum.models import (
    Assignee,
    Commitment,
    Deadline,
    Decision,
    Evidence,
    MeetingRecord,
    Risk,
)


@pytest.fixture
def memory(tmp_path) -> ProjectMemory:
    # Lexical embeddings keep this offline and instant; the semantic quality of
    # BGE is not what these tests are checking.
    return ProjectMemory(tmp_path / "memory", embedder=LexicalEmbedder())


def commitment(description: str, owner: str = "Yug Verma") -> Commitment:
    return Commitment(
        description=description,
        assignee=Assignee(speaker_id="spk", display_name=owner, confidence=0.9),
        deadline=Deadline(),
        evidence=[Evidence(utterance_id="u1", quote="a sufficiently long verbatim quote")],
    )


def decision(statement: str) -> Decision:
    return Decision(
        statement=statement,
        evidence=[Evidence(utterance_id="u1", quote="a sufficiently long verbatim quote")],
    )


def meeting(meeting_id: str, when: date, **kw) -> MeetingRecord:
    return MeetingRecord(meeting_id=meeting_id, meeting_date=when, **kw)


# --- indexing ---------------------------------------------------------------


def test_indexes_commitments_decisions_and_risks(memory):
    written = memory.index_meeting(meeting(
        "m1", date(2026, 3, 9),
        commitments=[commitment("the ingestion API spec")],
        decisions=[decision("Use Postgres rather than Mongo")],
        risks=[Risk(description="staging runs out of memory",
                    evidence=[Evidence(utterance_id="u", quote="a long enough quote here")])],
    ))
    assert written == 3
    assert memory.count() == 3


def test_empty_meeting_indexes_nothing(memory):
    assert memory.index_meeting(meeting("m1", date(2026, 3, 9))) == 0


def test_reindexing_a_meeting_replaces_rather_than_duplicates(memory):
    """A re-recorded meeting must not appear twice in every recall - duplicates
    crowd genuine results out of the top-k."""
    record = meeting("m1", date(2026, 3, 9), commitments=[commitment("the ingestion API spec")])
    memory.index_meeting(record)
    memory.index_meeting(record)

    assert memory.count() == 1


def test_indexing_a_second_meeting_adds_to_the_first(memory):
    memory.index_meeting(meeting("m1", date(2026, 3, 9),
                                 commitments=[commitment("the ingestion API spec")]))
    memory.index_meeting(meeting("m2", date(2026, 3, 16),
                                 commitments=[commitment("the billing reconciliation")]))
    assert memory.count() == 2


# --- retrieval --------------------------------------------------------------


def test_recall_ranks_the_relevant_item_first(memory):
    memory.index_meeting(meeting(
        "m1", date(2026, 3, 9),
        commitments=[
            commitment("the ingestion API spec document"),
            commitment("the billing reconciliation"),
            commitment("the kubernetes cluster autoscaling"),
        ],
    ))
    hits = memory.recall("ingestion API spec", k=3)
    assert hits and "ingestion" in hits[0].text.lower()


def test_recall_can_filter_by_kind(memory):
    memory.index_meeting(meeting(
        "m1", date(2026, 3, 9),
        commitments=[commitment("the database migration")],
        decisions=[decision("The database direction is Postgres")],
    ))
    hits = memory.recall("database", k=5, kind=MemoryKind.DECISION)
    assert hits and all(hit.kind is MemoryKind.DECISION for hit in hits)


def test_recall_on_an_empty_store(memory):
    assert memory.recall("anything") == []


def test_recall_of_blank_query(memory):
    memory.index_meeting(meeting("m1", date(2026, 3, 9),
                                 commitments=[commitment("the spec")]))
    assert memory.recall("   ") == []


def test_hits_carry_the_source_id_for_resolution(memory):
    """A hit must resolve back to a live ledger object, not just a quote."""
    item = commitment("the ingestion API spec")
    memory.index_meeting(meeting("m1", date(2026, 3, 9), commitments=[item]))

    hit = memory.recall("ingestion spec", k=1)[0]
    assert hit.ref_id == item.id
    assert hit.meeting_id == "m1"
    assert hit.meeting_date == "2026-03-09"


# --- the three uses ---------------------------------------------------------


def test_status_update_matches_the_right_commitment(memory):
    spec = commitment("the ingestion API spec document")
    memory.index_meeting(meeting(
        "m1", date(2026, 3, 9),
        commitments=[spec, commitment("the kubernetes cluster autoscaling")],
    ))
    assert memory.match_commitment("the ingestion API spec document") == spec.id


def test_match_abstains_when_nothing_is_close(memory):
    """Applying "that's done" to the wrong commitment closes work that is still
    outstanding, and nothing later reopens it."""
    memory.index_meeting(meeting("m1", date(2026, 3, 9),
                                 commitments=[commitment("the ingestion API spec")]))
    assert memory.match_commitment("the quarterly marketing budget review") is None


def test_contradiction_candidates_return_only_decisions(memory):
    memory.index_meeting(meeting(
        "m1", date(2026, 3, 9),
        commitments=[commitment("the database migration work")],
        decisions=[decision("The database direction is Postgres rather than Mongo")],
    ))
    candidates = memory.contradiction_candidates("The database direction is Mongo")
    assert candidates and all(c.kind is MemoryKind.DECISION for c in candidates)


def test_brief_summarises_project_context(memory):
    memory.index_meeting(meeting(
        "m1", date(2026, 3, 9),
        commitments=[commitment("the ingestion API spec")],
        decisions=[decision("Use Postgres rather than Mongo")],
    ))
    brief = memory.brief()

    assert "Postgres" in brief
    assert "background only" in brief, "must warn the model not to extract from it"


def test_brief_of_an_empty_store_is_empty(memory):
    assert memory.brief() == ""


def test_brief_respects_the_limit(memory):
    memory.index_meeting(meeting(
        "m1", date(2026, 3, 9),
        commitments=[commitment(f"work item number {i}") for i in range(20)],
    ))
    assert len(memory.brief(limit=3).strip().splitlines()) <= 4  # header + 3


# --- degradation ------------------------------------------------------------


def test_store_falls_back_when_the_vector_db_is_unavailable(tmp_path, monkeypatch):
    """A missing model or a locked file should cost retrieval quality, not the
    meeting."""
    memory = ProjectMemory(tmp_path / "memory", embedder=LexicalEmbedder())
    memory._fallback = []  # simulate the degraded path

    memory.index_meeting(meeting("m1", date(2026, 3, 9),
                                 commitments=[commitment("the ingestion API spec")]))

    assert memory.count() == 1
    assert memory.recall("ingestion spec", k=1)


def test_fallback_reindex_also_replaces(tmp_path):
    memory = ProjectMemory(tmp_path / "memory", embedder=LexicalEmbedder())
    memory._fallback = []
    record = meeting("m1", date(2026, 3, 9), commitments=[commitment("the spec")])
    memory.index_meeting(record)
    memory.index_meeting(record)
    assert memory.count() == 1
