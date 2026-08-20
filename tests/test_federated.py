"""Searching every project at once.

Two things have to hold, and the second is the one that would fail quietly.

Reads federate; **writes never do** - "mark the spec as done" across five
projects has no correct answer when two of them match, and picking one silently
closes work that is still outstanding.

And scores from two indexes are only comparable if the same embedder produced
them. `get_embedder()` degrades to a hashing fallback when the ONNX model will
not load, so one project could be indexed semantically and another lexically,
and merging by score would rank almost arbitrarily - silently, and worse the
more projects you have.
"""

from __future__ import annotations

from datetime import date

import pytest

from quorum.chat.federated import FederatedMemory, all_projects, resolve_across
from quorum.chat.tools import ToolContext, ToolRequest, run_tool
from quorum.memory.store import MemoryHit, MemoryKind
from quorum.models import (
    Assignee,
    Commitment,
    Deadline,
    Evidence,
    Speaker,
    Transcript,
    Utterance,
)
from quorum.workspace import Workspace


def transcript_for(title: str, meeting_id: str, when: date) -> Transcript:
    speaker = Speaker(id="spk_l", display_name="Lecturer")
    return Transcript(
        meeting_id=meeting_id, title=title, meeting_date=when, speakers=[speaker],
        utterances=[Utterance(id=f"{meeting_id}_u0", index=0, speaker_id=speaker.id,
                              text="a sliding window expands and contracts",
                              start_s=0.0, end_s=10.0)],
        source="fixture",
    )


@pytest.fixture
def workspace(tmp_path):
    ws = Workspace(tmp_path / "workspace")
    for name, meeting_id, title in [
        ("DSA", "mtg_dsa", "Sliding window"),
        ("Systems", "mtg_sys", "Rate limiting"),
    ]:
        project = ws.create(name)
        project.transcripts_dir.mkdir(parents=True, exist_ok=True)
        (project.transcripts_dir / f"{meeting_id}.json").write_text(
            transcript_for(title, meeting_id, date(2026, 8, 15)).model_dump_json(),
            encoding="utf-8",
        )
        project.memory_dir.mkdir(parents=True, exist_ok=True)
        ws.save(project)
    return ws


class StubStore:
    """Stands in for one project's index."""

    def __init__(self, hits) -> None:
        self.hits = hits
        self.calls: list[dict] = []

    def recall(self, query, k=5, kind=None, min_score=0.0, meeting_ids=None):
        self.calls.append({"query": query, "k": k})
        return [h for h in self.hits if h.score >= min_score][:k]

    def count(self):
        return len(self.hits)


def hit(text, score, meeting_id="m1") -> MemoryHit:
    return MemoryHit(MemoryKind.NOTE, f"ref_{text[:6]}", text, meeting_id, "2026-08-15", score)


def federated_over(workspace, per_project):
    projects = all_projects(workspace)
    memory = FederatedMemory(projects, embedder=object())
    for project in projects:
        memory._stores[project.meta.id] = StubStore(per_project[project.meta.id])
    return projects, memory


# --- merging ------------------------------------------------------------------


def test_results_from_every_project_are_merged_by_score(workspace):
    projects, memory = federated_over(workspace, {
        "dsa": [hit("sliding window counts substrings", 0.9)],
        "systems": [hit("rate limiting uses a sliding window", 0.8)],
    })

    hits = memory.recall("sliding window", k=5)

    assert [h.score for h in hits] == [0.9, 0.8]
    assert {h.project_id for h in hits} == {"dsa", "systems"}


def test_every_hit_says_which_project_it_came_from(workspace):
    """An answer drawn from three projects is only useful if the reader can
    tell which is which."""
    _, memory = federated_over(workspace, {
        "dsa": [hit("a", 0.9)], "systems": [hit("b", 0.8)],
    })

    assert all(h.project_id for h in memory.recall("q", k=5))


def test_one_project_may_supply_every_result(workspace):
    """An even split would force in weaker hits from elsewhere purely for being
    elsewhere."""
    _, memory = federated_over(workspace, {
        "dsa": [hit("a", 0.95), hit("b", 0.94), hit("c", 0.93)],
        "systems": [hit("d", 0.10)],
    })

    hits = memory.recall("q", k=3)

    assert {h.project_id for h in hits} == {"dsa"}


def test_each_store_is_asked_for_the_full_k(workspace):
    projects, memory = federated_over(workspace, {
        "dsa": [hit("a", 0.9)], "systems": [hit("b", 0.8)],
    })

    memory.recall("q", k=6)

    assert all(store.calls[0]["k"] == 6 for store in memory._stores.values())


def test_a_broken_index_does_not_end_the_search(workspace):
    class Broken(StubStore):
        def recall(self, *a, **kw):
            raise RuntimeError("lance is unhappy")

    projects = all_projects(workspace)
    memory = FederatedMemory(projects, embedder=object())
    memory._stores["dsa"] = Broken([])
    memory._stores["systems"] = StubStore([hit("still here", 0.7)])

    hits = memory.recall("q", k=5)

    assert [h.text for h in hits] == ["still here"]


def test_one_embedder_is_shared_by_every_store(workspace):
    """The failure this prevents is silent: two indexes built by different
    models score on different scales, and merging by score ranks arbitrarily."""
    sentinel = object()
    projects = all_projects(workspace)
    memory = FederatedMemory(projects, embedder=sentinel)

    stores = [memory.store_for(p) for p in projects]

    assert len(stores) > 1
    assert all(store.embedder is sentinel for store in stores)


# --- naming across projects ---------------------------------------------------


def test_a_meeting_is_found_in_whichever_project_holds_it(workspace):
    projects = all_projects(workspace)

    project, resolution = resolve_across(projects, "rate limiting")

    assert resolution.ok
    assert project.meta.id == "systems"


def test_the_same_title_in_two_projects_is_ambiguous(workspace):
    """Two teams each with a "Weekly sync" is exactly where guessing produces a
    confident answer about the wrong team."""
    extra = workspace.create("Other")
    extra.transcripts_dir.mkdir(parents=True, exist_ok=True)
    (extra.transcripts_dir / "mtg_x.json").write_text(
        transcript_for("Rate limiting", "mtg_x", date(2026, 8, 16)).model_dump_json(),
        encoding="utf-8",
    )
    workspace.save(extra)

    project, resolution = resolve_across(all_projects(workspace), "rate limiting")

    assert project is None
    assert resolution.ambiguous


def test_nothing_matching_resolves_to_nothing(workspace):
    _, resolution = resolve_across(all_projects(workspace), "quantum chromodynamics")

    assert resolution.how == "none"


# --- writes stay put ----------------------------------------------------------


def commitment_in(project):
    project.ledger.commitments.append(Commitment(
        id="c1", description="send the ingestion spec", meeting_id="mtg_dsa",
        assignee=Assignee(display_name="Priya", email="p@example.com", confidence=0.9),
        deadline=Deadline(resolved=None),
        evidence=[Evidence(utterance_id="u1", quote="I'll send the ingestion spec")],
    ))


@pytest.mark.parametrize("tool,args", [
    ("close_commitment", {"what": "ingestion spec"}),
    ("set_deadline", {"what": "ingestion spec", "when": "next Friday"}),
    ("sync_calendar", {}),
    ("draft_email", {"who": "Priya", "what": "the spec"}),
])
def test_no_write_tool_runs_while_searching_everywhere(workspace, tool, args):
    project = workspace.get("dsa")
    commitment_in(project)
    ctx = ToolContext(project=project, workspace=workspace, federated=True)

    result = run_tool(ctx, ToolRequest(tool=tool, **args))

    assert not result.ok
    assert "--project" in result.text
    assert project.ledger.commitments[0].deadline.resolved is None


def test_the_same_write_runs_when_scoped_to_one_project(workspace):
    project = workspace.get("dsa")
    commitment_in(project)
    ctx = ToolContext(project=project, workspace=workspace, federated=False)

    result = run_tool(ctx, ToolRequest(tool="set_deadline", what="ingestion spec",
                                       when="next Friday"))

    assert result.needs_confirmation


def test_reads_still_work_while_federated(workspace):
    project = workspace.get("dsa")
    ctx = ToolContext(project=project, workspace=workspace, federated=True,
                      memory=StubStore([hit("a sliding window", 0.9)]))

    assert run_tool(ctx, ToolRequest(tool="search", query="window")).ok


# --- discovery ----------------------------------------------------------------


def test_projects_with_nothing_recorded_are_skipped(workspace):
    """Opening a store for a project never recorded into costs time and returns
    nothing."""
    workspace.create("Empty")

    assert "empty" not in {p.meta.id for p in all_projects(workspace)}


def test_a_project_whose_indexing_failed_is_still_searchable(workspace, tmp_path):
    """Indexing is treated as non-fatal on purpose, so a project can have
    transcripts and no index. Filtering on the index would have made it
    invisible to @handle resolution as well as to search."""
    unindexed = workspace.create("Unindexed")
    unindexed.transcripts_dir.mkdir(parents=True, exist_ok=True)
    (unindexed.transcripts_dir / "mtg_u.json").write_text(
        transcript_for("Caching", "mtg_u", date(2026, 8, 17)).model_dump_json(),
        encoding="utf-8",
    )
    workspace.save(unindexed)

    assert not unindexed.memory_dir.exists()
    assert "unindexed" in {p.meta.id for p in all_projects(workspace)}
    assert resolve_across(all_projects(workspace), "caching")[1].ok
