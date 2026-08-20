"""The checkpointed ingest graph.

The claim this file exists to defend is narrow and specific: **a run that dies
partway does not lose the work already done**. That is the only reason a graph
library is here rather than five function calls in a row, so it is the thing
worth testing hardest.

No model is called. The extractor is replaced with a stub that can be told to
fail on demand, which is the only way to reproduce the situation this is all
for - a quota wall on segment nine - without waiting for one.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from quorum.agents.extractor import ExtractionResult, ExtractionStats
from quorum.models import (
    Assignee,
    Commitment,
    Deadline,
    Decision,
    Evidence,
)
from quorum.pipeline import graph as graph_module
from quorum.pipeline.graph import (
    IngestGraph,
    RunStatus,
    interrupted_runs,
    record_from_state,
)


@pytest.fixture
def checkpointer(tmp_path, monkeypatch):
    """A checkpoint database under tmp_path, so tests never touch runs/."""
    from langgraph.checkpoint.sqlite import SqliteSaver

    monkeypatch.setattr(graph_module, "RUNS_DIR", tmp_path)
    conn = sqlite3.connect(tmp_path / "pipeline.sqlite", check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    yield saver
    conn.close()


class StubExtractor:
    """Stands in for the one stage that costs quota.

    `fail_times` reproduces the failure this whole design is for: the first N
    attempts raise, later ones succeed - which is exactly what a rate limit
    looks like from inside the pipeline.
    """

    calls = 0
    fail_times = 0
    error: Exception | None = None

    def __init__(self, *args, **kwargs) -> None:
        pass

    def extract(self, transcript, segments) -> ExtractionResult:
        type(self).calls += 1
        if type(self).calls <= type(self).fail_times:
            raise type(self).error or RuntimeError("quota exhausted")

        utterance = transcript.utterances[1]
        return ExtractionResult(
            commitments=[
                Commitment(
                    description="send the spec document",
                    meeting_id=transcript.meeting_id,
                    assignee=Assignee(raw_mention="I"),
                    deadline=Deadline(raw_text="by Friday"),
                    evidence=[Evidence(
                        utterance_id=utterance.id,
                        quote="I'll have the spec document to you by Friday",
                    )],
                )
            ],
            decisions=[
                Decision(
                    statement="Sam reviews the spec once it lands",
                    meeting_id=transcript.meeting_id,
                    evidence=[Evidence(
                        utterance_id=transcript.utterances[3].id,
                        quote="Sure, I'll take a look over the weekend",
                    )],
                )
            ],
            stats=ExtractionStats(segments=len(segments), llm_calls=1, prompt_tokens=800),
        )


@pytest.fixture(autouse=True)
def stub_extraction(monkeypatch):
    StubExtractor.calls = 0
    StubExtractor.fail_times = 0
    StubExtractor.error = None
    monkeypatch.setattr("quorum.agents.extractor.Extractor", StubExtractor)
    return StubExtractor


# --- the happy path ---------------------------------------------------------


def test_every_stage_runs_in_order(transcript, checkpointer):
    outcome = IngestGraph(checkpointer=checkpointer).run(transcript)

    assert outcome.status is RunStatus.COMPLETE
    assert outcome.state["completed_stages"] == ["segment", "extract", "verify", "resolve"]


def test_the_run_produces_a_meeting_record(transcript, checkpointer):
    outcome = IngestGraph(checkpointer=checkpointer).run(transcript)

    record = outcome.record
    assert record.meeting_id == transcript.meeting_id
    assert [c.description for c in record.commitments] == ["send the spec document"]
    assert record.tokens_used == 800


def test_resolution_still_happens_inside_the_graph(transcript, checkpointer):
    """The nodes wrap the existing agents rather than reimplementing them - so
    "I" must still resolve to whoever spoke the cited line."""
    outcome = IngestGraph(checkpointer=checkpointer).run(transcript)

    (commitment,) = outcome.record.commitments
    assert commitment.assignee.display_name == "Yug Verma"


def test_ungrounded_items_are_still_deleted(transcript, checkpointer, monkeypatch):
    """The verifier is a node now; it must not have become advisory."""
    def fabricate(self, transcript_, segments):
        return ExtractionResult(
            commitments=[
                Commitment(
                    description="a commitment nobody made",
                    meeting_id=transcript_.meeting_id,
                    assignee=Assignee(raw_mention="Priya"),
                    evidence=[Evidence(
                        utterance_id=transcript_.utterances[0].id,
                        quote="I will personally rewrite the billing system",
                    )],
                )
            ],
            stats=ExtractionStats(),
        )

    monkeypatch.setattr(StubExtractor, "extract", fabricate)
    outcome = IngestGraph(checkpointer=checkpointer).run(transcript)

    assert outcome.record.commitments == []
    assert outcome.record.rejected_items == 1


def test_indexing_is_skipped_when_there_is_no_project(transcript, checkpointer):
    """The one real branch in the graph. Running the node anyway would load an
    embedding model to do nothing with it."""
    outcome = IngestGraph(checkpointer=checkpointer).run(transcript, project_id=None)

    assert "index" not in outcome.state["completed_stages"]


# --- what the graph is actually for -----------------------------------------


def test_a_failure_partway_keeps_the_work_already_done(
    transcript, checkpointer, stub_extraction
):
    """The reason this is a graph. Before checkpointing, a quota wall during
    extraction discarded the transcription that produced the segments - and the
    audio-seconds it cost do not come back."""
    stub_extraction.fail_times = 1
    outcome = IngestGraph(checkpointer=checkpointer).run(transcript)

    assert outcome.status is RunStatus.INTERRUPTED
    assert outcome.state["completed_stages"] == ["segment"]
    assert outcome.state["segments"], "segmentation survived the failure"
    assert outcome.state["transcript"]["utterances"], "so did the transcript"


def test_resume_restarts_at_the_failed_stage_not_the_beginning(
    transcript, checkpointer, stub_extraction
):
    stub_extraction.fail_times = 1
    pipeline = IngestGraph(checkpointer=checkpointer)
    pipeline.run(transcript)

    outcome = pipeline.resume(transcript.meeting_id)

    assert outcome.status is RunStatus.COMPLETE
    assert outcome.resumed_from == ["segment"], "segmentation was not redone"
    assert outcome.stages_run == ["extract", "verify", "resolve"]
    assert stub_extraction.calls == 2, "extraction was retried exactly once"


def test_resuming_an_unknown_run_says_so(transcript, checkpointer):
    outcome = IngestGraph(checkpointer=checkpointer).resume("mtg_never_ran")

    assert outcome.status is RunStatus.NOT_FOUND
    assert "mtg_never_ran" in outcome.error


def test_a_second_failure_keeps_the_progress_from_the_first(
    transcript, checkpointer, stub_extraction
):
    stub_extraction.fail_times = 2
    pipeline = IngestGraph(checkpointer=checkpointer)
    pipeline.run(transcript)

    again = pipeline.resume(transcript.meeting_id)
    assert again.status is RunStatus.INTERRUPTED
    assert again.state["completed_stages"] == ["segment"]

    third = pipeline.resume(transcript.meeting_id)
    assert third.status is RunStatus.COMPLETE


def test_the_error_that_stopped_the_run_is_reported(
    transcript, checkpointer, stub_extraction
):
    """"It failed" is not actionable. Whether it was quota or a parse error
    decides whether waiting an hour is the fix."""
    from quorum.llm.router import QuotaExhausted

    stub_extraction.fail_times = 1
    stub_extraction.error = QuotaExhausted("all models exhausted for tier balanced")

    outcome = IngestGraph(checkpointer=checkpointer).run(transcript)

    assert "QuotaExhausted" in outcome.error
    assert "all models exhausted" in outcome.error


# --- checkpoint contents ----------------------------------------------------


def test_checkpointed_state_is_plain_json(transcript, checkpointer, tmp_path):
    """Deliberate. A checkpoint you cannot read with sqlite3 and json.loads is
    a checkpoint you cannot debug at the moment you need to - and it breaks the
    first time a model class changes shape."""
    IngestGraph(checkpointer=checkpointer).run(transcript)

    state = IngestGraph(checkpointer=checkpointer).state_of(transcript.meeting_id)
    round_tripped = json.loads(json.dumps(state))

    assert round_tripped["transcript"]["meeting_id"] == transcript.meeting_id
    assert round_tripped["completed_stages"] == ["segment", "extract", "verify", "resolve"]


def test_interrupted_runs_lists_the_unfinished_and_nothing_else(
    transcript, checkpointer, stub_extraction
):
    stub_extraction.fail_times = 1
    pipeline = IngestGraph(checkpointer=checkpointer)
    pipeline.run(transcript)

    (run,) = interrupted_runs()
    assert run.meeting_id == transcript.meeting_id
    assert run.completed_stages == ["segment"]
    assert run.next_stage == "extract", "which stage `resume` would start at"

    pipeline.resume(transcript.meeting_id)
    assert interrupted_runs() == [], "a finished run is no longer offered for resume"


def test_no_checkpoint_database_is_not_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(graph_module, "RUNS_DIR", tmp_path / "nothing-here")
    assert interrupted_runs() == []


# --- assembling the record --------------------------------------------------


def test_record_from_state_survives_a_partial_run(transcript):
    """Used by `resume` on a run that has segments but no extraction yet."""
    state = {
        "transcript": transcript.model_dump(mode="json"),
        "project_id": "proj",
        "title": "Weekly sync",
    }
    record = record_from_state(state)

    assert record.commitments == [] and record.decisions == []
    assert record.title == "Weekly sync"
    assert record.project_id == "proj"
