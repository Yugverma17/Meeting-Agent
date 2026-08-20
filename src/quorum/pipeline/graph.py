"""The per-meeting pipeline as a checkpointed LangGraph.

## Why a graph library at all

The stages have not changed: segment, extract, verify, resolve, index. Written
as five function calls in a row they worked, and a graph adds nothing to a
straight line. What it adds is **durability**, and on this project that is not a
nicety.

Everything runs on free-tier quota. Groq allows 6,000 tokens/minute and Gemini
250 requests/day, and a 50-minute lecture is roughly forty thousand tokens split
across a dozen segments. Runs die partway. Before this, a `QuotaExhausted` on
segment nine threw away segments one to eight - along with the transcription
that produced them, which cost audio-seconds from a daily budget that does not
refill on demand. The work was gone and the only option was to run it again
tomorrow.

With a checkpointer, state is persisted after every node. `quorum resume` picks
up at the node that failed, with everything before it intact. That is a real
capability the hand-rolled version did not have, and it is the reason this file
exists. A graph adopted for any weaker reason would be decoration.

## What is deliberately *not* a graph

The between-meetings planner. It is a fixed set of date comparisons, and it is
deterministic on purpose - so that "why did it email my manager?" has an exact
answer, so that a daily sweep over hundreds of commitments costs nothing, and so
that escalation timing can be scored against ground truth. Expressing those
rules as a graph would make them harder to read and harder to test while
changing no behaviour. Being able to point at one loop and say why it is *not*
here is part of knowing the tool.

## State is JSON, not objects

Nodes take and return plain dicts, converting to Pydantic models at the edges.
It costs a serialisation round trip per node and buys two things: checkpoints
that survive a change to the model classes, and a checkpoint file that can be
read with `sqlite3` and `json.loads` when something has gone wrong. Debugging a
pickled object graph inside a checkpoint row is exactly the situation this
project's plain-JSON ledger exists to avoid.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Any, TypedDict

from quorum.config import RUNS_DIR
from quorum.models import (
    Commitment,
    Decision,
    MeetingRecord,
    Segment,
    StatusUpdate,
    Transcript,
)

log = logging.getLogger(__name__)

STAGES = ("segment", "extract", "verify", "resolve", "index")


class RunStatus(str, Enum):
    COMPLETE = "complete"
    INTERRUPTED = "interrupted"
    NOT_FOUND = "not_found"


class IngestState(TypedDict, total=False):
    """What flows between nodes. Everything here is JSON-serialisable."""

    transcript: dict[str, Any]
    project_id: str | None
    title: str

    segments: list[dict[str, Any]]
    commitments: list[dict[str, Any]]
    decisions: list[dict[str, Any]]
    status_updates: list[dict[str, Any]]

    rejected: int
    indexed: int
    completed_stages: list[str]
    """Appended by each node. This is what makes a resumed run legible: it says
    what had already been done when the previous attempt died."""

    stage_errors: list[str]
    tokens: int
    started_at: float


def checkpoint_path() -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    return RUNS_DIR / "pipeline.sqlite"


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def _transcript(state: IngestState) -> Transcript:
    return Transcript.model_validate(state["transcript"])


def _dump(items) -> list[dict[str, Any]]:
    return [item.model_dump(mode="json") for item in items]


def _done(state: IngestState, stage: str) -> list[str]:
    return [*state.get("completed_stages", []), stage]


def segment_node(state: IngestState) -> dict[str, Any]:
    """Topic-coherent chunks. Local and free - no model call, no quota."""
    from quorum.agents.embedding import LexicalEmbedder
    from quorum.agents.segmenter import Segmenter

    # Lexical, not the ONNX model - matching every other segmentation call site
    # in the project. Finding topic boundaries does not need semantic
    # embeddings, and the hashing embedder keeps this node instant and offline.
    # The ONNX model is reserved for retrieval, where paraphrase actually has to
    # match.
    transcript = _transcript(state)
    segments = Segmenter(embedder=LexicalEmbedder()).segment(transcript)
    log.info("segment: %d segment(s) from %d utterances",
             len(segments), len(transcript.utterances))
    return {"segments": _dump(segments), "completed_stages": _done(state, "segment")}


def extract_node(state: IngestState) -> dict[str, Any]:
    """The expensive stage: one model call per segment. This is where runs die."""
    from quorum.agents.extractor import Extractor

    transcript = _transcript(state)
    segments = [Segment.model_validate(s) for s in state.get("segments", [])]
    result = Extractor().extract(transcript, segments)

    return {
        "commitments": _dump(result.commitments),
        "decisions": _dump(result.decisions),
        "status_updates": _dump(result.status_updates),
        "tokens": state.get("tokens", 0) + result.stats.total_tokens,
        "completed_stages": _done(state, "extract"),
    }


def verify_node(state: IngestState) -> dict[str, Any]:
    """The grounding gate. Deterministic, so a resumed run reproduces it exactly."""
    from quorum.agents.verifier import GroundingVerifier

    transcript = _transcript(state)
    verifier = GroundingVerifier()

    commitments, report = verifier.verify(
        [Commitment.model_validate(c) for c in state.get("commitments", [])], transcript
    )
    decisions, decision_report = verifier.verify(
        [Decision.model_validate(d) for d in state.get("decisions", [])], transcript
    )
    updates, update_report = verifier.verify(
        [StatusUpdate.model_validate(u) for u in state.get("status_updates", [])], transcript
    )

    rejected = report.rejected + decision_report.rejected + update_report.rejected
    log.info("verify: %d item(s) deleted as ungrounded", rejected)
    return {
        "commitments": _dump(commitments),
        "decisions": _dump(decisions),
        "status_updates": _dump(updates),
        "rejected": rejected,
        "completed_stages": _done(state, "verify"),
    }


def resolve_node(state: IngestState) -> dict[str, Any]:
    """Who, and by when. Deterministic first; only ambiguity reaches a model."""
    from quorum.agents.resolver import Resolver

    transcript = _transcript(state)
    commitments = [Commitment.model_validate(c) for c in state.get("commitments", [])]
    Resolver().resolve(commitments, transcript)
    return {"commitments": _dump(commitments), "completed_stages": _done(state, "resolve")}


def index_node(state: IngestState) -> dict[str, Any]:
    """Fold into the project's vector memory. Only runs for a saved project."""
    from quorum.memory import ProjectMemory
    from quorum.workspace import Workspace

    project_id = state.get("project_id")
    project = Workspace().get(project_id) if project_id else None
    if project is None:
        return {"indexed": 0, "completed_stages": _done(state, "index")}

    transcript = _transcript(state)
    record = record_from_state(state)
    try:
        indexed = ProjectMemory(project.memory_dir).index_meeting(record, transcript)
    except Exception as exc:  # noqa: BLE001 - retrieval is an optimisation, never a blocker
        log.warning("Indexing failed (%s); the meeting is still saved", exc)
        return {
            "indexed": 0,
            "stage_errors": [*state.get("stage_errors", []), f"index: {exc}"],
            "completed_stages": _done(state, "index"),
        }
    return {"indexed": indexed, "completed_stages": _done(state, "index")}


def wants_indexing(state: IngestState) -> str:
    """The one real branch in the graph: a meeting with no project has nowhere
    to be indexed, and running the node anyway would load an embedding model to
    do nothing with it."""
    return "index" if state.get("project_id") else "__end__"


# ---------------------------------------------------------------------------
# Assembling
# ---------------------------------------------------------------------------


def build_ingest_graph(checkpointer=None, with_checkpoints: bool = True):
    """Compile the graph. `with_checkpoints=False` gives an in-memory run."""
    from langgraph.graph import END, START, StateGraph

    builder = StateGraph(IngestState)
    builder.add_node("segment", segment_node)
    builder.add_node("extract", extract_node)
    builder.add_node("verify", verify_node)
    builder.add_node("resolve", resolve_node)
    builder.add_node("index", index_node)

    builder.add_edge(START, "segment")
    builder.add_edge("segment", "extract")
    builder.add_edge("extract", "verify")
    builder.add_edge("verify", "resolve")
    builder.add_conditional_edges(
        "resolve", wants_indexing, {"index": "index", "__end__": END}
    )
    builder.add_edge("index", END)

    if checkpointer is None and with_checkpoints:
        checkpointer = _sqlite_checkpointer()
    return builder.compile(checkpointer=checkpointer)


def _sqlite_checkpointer():
    from langgraph.checkpoint.sqlite import SqliteSaver

    # check_same_thread=False because the CLI compiles the graph on one thread
    # and LangGraph may execute nodes on another. The default raises there, and
    # it presents as an unrelated-looking sqlite error mid-run.
    conn = sqlite3.connect(checkpoint_path(), check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    return saver


# ---------------------------------------------------------------------------
# Driving it
# ---------------------------------------------------------------------------


@dataclass
class IngestOutcome:
    record: MeetingRecord | None
    state: IngestState
    status: RunStatus
    error: str = ""
    resumed_from: list[str] = field(default_factory=list)
    latency_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status is RunStatus.COMPLETE

    @property
    def stages_run(self) -> list[str]:
        return [s for s in self.state.get("completed_stages", []) if s not in self.resumed_from]


class IngestGraph:
    """One meeting through the pipeline, resumable by meeting id.

    The meeting id is the LangGraph thread id. That is the whole resume story:
    invoking again with the same id continues from the last checkpoint instead
    of starting over, so `resume` needs no bookkeeping of its own.
    """

    def __init__(self, checkpointer=None, with_checkpoints: bool = True) -> None:
        self.graph = build_ingest_graph(checkpointer, with_checkpoints)
        self.with_checkpoints = with_checkpoints or checkpointer is not None

    def run(self, transcript: Transcript, project_id: str | None = None) -> IngestOutcome:
        state: IngestState = {
            "transcript": transcript.model_dump(mode="json"),
            "project_id": project_id,
            "title": transcript.title,
            "completed_stages": [],
            "stage_errors": [],
            "started_at": time.time(),
        }
        return self._invoke(transcript.meeting_id, state)

    def resume(self, meeting_id: str) -> IngestOutcome:
        """Continue an interrupted run. Passing None as input is what tells
        LangGraph to carry on from the checkpoint rather than start again."""
        saved = self.state_of(meeting_id)
        if saved is None:
            return IngestOutcome(
                None, {}, RunStatus.NOT_FOUND,
                error=f"No checkpointed run for {meeting_id}",
            )
        already = list(saved.get("completed_stages", []))
        outcome = self._invoke(meeting_id, None)
        outcome.resumed_from = already
        return outcome

    def _invoke(self, thread_id: str, state: IngestState | None) -> IngestOutcome:
        config = {"configurable": {"thread_id": thread_id}}
        started = time.time()
        try:
            final = self.graph.invoke(state, config=config)
        except Exception as exc:  # noqa: BLE001 - quota, provider and parse failures alike
            saved = self.state_of(thread_id) or (state or {})
            reached = saved.get("completed_stages", [])
            log.warning(
                "Pipeline interrupted after %s (%s: %s)",
                reached or "no stage", type(exc).__name__, exc,
            )
            return IngestOutcome(
                None, saved, RunStatus.INTERRUPTED,
                error=f"{type(exc).__name__}: {exc}",
                latency_s=time.time() - started,
            )

        return IngestOutcome(
            record_from_state(final), final, RunStatus.COMPLETE,
            latency_s=time.time() - started,
        )

    # -- checkpoint inspection --------------------------------------------

    def state_of(self, meeting_id: str) -> IngestState | None:
        if not self.with_checkpoints:
            return None
        try:
            snapshot = self.graph.get_state({"configurable": {"thread_id": meeting_id}})
        except Exception as exc:  # noqa: BLE001 - a missing thread raises variously
            log.debug("No checkpoint for %s (%s)", meeting_id, exc)
            return None
        if snapshot is None or not snapshot.values:
            return None
        return snapshot.values


def record_from_state(state: IngestState) -> MeetingRecord:
    """Assemble the meeting record from whatever the graph produced."""
    transcript = _transcript(state)
    return MeetingRecord(
        meeting_id=transcript.meeting_id,
        project_id=state.get("project_id"),
        meeting_date=transcript.meeting_date,
        title=state.get("title") or transcript.title,
        commitments=[Commitment.model_validate(c) for c in state.get("commitments", [])],
        decisions=[Decision.model_validate(d) for d in state.get("decisions", [])],
        status_updates=[
            StatusUpdate.model_validate(u) for u in state.get("status_updates", [])
        ],
        rejected_items=state.get("rejected", 0),
        tokens_used=state.get("tokens", 0),
        latency_s=time.time() - state.get("started_at", time.time()),
    )


@dataclass
class InterruptedRun:
    meeting_id: str
    title: str
    meeting_date: date | None
    completed_stages: list[str]

    @property
    def next_stage(self) -> str:
        for stage in STAGES:
            if stage not in self.completed_stages:
                return stage
        return "done"


def interrupted_runs(limit: int = 20) -> list[InterruptedRun]:
    """Every checkpointed run that never reached the end.

    Read straight out of the checkpoint database rather than tracked separately,
    so this cannot drift from what `resume` would actually do.
    """
    path = checkpoint_path()
    if not path.exists():
        return []

    from langgraph.checkpoint.sqlite import SqliteSaver

    conn = sqlite3.connect(path, check_same_thread=False)
    try:
        saver = SqliteSaver(conn)

        # Every super-step writes a checkpoint, so one run leaves several rows
        # and `list` makes no promise about their order. Taking the first row
        # per thread reported a completed run as interrupted, because the row it
        # happened to see was from the middle of it. Collect the thread ids,
        # then ask for each one's *latest* checkpoint explicitly.
        threads: list[str] = []
        for item in saver.list(None, limit=limit * 12):
            thread_id = item.config.get("configurable", {}).get("thread_id", "")
            if thread_id and thread_id not in threads:
                threads.append(thread_id)

        found = []
        for thread_id in threads:
            latest = saver.get_tuple({"configurable": {"thread_id": thread_id}})
            if latest is None:
                continue
            values = latest.checkpoint.get("channel_values", {}) or {}
            stages = list(values.get("completed_stages", []))
            if not stages or _finished(values, stages):
                continue
            transcript = values.get("transcript") or {}
            found.append(InterruptedRun(
                meeting_id=thread_id,
                title=values.get("title") or transcript.get("title", ""),
                meeting_date=_as_date(transcript.get("meeting_date")),
                completed_stages=stages,
            ))
        return found[:limit]
    finally:
        conn.close()


def _finished(values: dict[str, Any], stages: list[str]) -> bool:
    """A run ended cleanly if it indexed, or reached resolve with no project to
    index into - which is exactly the conditional edge the graph takes."""
    if "index" in stages:
        return True
    return "resolve" in stages and not values.get("project_id")


def _as_date(value: Any) -> date | None:
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError:
            return None
    return value if isinstance(value, date) else None
