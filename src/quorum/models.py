"""Core domain models.

Design rule that everything else depends on: **no extracted item exists without
evidence**. `Evidence` is a required field on every commitment, decision, risk and
open question, and it must quote the transcript verbatim. The verifier
(quorum.agents.verifier) mechanically checks each quote against the source
utterance and deletes anything it cannot ground. That turns "don't hallucinate"
from a prompt request into a code-enforced invariant.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from enum import Enum

from pydantic import BaseModel, Field, model_validator


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# Transcript primitives
# ---------------------------------------------------------------------------


class Speaker(BaseModel):
    """A meeting participant.

    `aliases` carries the messy ways people are actually referred to out loud -
    nicknames, first names, role labels ("the frontend guy"). The resolver uses
    these to map a spoken mention onto a real person.
    """

    id: str = Field(default_factory=lambda: _new_id("spk"))
    display_name: str
    email: str | None = None
    aliases: list[str] = Field(default_factory=list)
    github_login: str | None = None

    def matches(self, mention: str) -> bool:
        needle = mention.strip().lower()
        if not needle:
            return False
        candidates = [self.display_name, *self.aliases]
        if self.email:
            candidates.append(self.email.split("@")[0])
        return any(needle == c.strip().lower() for c in candidates if c)


class Utterance(BaseModel):
    """One contiguous turn of speech."""

    id: str = Field(default_factory=lambda: _new_id("utt"))
    index: int
    speaker_id: str
    text: str
    start_s: float | None = None
    end_s: float | None = None

    @property
    def timestamp(self) -> str:
        if self.start_s is None:
            return "--:--"
        minutes, seconds = divmod(int(self.start_s), 60)
        return f"{minutes:02d}:{seconds:02d}"


class Transcript(BaseModel):
    """A single meeting's transcript plus the roster needed to resolve mentions."""

    meeting_id: str = Field(default_factory=lambda: _new_id("mtg"))
    title: str = ""
    meeting_date: date
    speakers: list[Speaker] = Field(default_factory=list)
    utterances: list[Utterance] = Field(default_factory=list)
    source: str = "unknown"
    """Provenance: 'ami', 'synthetic', 'live', 'upload'. Kept because metrics must
    always be reported per-source - synthetic scores are not comparable to AMI."""

    project_id: str | None = None
    """Links meetings into a series. The cross-meeting ledger keys on this."""

    def speaker(self, speaker_id: str) -> Speaker | None:
        return next((s for s in self.speakers if s.id == speaker_id), None)

    def utterance(self, utterance_id: str) -> Utterance | None:
        return next((u for u in self.utterances if u.id == utterance_id), None)

    def resolve_mention(self, mention: str) -> Speaker | None:
        """Best-effort exact/alias match. Fuzzy and pronoun resolution live in
        the resolver agent - this is only the cheap deterministic path."""
        return next((s for s in self.speakers if s.matches(mention)), None)

    def as_dialogue(self, start: int = 0, end: int | None = None) -> str:
        """Render as speaker-labelled dialogue for prompting."""
        names = {s.id: s.display_name for s in self.speakers}
        lines = []
        for utt in self.utterances[start:end]:
            speaker = names.get(utt.speaker_id, utt.speaker_id)
            lines.append(f"[{utt.index}] {speaker} ({utt.timestamp}): {utt.text}")
        return "\n".join(lines)

    @property
    def duration_s(self) -> float | None:
        ends = [u.end_s for u in self.utterances if u.end_s is not None]
        return max(ends) if ends else None


class Segment(BaseModel):
    """A topic-coherent run of utterances.

    Segmentation is not cosmetic: the free-tier budget is 6,000 tokens/minute, so
    a 40-minute transcript can never be sent whole. Segments are the unit of
    extraction and keep every prompt small.
    """

    id: str = Field(default_factory=lambda: _new_id("seg"))
    meeting_id: str
    start_index: int
    end_index: int
    topic: str = ""

    @property
    def n_utterances(self) -> int:
        return self.end_index - self.start_index + 1


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------


class Evidence(BaseModel):
    """A verbatim citation into the transcript.

    `quote` must appear in the referenced utterance. `verified` and `match_score`
    are filled in by the verifier, never by the model.
    """

    utterance_id: str
    quote: str
    speaker_id: str | None = None
    timestamp: str | None = None
    verified: bool = False
    match_score: float = 0.0

    @model_validator(mode="after")
    def _quote_not_empty(self) -> Evidence:
        if not self.quote.strip():
            raise ValueError("Evidence.quote cannot be empty - an uncited item must not exist")
        return self


class Grounded(BaseModel):
    """Base for anything a model extracts. Evidence is required, minimum one item."""

    id: str = Field(default_factory=lambda: _new_id("item"))
    meeting_id: str = ""
    evidence: list[Evidence] = Field(min_length=1)

    @property
    def is_grounded(self) -> bool:
        return bool(self.evidence) and all(e.verified for e in self.evidence)


# ---------------------------------------------------------------------------
# Commitments
# ---------------------------------------------------------------------------


class CommitmentStrength(str, Enum):
    """How firm the commitment actually was.

    This single classification decides whether the product is usable. Treating
    every "we should probably look at that sometime" as a task generates dozens
    of fake items per meeting and nobody uses the tool twice.
    """

    FIRM = "firm"  # "I'll send it Friday" - explicit owner accepts explicit work
    TENTATIVE = "tentative"  # "I can maybe take a look" - hedged, no clear deadline
    MUSING = "musing"  # "someone should look into that" - no owner, aspirational


class CommitmentStatus(str, Enum):
    OPEN = "open"
    CLAIMED_DONE = "claimed_done"  # someone said it was done; not yet verified
    VERIFIED_DONE = "verified_done"  # confirmed against GitHub/Gmail evidence
    OVERDUE = "overdue"
    BLOCKED = "blocked"  # waiting on another commitment
    DROPPED = "dropped"  # deadline passed, no activity, nobody mentioned it again
    CANCELLED = "cancelled"  # explicitly called off in a later meeting


class DeadlineResolution(str, Enum):
    EXPLICIT = "explicit"  # "by March 14th"
    RELATIVE = "relative"  # "end of next week" -> computed from meeting date
    ANCHORED = "anchored"  # "before the demo" -> looked up against the calendar
    NONE = "none"  # no deadline was stated


class Deadline(BaseModel):
    raw_text: str | None = None
    resolved: date | None = None
    method: DeadlineResolution = DeadlineResolution.NONE
    confidence: float = 0.0


class Assignee(BaseModel):
    """Who owns the commitment.

    `raw_mention` keeps what was literally said ("you", "Yug", "the frontend
    team") so the resolver's accuracy can be scored against ground truth.
    """

    raw_mention: str | None = None
    speaker_id: str | None = None
    display_name: str | None = None
    email: str | None = None
    github_login: str | None = None
    """Carried through so the reality-verification layer can search that
    person's pull requests without re-resolving the identity."""

    confidence: float = 0.0
    unresolved_reason: str | None = None


class Commitment(Grounded):
    """Something a person said they would do."""

    description: str
    assignee: Assignee = Field(default_factory=Assignee)
    deadline: Deadline = Field(default_factory=Deadline)
    strength: CommitmentStrength = CommitmentStrength.TENTATIVE
    status: CommitmentStatus = CommitmentStatus.OPEN

    project_id: str | None = None
    created_on: date | None = None
    depends_on: list[str] = Field(default_factory=list)
    """Commitment ids this one waits on. Lets the planner recompute a chain when
    an upstream deadline slips."""

    # --- lifecycle, appended by the between-meetings planner ---
    nudge_count: int = 0
    last_action_on: date | None = None
    resolution_note: str | None = None

    @property
    def is_actionable(self) -> bool:
        """Only firm commitments with a real owner are worth acting on."""
        return self.strength is CommitmentStrength.FIRM and self.assignee.speaker_id is not None


class StatusKind(str, Enum):
    DELIVERED = "delivered"
    SLIPPED = "slipped"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class StatusUpdate(Grounded):
    """News about a commitment made in an *earlier* meeting.

    This is what makes tracking cross-meeting rather than per-meeting. "I didn't
    get to the spec, I'll have it Friday" creates nothing new - it moves an
    existing obligation. Without this, every restatement looks like a fresh
    commitment and the open list fills with duplicates of the same work.
    """

    about: str
    """The work being referred to, as spoken. Matched against the ledger."""

    kind: StatusKind
    blocker: str | None = None
    new_deadline_text: str | None = None


class Decision(Grounded):
    """A choice the group made. Tracked across meetings so reversals are detectable."""

    statement: str
    reverses: str | None = None  # id of an earlier Decision this contradicts


class OpenQuestion(Grounded):
    question: str
    raised_by: str | None = None


class Risk(Grounded):
    description: str
    severity: str = "medium"


# ---------------------------------------------------------------------------
# Aggregates
# ---------------------------------------------------------------------------


class MeetingRecord(BaseModel):
    """Everything the agent extracted from one meeting, post-verification."""

    meeting_id: str
    project_id: str | None = None
    meeting_date: date
    title: str = ""
    summary: str = ""
    commitments: list[Commitment] = Field(default_factory=list)
    decisions: list[Decision] = Field(default_factory=list)
    open_questions: list[OpenQuestion] = Field(default_factory=list)
    risks: list[Risk] = Field(default_factory=list)
    status_updates: list[StatusUpdate] = Field(default_factory=list)

    # --- provenance for the metrics dashboard ---
    extracted_at: datetime = Field(default_factory=datetime.now)
    rejected_items: int = 0
    """Count of model-proposed items the verifier deleted as ungrounded. This is
    the numerator of the hallucinated-commitment rate."""

    tokens_used: int = 0
    latency_s: float = 0.0

    @property
    def actionable_commitments(self) -> list[Commitment]:
        return [c for c in self.commitments if c.is_actionable]
