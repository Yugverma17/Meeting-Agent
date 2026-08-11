"""The commitment ledger - state that outlives a single meeting.

Everything upstream of here is per-meeting. The ledger is what turns a pile of
extractions into an obligation with a history: made in week 2, slipped in week 3,
chased twice, delivered silently in week 5.

The hard part is **merging**. When someone says "I didn't get to the spec, I'll
have it Friday", that is the *same* obligation with a new date, not a second
one. Getting this wrong is the failure that makes cross-meeting tracking
worthless: duplicates inflate the open list, and every duplicate becomes a
separate nag to the same person about the same work.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date
from enum import Enum
from pathlib import Path

from rapidfuzz import fuzz

from quorum.agents.dates import resolve_deadline
from quorum.models import (
    Commitment,
    CommitmentStatus,
    Deadline,
    Decision,
    MeetingRecord,
    StatusKind,
    StatusUpdate,
)

log = logging.getLogger(__name__)


class MergeOutcome(str, Enum):
    ADDED = "added"
    UPDATED = "updated"
    """Matched an existing commitment - typically a slip with a new deadline."""

    IGNORED = "ignored"
    """Not actionable (a musing, or unowned), so never entered the ledger."""


@dataclass
class LedgerStats:
    added: int = 0
    updated: int = 0
    ignored: int = 0
    status_applied: int = 0

    def record(self, outcome: MergeOutcome) -> None:
        setattr(self, outcome.value, getattr(self, outcome.value) + 1)

    def as_dict(self) -> dict[str, int]:
        return {
            "added": self.added,
            "updated": self.updated,
            "ignored": self.ignored,
            "status_applied": self.status_applied,
        }


class Ledger:
    """Commitments for one project, accumulated across meetings."""

    MATCH_THRESHOLD = 82.0
    """Fuzzy score above which two descriptions are treated as the same work.
    Tuned to merge "the ingestion API spec" with "ingestion API spec document"
    while keeping "the auth migration" and "the schema migration" apart."""

    def __init__(self, project_id: str) -> None:
        self.project_id = project_id
        self.commitments: list[Commitment] = []
        self.decisions: list[Decision] = []
        self.meeting_dates: dict[str, date] = {}

    # -- ingestion ---------------------------------------------------------

    def ingest(self, record: MeetingRecord) -> LedgerStats:
        """Fold one meeting's verified output into the running ledger."""
        stats = LedgerStats()
        self.meeting_dates[record.meeting_id] = record.meeting_date

        for commitment in record.commitments:
            stats.record(self.merge(commitment))

        # Status updates are applied after new commitments, so an update can
        # refer to work committed to earlier in the same meeting.
        for update in record.status_updates:
            stats.status_applied += int(self.apply_status(update, record.meeting_date))

        for decision in record.decisions:
            self.decisions.append(decision)

        return stats

    def apply_status(self, update: StatusUpdate, meeting_date: date) -> bool:
        """Apply news about an existing commitment. Returns whether it landed.

        An update that matches nothing is dropped rather than turned into a new
        commitment: "I sent that Tuesday" about work we never recorded is a gap
        in our extraction, not a fresh obligation to chase.
        """
        target = self._find_by_description(update.about)
        if target is None:
            log.debug("Status update about %r matched no commitment", update.about)
            return False

        if update.kind is StatusKind.DELIVERED:
            # Claimed, not confirmed. Someone saying it is done is a report, not
            # proof - only the evidence layer may set VERIFIED_DONE. But we stop
            # nagging, because chasing someone who just told you they delivered
            # is the fastest way to get the tool switched off.
            target.status = CommitmentStatus.CLAIMED_DONE
            target.resolution_note = f"claimed complete on {meeting_date.isoformat()}"

        elif update.kind is StatusKind.CANCELLED:
            target.status = CommitmentStatus.CANCELLED
            target.resolution_note = f"cancelled on {meeting_date.isoformat()}"

        elif update.kind is StatusKind.SLIPPED:
            resolved = resolve_deadline(update.new_deadline_text, meeting_date)
            if resolved.value:
                target.deadline = Deadline(
                    raw_text=update.new_deadline_text, resolved=resolved.value,
                    method=resolved.method, confidence=resolved.confidence,
                )
            target.status = CommitmentStatus.OPEN
            # A slip resets the chase clock: the owner has just given a new date,
            # so nudging them tomorrow about the old one would be nonsense.
            target.nudge_count = 0
            target.last_action_on = meeting_date

        elif update.kind is StatusKind.BLOCKED:
            target.status = CommitmentStatus.BLOCKED
            if update.blocker:
                upstream = self._find_by_description(update.blocker, exclude=target.id)
                if upstream is not None and upstream.id not in target.depends_on:
                    target.depends_on.append(upstream.id)

        target.evidence.extend(update.evidence)
        return True

    def _find_by_description(self, text: str, exclude: str | None = None) -> Commitment | None:
        """Locate a commitment by how it was spoken about, regardless of owner."""
        if not text or not text.strip():
            return None
        best, best_score = None, 0.0
        for candidate in self.commitments:
            if candidate.id == exclude:
                continue
            score = fuzz.token_set_ratio(candidate.description.lower(), text.lower())
            if score > best_score:
                best, best_score = candidate, score
        return best if best_score >= self.MATCH_THRESHOLD else None

    def merge(self, incoming: Commitment) -> MergeOutcome:
        """Add, or fold into an existing commitment if it is the same work.

        Only actionable commitments enter the ledger. A musing or an unowned
        item is real conversation but not an obligation, and admitting it here
        is what produces a nag list nobody trusts.
        """
        if not incoming.is_actionable:
            return MergeOutcome.IGNORED

        existing = self._find_match(incoming)
        if existing is None:
            self.commitments.append(incoming)
            return MergeOutcome.ADDED

        # Same obligation, restated. Take the newer deadline, keep the original
        # identity and history.
        if incoming.deadline.resolved and (
            existing.deadline.resolved is None
            or incoming.deadline.resolved > existing.deadline.resolved
        ):
            existing.deadline = incoming.deadline
            existing.status = CommitmentStatus.OPEN

        existing.evidence.extend(incoming.evidence)
        if incoming.created_on:
            existing.last_action_on = incoming.created_on
        return MergeOutcome.UPDATED

    def _find_match(self, incoming: Commitment) -> Commitment | None:
        """Same owner and near-identical description means the same work."""
        best, best_score = None, 0.0
        for candidate in self.commitments:
            if candidate.status in (
                CommitmentStatus.CANCELLED,
                CommitmentStatus.VERIFIED_DONE,
            ):
                continue
            if candidate.assignee.speaker_id != incoming.assignee.speaker_id:
                continue
            score = fuzz.token_set_ratio(
                candidate.description.lower(), incoming.description.lower()
            )
            if score > best_score:
                best, best_score = candidate, score
        return best if best_score >= self.MATCH_THRESHOLD else None

    # -- queries -----------------------------------------------------------

    def open_commitments(self) -> list[Commitment]:
        closed = (
            CommitmentStatus.VERIFIED_DONE,
            CommitmentStatus.CANCELLED,
            CommitmentStatus.DROPPED,
        )
        return [c for c in self.commitments if c.status not in closed]

    def overdue(self, today: date) -> list[Commitment]:
        return [
            c
            for c in self.open_commitments()
            if c.deadline.resolved and c.deadline.resolved < today
        ]

    def for_owner(self, speaker_id: str) -> list[Commitment]:
        return [c for c in self.commitments if c.assignee.speaker_id == speaker_id]

    def by_id(self, commitment_id: str) -> Commitment | None:
        return next((c for c in self.commitments if c.id == commitment_id), None)

    def contradictions(self) -> list[tuple[Decision, Decision]]:
        """Pairs where a later decision was marked as reversing an earlier one."""
        by_id = {d.id: d for d in self.decisions}
        pairs = []
        for decision in self.decisions:
            if decision.reverses and decision.reverses in by_id:
                pairs.append((by_id[decision.reverses], decision))
        return pairs

    # -- persistence -------------------------------------------------------

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "project_id": self.project_id,
            "commitments": [json.loads(c.model_dump_json()) for c in self.commitments],
            "decisions": [json.loads(d.model_dump_json()) for d in self.decisions],
            "meeting_dates": {k: v.isoformat() for k, v in self.meeting_dates.items()},
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Ledger:
        payload = json.loads(path.read_text(encoding="utf-8"))
        ledger = cls(payload["project_id"])
        ledger.commitments = [Commitment.model_validate(c) for c in payload["commitments"]]
        ledger.decisions = [Decision.model_validate(d) for d in payload["decisions"]]
        ledger.meeting_dates = {
            k: date.fromisoformat(v) for k, v in payload.get("meeting_dates", {}).items()
        }
        return ledger
