"""What changed between meetings.

Every meeting tool summarises a *meeting*. Nothing summarises the **gap**. That
gap is where this project's whole claim lives: a commitment that quietly
evaporates does so between meetings, a decision reverses across two of them, and
work delivered without announcement is invisible in any single transcript.

None of the five sections below can be produced from one meeting, and that is
the point of each:

- **Reversed** needs two meetings to compare.
- **Dropped** is defined by *absence* - you cannot extract silence from a
  transcript, only notice it by comparing a ledger to itself over time.
- **Delivered quietly** needs evidence from outside the conversation entirely.
- **Slipping** needs the history of a date, not its current value. A deadline
  that has moved three times is a different problem from one that is merely
  late, and only the history tells them apart.
- **Newly promised** is the one thing a meeting summary already covers, and it
  is here for balance: a report that only lists failures reads as an
  indictment and gets closed.

Nothing here calls a model. Every line is a query over state the ledger already
holds, which makes the report free, instant, and identical on every run -
properties that matter for something meant to be read every Monday.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta

from quorum.models import Commitment, CommitmentStatus, Decision
from quorum.tracking.ledger import Ledger

log = logging.getLogger(__name__)

DEFAULT_WINDOW_DAYS = 7
QUIET_DAYS = 14
"""How long a commitment must be overdue and unmentioned before "dropped" is a
fair description rather than an accusation."""


@dataclass
class Reversal:
    earlier: Decision
    later: Decision


@dataclass
class Slip:
    commitment: Commitment

    @property
    def times(self) -> int:
        return self.commitment.times_slipped

    @property
    def original(self) -> date | None:
        history = self.commitment.deadline_history
        return history[0].was if history else None


@dataclass
class WeeklyReport:
    project: str
    since: date
    until: date

    reversed_decisions: list[Reversal] = field(default_factory=list)
    dropped: list[Commitment] = field(default_factory=list)
    delivered_quietly: list[Commitment] = field(default_factory=list)
    slipping: list[Slip] = field(default_factory=list)
    newly_promised: list[Commitment] = field(default_factory=list)
    overdue: list[Commitment] = field(default_factory=list)

    @property
    def is_quiet(self) -> bool:
        return not any([
            self.reversed_decisions, self.dropped, self.delivered_quietly,
            self.slipping, self.newly_promised, self.overdue,
        ])

    def as_markdown(self) -> str:
        lines = [
            f"# {self.project} - week of {self.since.isoformat()}",
            "",
            f"*{self.since.isoformat()} to {self.until.isoformat()}*",
            "",
        ]

        if self.is_quiet:
            lines += ["Nothing changed this week.", ""]
            return "\n".join(lines)

        if self.reversed_decisions:
            lines += ["## Reversed", ""]
            for item in self.reversed_decisions:
                lines.append(f"- **{item.later.statement}**")
                lines.append(f"  - reverses: {item.earlier.statement}")
            lines.append("")

        if self.dropped:
            lines += ["## Quietly dropped", ""]
            for item in self.dropped:
                owner = item.assignee.display_name or "unassigned"
                due = item.deadline.resolved
                lines.append(
                    f"- {item.description} - {owner}"
                    + (f", due {due.isoformat()}" if due else "")
                )
                for evidence in item.evidence[:1]:
                    lines.append(f"  - promised: {evidence.quote.strip()!r}")
            lines.append("")

        if self.delivered_quietly:
            lines += ["## Delivered without anyone saying so", ""]
            for item in self.delivered_quietly:
                owner = item.assignee.display_name or "unassigned"
                lines.append(f"- {item.description} - {owner}")
                if item.resolution_note:
                    lines.append(f"  - {item.resolution_note}")
            lines.append("")

        if self.slipping:
            lines += ["## Slipping", ""]
            for slip in self.slipping:
                current = slip.commitment.deadline.resolved
                origin = slip.original
                detail = f"now {current.isoformat()}" if current else "no date"
                if origin:
                    detail += f", originally {origin.isoformat()}"
                lines.append(
                    f"- {slip.commitment.description} - moved {slip.times}x ({detail})"
                )
            lines.append("")

        if self.overdue:
            lines += ["## Overdue", ""]
            for item in self.overdue:
                owner = item.assignee.display_name or "unassigned"
                late = (self.until - item.deadline.resolved).days
                lines.append(f"- {item.description} - {owner}, {late}d late")
            lines.append("")

        if self.newly_promised:
            lines += ["## Newly promised", ""]
            for item in self.newly_promised:
                owner = item.assignee.display_name or "unassigned"
                due = item.deadline.resolved
                when = due.isoformat() if due else "no date yet"
                lines.append(f"- {item.description} - {owner}, {when}")
            lines.append("")

        return "\n".join(lines)


def build_report(
    ledger: Ledger,
    project_name: str,
    until: date | None = None,
    days: int = DEFAULT_WINDOW_DAYS,
) -> WeeklyReport:
    """Everything that changed in the window. Reads only; changes nothing."""
    until = until or date.today()
    since = until - timedelta(days=days)
    report = WeeklyReport(project=project_name, since=since, until=until)

    meeting_dates = ledger.meeting_dates
    in_window = {
        meeting_id for meeting_id, when in meeting_dates.items()
        if since <= when <= until
    }

    for earlier, later in ledger.contradictions():
        # Dated by the meeting that produced the reversal, not by when the
        # original decision was taken - the change is the event.
        if later.meeting_id in in_window or not meeting_dates:
            report.reversed_decisions.append(Reversal(earlier, later))

    for commitment in ledger.commitments:
        due = commitment.deadline.resolved

        if commitment.status is CommitmentStatus.DROPPED:
            report.dropped.append(commitment)
            continue

        if commitment.status is CommitmentStatus.VERIFIED_DONE:
            # "Quietly" means the ledger learned it from evidence rather than
            # from someone saying so in a meeting. That distinction is the whole
            # justification for the external-evidence layer.
            note = (commitment.resolution_note or "").lower()
            if _closed_in_window(commitment, since, until) and _from_evidence(note):
                report.delivered_quietly.append(commitment)
            continue

        if commitment.times_slipped >= 2:
            report.slipping.append(Slip(commitment))

        if commitment.status in (CommitmentStatus.OPEN, CommitmentStatus.OVERDUE):
            if due and due < until:
                overdue_days = (until - due).days
                if overdue_days >= QUIET_DAYS and _unmentioned(commitment, since):
                    # Long overdue, never raised again, no evidence. Reported as
                    # dropped even though nothing formally marked it so - that
                    # silence is exactly what the section is for.
                    report.dropped.append(commitment)
                else:
                    report.overdue.append(commitment)

        if commitment.meeting_id in in_window and commitment.is_actionable:
            report.newly_promised.append(commitment)

    report.overdue.sort(key=lambda c: c.deadline.resolved or until)
    report.slipping.sort(key=lambda s: -s.times)
    return report


def _closed_in_window(commitment: Commitment, since: date, until: date) -> bool:
    when = commitment.last_action_on
    return when is None or since <= when <= until


def _from_evidence(note: str) -> bool:
    """Whether the close came from outside the conversation."""
    return any(marker in note for marker in ("github", "gmail", "pr #", "verified"))


def _unmentioned(commitment: Commitment, since: date) -> bool:
    """No status update and no chase touched it during the window."""
    return commitment.last_action_on is None or commitment.last_action_on < since
