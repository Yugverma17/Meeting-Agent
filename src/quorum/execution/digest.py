"""Per-person digests.

One digest per owner, containing only *their* commitments. A single email listing
everyone's tasks is what most tools send, and it is why most tools get filtered
to a folder nobody opens - a wall of other people's work makes your own items
invisible.

Every line cites the meeting and the words that created the obligation, so the
recipient can check the claim rather than take it on trust. That is the same
principle the verifier enforces internally, carried through to the human.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from quorum.models import Commitment
from quorum.tracking.ledger import Ledger
from quorum.tracking.planner import ActionType, PlannedAction

TONE = {
    ActionType.REMIND: "Coming up",
    ActionType.NUDGE: "Overdue",
    ActionType.ESCALATE: "Needs attention",
}


@dataclass
class DigestLine:
    commitment: Commitment
    action: PlannedAction

    def render(self, ledger: Ledger) -> str:
        due = self.commitment.deadline.resolved
        heading = TONE.get(self.action.action, "Update")
        parts = [f"  [{heading}] {self.commitment.description}"]
        if due:
            parts.append(f"    due {due.isoformat()}" + (
                f" ({self.action.days_overdue} days ago)" if self.action.days_overdue else ""
            ))
        for evidence in self.commitment.evidence[:1]:
            when = ledger.meeting_dates.get(self.commitment.meeting_id)
            stamp = f" on {when.isoformat()}" if when else ""
            parts.append(f'    you said{stamp}: "{evidence.quote.strip()}"')
        return "\n".join(parts)


@dataclass
class Digest:
    recipient_email: str
    recipient_name: str
    lines: list[DigestLine] = field(default_factory=list)

    @property
    def subject(self) -> str:
        overdue = sum(1 for line in self.lines if line.action.days_overdue > 0)
        if overdue:
            return f"{overdue} overdue commitment(s) and {len(self.lines) - overdue} upcoming"
        return f"{len(self.lines)} upcoming commitment(s)"

    def render(self, ledger: Ledger, today: date) -> str:
        header = (
            f"Hi {self.recipient_name.split()[0]},\n\n"
            "From your recent meetings, these are still open:\n"
        )
        body = "\n\n".join(line.render(ledger) for line in self.lines)
        footer = (
            "\n\nIf any of these are already done, reply and they'll be closed.\n"
            f"Generated {today.isoformat()} by Quorum. Nothing here was sent without "
            "a human approving it first."
        )
        return header + "\n" + body + footer


def build_digests(
    actions: list[PlannedAction], ledger: Ledger, today: date
) -> list[Digest]:
    """Group outbound actions by owner, one digest each.

    Escalations are excluded: they go to a different audience and putting one in
    the owner's own digest tells them they are being escalated, which is neither
    kind nor useful.
    """
    grouped: dict[str, Digest] = {}

    # An explicit allow-list rather than "outbound minus escalations". Outbound
    # now also covers calendar writes, which have no recipient and would other-
    # wise fall through to a lookup that happens to return nothing.
    for action in actions:
        if action.action not in (ActionType.REMIND, ActionType.NUDGE):
            continue
        commitment = ledger.by_id(action.commitment_id)
        if commitment is None or not commitment.assignee.email:
            continue

        digest = grouped.setdefault(
            commitment.assignee.email,
            Digest(
                recipient_email=commitment.assignee.email,
                recipient_name=commitment.assignee.display_name or "there",
            ),
        )
        digest.lines.append(DigestLine(commitment, action))

    for digest in grouped.values():
        # Most urgent first - a digest read only to the third line should still
        # surface the thing that is most late.
        digest.lines.sort(key=lambda line: -line.action.days_overdue)

    return list(grouped.values())
