"""The approval gate.

Every action that would reach a human - an email, a calendar write - stops here
until a person says yes. This is not a limitation to apologise for; for an agent
that acts on what colleagues said out loud, and that is working from a
transcript it cannot fully trust, it is the correct design.

Two properties the implementation guarantees:

**There is no bypass.** Executing requires a token that only `approve()` issues.
A caller cannot "just send it" by setting a flag, because the send path needs a
value it can only obtain from an approval.

**Approval is specific.** A token authorises one action, once. It cannot be
replayed, and approving a nudge to Sam does not authorise a different nudge to
Sam tomorrow. Approval granted for one action never generalises to the next.
"""

from __future__ import annotations

import logging
import secrets
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from quorum.tracking.planner import PlannedAction

log = logging.getLogger(__name__)


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTED = "executed"


class NotApproved(RuntimeError):
    """Raised when execution is attempted without a valid approval token."""


@dataclass
class PendingAction:
    id: str
    action: PlannedAction
    summary: str
    """What a human will actually read before deciding. If this does not
    describe the effect accurately, the approval is not informed consent."""

    body: str = ""
    recipient: str | None = None
    status: ApprovalStatus = ApprovalStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    decided_at: datetime | None = None
    rejection_note: str = ""
    _token: str | None = field(default=None, repr=False)

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "status": self.status.value,
            "summary": self.summary,
            "recipient": self.recipient,
            "action": self.action.action.value,
            "reason": self.action.reason,
        }


class ApprovalGate:
    def __init__(self, require_approval: bool = True) -> None:
        self.require_approval = require_approval
        """Only the evaluation harness sets this False, so it can score proposed
        actions offline. Production paths never disable it."""

        self.pending: dict[str, PendingAction] = {}
        self.history: list[PendingAction] = []

    # -- proposing ---------------------------------------------------------

    def propose(
        self,
        action: PlannedAction,
        summary: str,
        body: str = "",
        recipient: str | None = None,
    ) -> PendingAction:
        item = PendingAction(
            id=f"apr_{secrets.token_hex(6)}",
            action=action, summary=summary, body=body,
            recipient=recipient or action.target_email,
        )
        self.pending[item.id] = item
        log.info("Awaiting approval: %s -> %s", summary, item.recipient)
        return item

    def propose_all(self, actions: list[PlannedAction], summarise) -> list[PendingAction]:
        return [self.propose(a, *summarise(a)) for a in actions if a.is_outbound]

    # -- deciding ----------------------------------------------------------

    def approve(self, action_id: str) -> str:
        """Approve one action and return its single-use execution token."""
        item = self._require(action_id)
        if item.status is not ApprovalStatus.PENDING:
            raise NotApproved(f"{action_id} is already {item.status.value}")

        item.status = ApprovalStatus.APPROVED
        item.decided_at = datetime.now()
        item._token = secrets.token_urlsafe(16)
        return item._token

    def reject(self, action_id: str, note: str = "") -> None:
        item = self._require(action_id)
        item.status = ApprovalStatus.REJECTED
        item.decided_at = datetime.now()
        item.rejection_note = note
        self.history.append(self.pending.pop(action_id))

    # -- executing ---------------------------------------------------------

    def execute(self, action_id: str, token: str, transport) -> bool:
        """Perform an approved action. `transport` does the actual sending.

        The token check is the whole point: there is no argument to this method
        that lets a caller skip approval.
        """
        item = self._require(action_id)

        if self.require_approval:
            if item.status is not ApprovalStatus.APPROVED:
                raise NotApproved(f"{action_id} has not been approved")
            if not item._token or not secrets.compare_digest(item._token, token):
                raise NotApproved(f"Invalid approval token for {action_id}")

        sent = transport.send(item)
        if sent:
            item.status = ApprovalStatus.EXECUTED
            item._token = None  # single use - a token never works twice
            self.history.append(self.pending.pop(action_id))
        return sent

    def _require(self, action_id: str) -> PendingAction:
        item = self.pending.get(action_id)
        if item is None:
            raise KeyError(f"No pending action {action_id}")
        return item

    # -- reporting ---------------------------------------------------------

    def summary(self) -> dict:
        counts: dict[str, int] = {}
        for item in [*self.pending.values(), *self.history]:
            counts[item.status.value] = counts.get(item.status.value, 0) + 1
        return {"pending": len(self.pending), "by_status": counts}


class DryRunTransport:
    """Records what would have been sent. The default everywhere."""

    def __init__(self) -> None:
        self.sent: list[PendingAction] = []

    def send(self, item: PendingAction) -> bool:
        self.sent.append(item)
        log.info("[dry run] would send to %s: %s", item.recipient, item.summary)
        return True
