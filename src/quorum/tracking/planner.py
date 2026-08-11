"""The between-meetings planner - what makes this an agent, not a pipeline.

Everything before this stage runs once per meeting and stops. This runs every
day with a standing goal - *every open commitment reaches a resolution* - and
has to decide, per item, with incomplete information, what to do about it:
chase it, escalate it, close it, warn about a knock-on slip, or admit it was
quietly abandoned.

The rules are deliberately deterministic. Three reasons:

1. **Auditable.** "Why did it email my manager?" has an exact answer, which is
   the difference between a tool people tolerate and one they switch off.
2. **Free.** A daily sweep over hundreds of commitments would otherwise burn the
   entire free-tier budget on judgement calls that a date comparison settles.
3. **Testable.** Escalation timing can be scored against ground truth without
   sampling a model's mood.

The single most important behaviour is what happens *before* nagging: the
planner asks an evidence provider whether the work actually happened. Chasing
someone who already delivered is the error that gets a tool uninstalled, so
external evidence is consulted first and outranks silence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum
from typing import Protocol, runtime_checkable

from quorum.models import Commitment, CommitmentStatus
from quorum.tracking.ledger import Ledger

log = logging.getLogger(__name__)


class ActionType(str, Enum):
    WAIT = "wait"
    """Not due yet, nothing to do. The most common correct answer."""

    REMIND = "remind"
    """Deadline approaching. A gentle heads-up before it is late."""

    NUDGE = "nudge"
    """Overdue with no sign of progress."""

    ESCALATE = "escalate"
    """Repeatedly nudged and still silent - raise it beyond the owner."""

    MARK_DONE = "mark_done"
    """External evidence shows it was delivered."""

    MARK_DROPPED = "mark_dropped"
    """Long overdue, never mentioned again, no evidence. Abandoned."""

    FLAG_CONFLICT = "flag_conflict"
    """Something is wrong with the commitment itself - no deadline, or a date
    past the project end."""

    PROPAGATE_SLIP = "propagate_slip"
    """An upstream commitment slipped, so this one is at risk through no fault
    of its owner. Nagging them would be unfair and useless."""


@dataclass
class DeliveryEvidence:
    """Proof from outside the meeting that work actually happened."""

    source: str  # "github" | "gmail"
    reference: str  # "PR #204", message id
    found_on: date
    confidence: float = 1.0
    detail: str = ""


@runtime_checkable
class EvidenceProvider(Protocol):
    def find_evidence(self, commitment: Commitment) -> DeliveryEvidence | None: ...


@dataclass
class PlannedAction:
    commitment_id: str
    action: ActionType
    reason: str
    """Human-readable justification. Surfaced in the UI and used in eval
    failure analysis - an action nobody can explain is a bug."""

    target_email: str | None = None
    priority: int = 0
    evidence: DeliveryEvidence | None = None
    days_overdue: int = 0

    @property
    def is_outbound(self) -> bool:
        """Whether performing this would contact a human. These are the only
        actions that require approval."""
        return self.action in (ActionType.REMIND, ActionType.NUDGE, ActionType.ESCALATE)

    def as_dict(self) -> dict:
        return {
            "commitment_id": self.commitment_id,
            "action": self.action.value,
            "reason": self.reason,
            "target_email": self.target_email,
            "priority": self.priority,
            "days_overdue": self.days_overdue,
            "evidence": self.evidence.reference if self.evidence else None,
        }


@dataclass
class PlannerConfig:
    remind_lead_days: int = 2
    """How far ahead of a deadline to send a heads-up."""

    nudge_interval_days: int = 3
    """Minimum gap between nudges for the same commitment. Without this the
    daily sweep would nag every single day, which is worse than not nagging."""

    escalate_after_nudges: int = 2
    escalate_after_days: int = 7
    drop_after_days: int = 21
    """Overdue for three weeks with no evidence and no mention: abandoned.
    Deliberately generous - declaring something dropped while someone is still
    working on it is insulting and wrong."""

    project_end: date | None = None


@dataclass
class PlanReport:
    actions: list[PlannedAction] = field(default_factory=list)
    evidence_checks: int = 0
    evidence_found: int = 0

    def of_type(self, action: ActionType) -> list[PlannedAction]:
        return [a for a in self.actions if a.action is action]

    @property
    def outbound(self) -> list[PlannedAction]:
        return [a for a in self.actions if a.is_outbound]

    def as_dict(self) -> dict:
        counts: dict[str, int] = {}
        for action in self.actions:
            counts[action.action.value] = counts.get(action.action.value, 0) + 1
        return {
            "total": len(self.actions),
            "by_action": counts,
            "outbound": len(self.outbound),
            "evidence_checks": self.evidence_checks,
            "evidence_found": self.evidence_found,
        }


class Planner:
    def __init__(self, config: PlannerConfig | None = None) -> None:
        self.config = config or PlannerConfig()

    def plan(
        self,
        ledger: Ledger,
        today: date,
        evidence: EvidenceProvider | None = None,
    ) -> PlanReport:
        """Decide what to do about every open commitment, as of `today`."""
        report = PlanReport()
        for commitment in ledger.open_commitments():
            action = self._decide(commitment, ledger, today, evidence, report)
            report.actions.append(action)

        report.actions.sort(key=lambda a: -a.priority)
        return report

    # -- the decision ------------------------------------------------------

    def _decide(
        self,
        commitment: Commitment,
        ledger: Ledger,
        today: date,
        evidence: EvidenceProvider | None,
        report: PlanReport,
    ) -> PlannedAction:
        cfg = self.config
        email = commitment.assignee.email

        # 1. Did it actually happen? Asked before anything else, because every
        #    other branch risks chasing someone who already delivered.
        if evidence is not None:
            report.evidence_checks += 1
            found = evidence.find_evidence(commitment)
            if found is not None:
                report.evidence_found += 1
                commitment.status = CommitmentStatus.VERIFIED_DONE
                commitment.resolution_note = f"{found.source}: {found.reference}"
                return PlannedAction(
                    commitment.id, ActionType.MARK_DONE,
                    f"Verified delivered via {found.source} ({found.reference})",
                    target_email=email, evidence=found, priority=1,
                )

        # 2. Someone reported it done. No external proof yet, but chasing a
        #    person who just told the room they delivered is the single fastest
        #    way to get the tool switched off.
        if commitment.status is CommitmentStatus.CLAIMED_DONE:
            return PlannedAction(
                commitment.id, ActionType.WAIT,
                "Reported complete in a meeting; awaiting external confirmation",
                target_email=email, priority=2,
            )

        # 3. A commitment with no date cannot be chased on time.
        due = commitment.deadline.resolved
        if due is None:
            return PlannedAction(
                commitment.id, ActionType.FLAG_CONFLICT,
                "No resolvable deadline - needs a date before it can be tracked",
                target_email=email, priority=4,
            )

        if cfg.project_end and due > cfg.project_end:
            return PlannedAction(
                commitment.id, ActionType.FLAG_CONFLICT,
                f"Due {due}, after the project ends {cfg.project_end}",
                target_email=email, priority=6,
            )

        # 4. Blocked upstream: the owner is not at fault and nagging is useless.
        blocker = self._blocking_upstream(commitment, ledger, today)
        if blocker is not None:
            commitment.status = CommitmentStatus.BLOCKED
            return PlannedAction(
                commitment.id, ActionType.PROPAGATE_SLIP,
                f"Blocked by {blocker.description!r}, itself overdue since "
                f"{blocker.deadline.resolved}",
                target_email=email, priority=7,
            )

        # Reported blocked without a resolvable upstream. Still not the owner's
        # fault, so it must not turn into a nudge.
        if commitment.status is CommitmentStatus.BLOCKED:
            return PlannedAction(
                commitment.id, ActionType.PROPAGATE_SLIP,
                "Reported blocked; upstream dependency not yet identified",
                target_email=email, priority=7,
            )

        overdue_days = (today - due).days

        # 4. Not due yet.
        if overdue_days < -cfg.remind_lead_days:
            return PlannedAction(
                commitment.id, ActionType.WAIT, f"Not due until {due}", target_email=email
            )

        if overdue_days < 0:
            return PlannedAction(
                commitment.id, ActionType.REMIND,
                f"Due in {-overdue_days} day(s) on {due}",
                target_email=email, priority=3, days_overdue=0,
            )

        # 5. Overdue. Abandoned, escalate, or nudge - in that order.
        if overdue_days >= cfg.drop_after_days:
            commitment.status = CommitmentStatus.DROPPED
            return PlannedAction(
                commitment.id, ActionType.MARK_DROPPED,
                f"Overdue {overdue_days} days with no evidence and no mention since",
                target_email=email, priority=5, days_overdue=overdue_days,
            )

        if (
            commitment.nudge_count >= cfg.escalate_after_nudges
            and overdue_days >= cfg.escalate_after_days
        ):
            return PlannedAction(
                commitment.id, ActionType.ESCALATE,
                f"Nudged {commitment.nudge_count} times, still overdue by {overdue_days} days",
                target_email=email, priority=9, days_overdue=overdue_days,
            )

        # Respect the quiet period, or the daily sweep becomes daily nagging.
        if commitment.last_action_on is not None:
            since = (today - commitment.last_action_on).days
            if since < cfg.nudge_interval_days:
                return PlannedAction(
                    commitment.id, ActionType.WAIT,
                    f"Nudged {since} day(s) ago; waiting {cfg.nudge_interval_days}",
                    target_email=email, days_overdue=overdue_days,
                )

        commitment.status = CommitmentStatus.OVERDUE
        return PlannedAction(
            commitment.id, ActionType.NUDGE,
            f"Overdue by {overdue_days} day(s), no sign of progress",
            target_email=email, priority=8, days_overdue=overdue_days,
        )

    def _blocking_upstream(
        self, commitment: Commitment, ledger: Ledger, today: date
    ) -> Commitment | None:
        for upstream_id in commitment.depends_on:
            upstream = ledger.by_id(upstream_id)
            if upstream is None or upstream.status in (
                CommitmentStatus.VERIFIED_DONE,
                CommitmentStatus.CANCELLED,
            ):
                continue
            if upstream.deadline.resolved and upstream.deadline.resolved < today:
                return upstream
        return None

    # -- applying ----------------------------------------------------------

    @staticmethod
    def apply(action: PlannedAction, commitment: Commitment, today: date) -> None:
        """Record that an outbound action was actually performed.

        Called only after a human approves. Keeping this separate from `plan`
        is what lets the eval harness score proposed actions offline without
        mutating nudge counts.
        """
        if action.is_outbound:
            commitment.nudge_count += 1
            commitment.last_action_on = today
