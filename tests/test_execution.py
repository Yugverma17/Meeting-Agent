from __future__ import annotations

from datetime import date

import pytest

from quorum.execution import ApprovalGate, ApprovalStatus, build_digests
from quorum.execution.approval import DryRunTransport, NotApproved
from quorum.models import Assignee, Commitment, Deadline, Evidence
from quorum.tracking import ActionType, Ledger
from quorum.tracking.planner import PlannedAction
from quorum.verify.github import keywords_for

TODAY = date(2026, 3, 16)


def action(kind=ActionType.NUDGE, commitment_id="c1", overdue=3) -> PlannedAction:
    return PlannedAction(
        commitment_id, kind, "overdue", target_email="yug@example.com", days_overdue=overdue
    )


# --- the gate cannot be bypassed -------------------------------------------


def test_execution_without_approval_is_refused():
    gate = ApprovalGate()
    pending = gate.propose(action(), "Nudge Yug about the spec")

    with pytest.raises(NotApproved):
        gate.execute(pending.id, "made-up-token", DryRunTransport())


def test_approved_action_executes():
    gate = ApprovalGate()
    transport = DryRunTransport()
    pending = gate.propose(action(), "Nudge Yug about the spec")

    token = gate.approve(pending.id)
    assert gate.execute(pending.id, token, transport)
    assert len(transport.sent) == 1
    assert pending.status is ApprovalStatus.EXECUTED


def test_token_is_single_use():
    """A replayed approval must not send a second time."""
    gate = ApprovalGate()
    pending = gate.propose(action(), "Nudge")
    token = gate.approve(pending.id)
    gate.execute(pending.id, token, DryRunTransport())

    with pytest.raises(KeyError):
        gate.execute(pending.id, token, DryRunTransport())


def test_a_token_does_not_authorise_a_different_action():
    """Approval is per-action and never generalises."""
    gate = ApprovalGate()
    first = gate.propose(action(commitment_id="c1"), "Nudge about c1")
    second = gate.propose(action(commitment_id="c2"), "Nudge about c2")
    token = gate.approve(first.id)

    with pytest.raises(NotApproved):
        gate.execute(second.id, token, DryRunTransport())


def test_rejected_action_cannot_be_approved_afterwards():
    gate = ApprovalGate()
    pending = gate.propose(action(), "Nudge")
    gate.reject(pending.id, "not needed, already done")

    with pytest.raises(KeyError):
        gate.approve(pending.id)
    assert pending.status is ApprovalStatus.REJECTED
    assert pending.rejection_note == "not needed, already done"


def test_double_approval_is_refused():
    gate = ApprovalGate()
    pending = gate.propose(action(), "Nudge")
    gate.approve(pending.id)
    with pytest.raises(NotApproved):
        gate.approve(pending.id)


def test_only_outbound_actions_are_proposed():
    gate = ApprovalGate()
    actions = [
        action(ActionType.NUDGE),
        action(ActionType.WAIT),
        action(ActionType.MARK_DONE),
        action(ActionType.REMIND),
    ]
    proposed = gate.propose_all(actions, lambda a: (f"{a.action.value}", ""))
    assert len(proposed) == 2


def test_summary_counts_by_status():
    gate = ApprovalGate()
    a = gate.propose(action(), "one")
    gate.propose(action(), "two")
    gate.reject(a.id)

    summary = gate.summary()
    assert summary["pending"] == 1
    assert summary["by_status"]["rejected"] == 1


# --- digests ----------------------------------------------------------------


def ledger_with(*commitments) -> Ledger:
    ledger = Ledger("proj")
    ledger.commitments.extend(commitments)
    ledger.meeting_dates["m1"] = date(2026, 3, 9)
    return ledger


def commitment(description, email="yug@example.com", name="Yug Verma") -> Commitment:
    return Commitment(
        description=description, meeting_id="m1",
        assignee=Assignee(speaker_id="spk_yug", email=email, display_name=name, confidence=0.9),
        deadline=Deadline(resolved=date(2026, 3, 13)),
        evidence=[Evidence(utterance_id="u1", quote="I'll have the spec to you by Friday")],
    )


def test_each_person_gets_only_their_own_items():
    """A wall of other people's tasks is why these emails get filtered away."""
    mine = commitment("the spec")
    theirs = commitment("the migration", email="sam@example.com", name="Sam Okafor")
    theirs.assignee.speaker_id = "spk_sam"
    ledger = ledger_with(mine, theirs)

    digests = build_digests(
        [action(commitment_id=mine.id), action(commitment_id=theirs.id)], ledger, TODAY
    )

    assert len(digests) == 2
    for digest in digests:
        assert len(digest.lines) == 1


def test_digest_quotes_the_words_that_created_the_obligation():
    item = commitment("the spec")
    ledger = ledger_with(item)
    digest = build_digests([action(commitment_id=item.id)], ledger, TODAY)[0]
    body = digest.render(ledger, TODAY)

    assert "I'll have the spec to you by Friday" in body
    assert "2026-03-09" in body, "the meeting date should be cited"
    assert "Yug" in body


def test_escalations_stay_out_of_the_owners_digest():
    """Telling someone they are being escalated, inside their own reminder, is
    neither kind nor useful."""
    item = commitment("the spec")
    ledger = ledger_with(item)
    digests = build_digests(
        [action(ActionType.ESCALATE, commitment_id=item.id)], ledger, TODAY
    )
    assert digests == []


def test_most_overdue_appears_first():
    a = commitment("slightly late")
    b = commitment("very late")
    ledger = ledger_with(a, b)

    digest = build_digests(
        [
            action(commitment_id=a.id, overdue=1),
            action(commitment_id=b.id, overdue=20),
        ],
        ledger, TODAY,
    )[0]
    assert digest.lines[0].commitment.description == "very late"


def test_subject_distinguishes_overdue_from_upcoming():
    item = commitment("the spec")
    ledger = ledger_with(item)

    overdue = build_digests([action(commitment_id=item.id, overdue=4)], ledger, TODAY)[0]
    assert "overdue" in overdue.subject

    upcoming = build_digests(
        [action(ActionType.REMIND, commitment_id=item.id, overdue=0)], ledger, TODAY
    )[0]
    assert "upcoming" in upcoming.subject


def test_owner_without_an_email_is_skipped():
    item = commitment("the spec", email=None)
    item.assignee.email = None
    ledger = ledger_with(item)
    assert build_digests([action(commitment_id=item.id)], ledger, TODAY) == []


# --- github keyword extraction ---------------------------------------------


def test_keywords_drop_filler_and_prefer_distinctive_words():
    keywords = keywords_for("the billing reconciliation job")
    assert "reconciliation" in keywords
    assert "the" not in keywords and "job" not in keywords
    assert keywords[0] == "reconciliation", "longest/most distinctive first"


def test_keywords_are_deduplicated_and_capped():
    keywords = keywords_for("the migration migration migration of the auth service database")
    assert len(keywords) <= 4
    assert len(set(keywords)) == len(keywords)


def test_vague_description_yields_too_few_keywords_to_search():
    """Searching on one generic word would match half a repository and close
    the wrong commitment."""
    assert len(keywords_for("do the thing")) < 2
