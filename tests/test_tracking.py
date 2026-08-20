from __future__ import annotations

from datetime import date, timedelta

import pytest

from quorum.models import (
    Assignee,
    Commitment,
    CommitmentStatus,
    CommitmentStrength,
    Deadline,
    DeadlineResolution,
    Decision,
    Evidence,
    MeetingRecord,
)
from quorum.tracking import ActionType, Ledger, MergeOutcome, Planner, PlannerConfig
from quorum.tracking.planner import DeliveryEvidence

TODAY = date(2026, 3, 16)


def commit(
    description: str,
    *,
    owner: str = "spk_yug",
    email: str = "yug@example.com",
    due: date | None = date(2026, 3, 13),
    strength: CommitmentStrength = CommitmentStrength.FIRM,
    **kw,
) -> Commitment:
    return Commitment(
        description=description,
        assignee=Assignee(speaker_id=owner, email=email, display_name="Yug Verma", confidence=0.9),
        deadline=Deadline(resolved=due, method=DeadlineResolution.RELATIVE, confidence=0.9),
        strength=strength,
        evidence=[Evidence(utterance_id="utt_1", quote="a sufficiently long quote")],
        **kw,
    )


class StubEvidence:
    def __init__(self, hits: dict[str, str]) -> None:
        self.hits = hits
        self.calls = 0

    def find_evidence(self, commitment):
        self.calls += 1
        reference = self.hits.get(commitment.description)
        if reference is None:
            return None
        return DeliveryEvidence("github", reference, TODAY)


# --- ledger merging --------------------------------------------------------


def test_new_commitment_is_added():
    ledger = Ledger("proj")
    assert ledger.merge(commit("the ingestion API spec")) is MergeOutcome.ADDED
    assert len(ledger.commitments) == 1


def test_restated_commitment_updates_rather_than_duplicating():
    """"I didn't get to the spec, I'll have it Friday" is the same obligation
    with a new date. A duplicate here means two nags for one piece of work."""
    ledger = Ledger("proj")
    ledger.merge(commit("the ingestion API spec", due=date(2026, 3, 13)))
    outcome = ledger.merge(commit("ingestion API spec document", due=date(2026, 3, 20)))

    assert outcome is MergeOutcome.UPDATED
    assert len(ledger.commitments) == 1
    assert ledger.commitments[0].deadline.resolved == date(2026, 3, 20)


def test_different_work_by_the_same_person_stays_separate():
    ledger = Ledger("proj")
    ledger.merge(commit("the auth migration"))
    ledger.merge(commit("the billing reconciliation job"))
    assert len(ledger.commitments) == 2


def test_same_description_different_owner_stays_separate():
    ledger = Ledger("proj")
    ledger.merge(commit("the auth migration", owner="spk_yug"))
    ledger.merge(commit("the auth migration", owner="spk_sam", email="sam@example.com"))
    assert len(ledger.commitments) == 2


def test_musings_and_unowned_items_never_enter_the_ledger():
    ledger = Ledger("proj")
    assert ledger.merge(commit("x", strength=CommitmentStrength.MUSING)) is MergeOutcome.IGNORED

    unowned = commit("y")
    unowned.assignee = Assignee(raw_mention="someone")
    assert ledger.merge(unowned) is MergeOutcome.IGNORED
    assert ledger.commitments == []


def test_merge_does_not_move_a_deadline_backwards():
    ledger = Ledger("proj")
    ledger.merge(commit("the spec", due=date(2026, 3, 20)))
    ledger.merge(commit("the spec", due=date(2026, 3, 13)))
    assert ledger.commitments[0].deadline.resolved == date(2026, 3, 20)


def test_ingest_reports_counts_and_records_the_meeting():
    ledger = Ledger("proj")
    record = MeetingRecord(
        meeting_id="m1", meeting_date=date(2026, 3, 9),
        commitments=[commit("the spec"), commit("x", strength=CommitmentStrength.MUSING)],
        decisions=[Decision(statement="use Postgres",
                            evidence=[Evidence(utterance_id="u", quote="long enough quote")])],
    )
    stats = ledger.ingest(record)

    assert stats.added == 1 and stats.ignored == 1
    assert ledger.meeting_dates["m1"] == date(2026, 3, 9)
    assert len(ledger.decisions) == 1


# --- ledger queries --------------------------------------------------------


def test_open_excludes_closed_states():
    ledger = Ledger("proj")
    for status in (
        CommitmentStatus.VERIFIED_DONE, CommitmentStatus.CANCELLED, CommitmentStatus.DROPPED
    ):
        item = commit(f"work {status.value}")
        item.status = status
        ledger.commitments.append(item)
    ledger.commitments.append(commit("still open"))

    assert [c.description for c in ledger.open_commitments()] == ["still open"]


def test_overdue_uses_the_supplied_date_not_today():
    ledger = Ledger("proj")
    ledger.merge(commit("late one", due=date(2026, 3, 10)))
    ledger.merge(commit("future one", due=date(2026, 4, 10)))

    assert [c.description for c in ledger.overdue(TODAY)] == ["late one"]


def test_contradictions_pair_reversed_decisions():
    ledger = Ledger("proj")
    first = Decision(statement="use Mongo",
                     evidence=[Evidence(utterance_id="u", quote="long enough quote")])
    second = Decision(statement="use Postgres", reverses=first.id,
                      evidence=[Evidence(utterance_id="u", quote="long enough quote")])
    ledger.decisions = [first, second]

    pairs = ledger.contradictions()
    assert len(pairs) == 1 and pairs[0][0].statement == "use Mongo"


def test_ledger_round_trips_through_disk(tmp_path):
    ledger = Ledger("proj")
    ledger.merge(commit("the spec"))
    ledger.meeting_dates["m1"] = date(2026, 3, 9)
    path = tmp_path / "ledger.json"
    ledger.save(path)

    restored = Ledger.load(path)
    assert restored.project_id == "proj"
    assert restored.commitments[0].description == "the spec"
    assert restored.meeting_dates["m1"] == date(2026, 3, 9)


# --- planner: the core decisions -------------------------------------------


def plan_one(commitment, today=TODAY, evidence=None, config=None):
    ledger = Ledger("proj")
    ledger.commitments.append(commitment)
    report = Planner(config or PlannerConfig()).plan(ledger, today, evidence)
    return report.actions[0], report


def test_not_yet_due_waits():
    action, _ = plan_one(commit("x", due=date(2026, 4, 1)))
    assert action.action is ActionType.WAIT
    assert not action.is_outbound


def test_approaching_deadline_reminds():
    action, _ = plan_one(commit("x", due=date(2026, 3, 17)))
    assert action.action is ActionType.REMIND


def test_overdue_nudges():
    action, _ = plan_one(commit("x", due=date(2026, 3, 13)))
    assert action.action is ActionType.NUDGE
    assert action.days_overdue == 3
    assert action.target_email == "yug@example.com"


def test_evidence_closes_the_commitment_without_nagging():
    """The behaviour that justifies the reality-verification layer: overdue and
    silent, but actually delivered."""
    item = commit("the auth migration", due=date(2026, 3, 10))
    action, report = plan_one(item, evidence=StubEvidence({"the auth migration": "PR #204"}))

    assert action.action is ActionType.MARK_DONE
    assert not action.is_outbound, "a delivered commitment must never be chased"
    assert item.status is CommitmentStatus.VERIFIED_DONE
    assert report.evidence_found == 1


def test_evidence_is_checked_before_any_nudge_decision():
    item = commit("x", due=date(2026, 1, 1))  # very overdue
    action, _ = plan_one(item, evidence=StubEvidence({"x": "PR #1"}))
    assert action.action is ActionType.MARK_DONE, "evidence must outrank silence"


def test_repeated_nudges_escalate():
    item = commit("x", due=date(2026, 3, 5))
    item.nudge_count = 2
    action, _ = plan_one(item)

    assert action.action is ActionType.ESCALATE
    assert action.priority > 0


def test_quiet_period_prevents_daily_nagging():
    """A daily sweep must not send a daily email about the same thing."""
    item = commit("x", due=date(2026, 3, 10))
    item.last_action_on = TODAY - timedelta(days=1)
    action, _ = plan_one(item)

    assert action.action is ActionType.WAIT
    assert "waiting" in action.reason


def test_quiet_period_expires():
    item = commit("x", due=date(2026, 3, 10))
    item.last_action_on = TODAY - timedelta(days=5)
    action, _ = plan_one(item)
    assert action.action is ActionType.NUDGE


def test_long_overdue_and_silent_is_marked_dropped():
    item = commit("x", due=date(2026, 2, 1))
    action, _ = plan_one(item)

    assert action.action is ActionType.MARK_DROPPED
    assert item.status is CommitmentStatus.DROPPED
    assert not action.is_outbound


def test_missing_deadline_is_flagged_not_guessed():
    action, _ = plan_one(commit("x", due=None))
    assert action.action is ActionType.FLAG_CONFLICT
    assert "No resolvable deadline" in action.reason


def test_deadline_after_project_end_is_flagged():
    config = PlannerConfig(project_end=date(2026, 3, 31))
    action, _ = plan_one(commit("x", due=date(2026, 5, 1)), config=config)
    assert action.action is ActionType.FLAG_CONFLICT
    assert "after the project ends" in action.reason


# --- dependency propagation -------------------------------------------------


def test_blocked_by_overdue_upstream_propagates_instead_of_nagging():
    """The downstream owner is not at fault; chasing them is both unfair and
    useless."""
    ledger = Ledger("proj")
    upstream = commit("the schema migration", due=date(2026, 3, 2))
    downstream = commit("the reporting view", due=date(2026, 3, 13))
    downstream.depends_on = [upstream.id]
    ledger.commitments.extend([upstream, downstream])

    report = Planner().plan(ledger, TODAY)
    action = next(a for a in report.actions if a.commitment_id == downstream.id)

    assert action.action is ActionType.PROPAGATE_SLIP
    assert "the schema migration" in action.reason
    assert downstream.status is CommitmentStatus.BLOCKED


def test_completed_upstream_does_not_block():
    ledger = Ledger("proj")
    upstream = commit("the schema migration", due=date(2026, 3, 2))
    upstream.status = CommitmentStatus.VERIFIED_DONE
    downstream = commit("the reporting view", due=date(2026, 3, 13))
    downstream.depends_on = [upstream.id]
    ledger.commitments.append(downstream)
    ledger.commitments.append(upstream)

    report = Planner().plan(ledger, TODAY)
    action = next(a for a in report.actions if a.commitment_id == downstream.id)
    assert action.action is ActionType.NUDGE


# --- reporting and application ---------------------------------------------


def test_actions_are_ordered_by_priority():
    ledger = Ledger("proj")
    ledger.commitments.append(commit("waiting", due=date(2026, 5, 1)))
    escalating = commit("escalating", due=date(2026, 3, 5))
    escalating.nudge_count = 3
    ledger.commitments.append(escalating)

    report = Planner().plan(ledger, TODAY)
    assert report.actions[0].action is ActionType.ESCALATE


def test_only_side_effecting_actions_are_outbound():
    """Every action that leaves the process needs approval; nothing else does.

    SCHEDULE writes to the user's own calendar rather than mailing a colleague,
    and is still outbound - the test is whether it changes state the user would
    have to undo by hand.
    """
    from quorum.tracking.planner import PlannedAction

    outbound = {
        ActionType.REMIND, ActionType.NUDGE, ActionType.ESCALATE, ActionType.SCHEDULE
    }
    for action_type in ActionType:
        assert PlannedAction("id", action_type, "r").is_outbound == (action_type in outbound)


def test_planner_never_proposes_a_schedule_action():
    """SCHEDULE belongs to the calendar sync, not the daily sweep. If the
    planner started emitting it, `today` would silently try to write events."""
    ledger = Ledger("p")
    ledger.commitments.append(commit("overdue", due=date(2026, 3, 1)))
    ledger.commitments.append(commit("upcoming", due=date(2026, 3, 11)))
    ledger.commitments.append(commit("undated"))

    report = Planner().plan(ledger, TODAY)
    assert all(a.action is not ActionType.SCHEDULE for a in report.actions)


def test_apply_records_the_nudge_only_for_outbound_actions():
    item = commit("x", due=date(2026, 3, 10))
    action, _ = plan_one(item)
    Planner.apply(action, item, TODAY)

    assert item.nudge_count == 1 and item.last_action_on == TODAY

    waiting = commit("y", due=date(2026, 5, 1))
    wait_action, _ = plan_one(waiting)
    Planner.apply(wait_action, waiting, TODAY)
    assert waiting.nudge_count == 0


def test_planning_does_not_mutate_nudge_counts():
    """Scoring proposed actions offline must not change state."""
    item = commit("x", due=date(2026, 3, 10))
    plan_one(item)
    plan_one(item)
    assert item.nudge_count == 0


def test_report_summarises_for_metrics():
    ledger = Ledger("proj")
    ledger.commitments.append(commit("a", due=date(2026, 3, 10)))
    ledger.commitments.append(commit("b", due=date(2026, 5, 1)))

    payload = Planner().plan(ledger, TODAY).as_dict()
    assert payload["total"] == 2
    assert payload["outbound"] == 1
    assert "nudge" in payload["by_action"]
