"""The between-meetings report.

Each section exists because it cannot be produced from a single meeting, and
the tests are written to hold that line: a reversal needs two meetings, a
dropped commitment is defined by absence, a silent delivery comes from outside
the conversation entirely, and a slip needs the *history* of a date rather than
its current value.

Nothing here calls a model. If any of it ever needs one, something has drifted.
"""

from __future__ import annotations

from datetime import date, timedelta

from quorum.models import (
    Assignee,
    Commitment,
    CommitmentStatus,
    CommitmentStrength,
    Deadline,
    Decision,
    Evidence,
)
from quorum.tracking import Ledger, build_report

TODAY = date(2026, 8, 20)
LAST_WEEK = TODAY - timedelta(days=7)


def commitment(
    description: str,
    *,
    due: date | None = None,
    status: CommitmentStatus = CommitmentStatus.OPEN,
    cid: str = "c1",
    meeting_id: str = "m_recent",
    last_action: date | None = None,
    note: str | None = None,
    strength: CommitmentStrength = CommitmentStrength.FIRM,
) -> Commitment:
    return Commitment(
        id=cid, description=description, meeting_id=meeting_id,
        assignee=Assignee(speaker_id="spk_p", display_name="Priya Raghavan",
                          email="priya@example.com", confidence=0.9),
        deadline=Deadline(resolved=due),
        strength=strength, status=status,
        last_action_on=last_action, resolution_note=note,
        evidence=[Evidence(utterance_id="u1", quote=f"I'll {description}")],
    )


def ledger_with(*commitments, decisions=(), meetings=None) -> Ledger:
    ledger = Ledger("proj")
    ledger.commitments.extend(commitments)
    ledger.decisions.extend(decisions)
    ledger.meeting_dates.update(
        meetings or {"m_recent": TODAY - timedelta(days=2), "m_old": TODAY - timedelta(days=40)}
    )
    return ledger


def report_for(ledger: Ledger):
    return build_report(ledger, "Proj", until=TODAY, days=7)


# --- reversals: needs two meetings -------------------------------------------


def test_a_reversed_decision_is_reported_with_what_it_replaced():
    earlier = Decision(
        id="d1", statement="Use Postgres", meeting_id="m_old",
        evidence=[Evidence(utterance_id="u1", quote="let's use Postgres")],
    )
    later = Decision(
        id="d2", statement="Use DynamoDB instead", meeting_id="m_recent", reverses="d1",
        evidence=[Evidence(utterance_id="u2", quote="actually, DynamoDB")],
    )

    report = report_for(ledger_with(decisions=[earlier, later]))

    assert len(report.reversed_decisions) == 1
    assert report.reversed_decisions[0].earlier.statement == "Use Postgres"
    assert "Postgres" in report.as_markdown()


def test_a_reversal_from_before_the_window_is_not_this_week_s_news():
    earlier = Decision(id="d1", statement="Use Postgres", meeting_id="m_old",
                       evidence=[Evidence(utterance_id="u1", quote="postgres")])
    later = Decision(id="d2", statement="Use Mongo", meeting_id="m_old", reverses="d1",
                     evidence=[Evidence(utterance_id="u2", quote="mongo")])

    assert report_for(ledger_with(decisions=[earlier, later])).reversed_decisions == []


# --- dropped: defined by absence ---------------------------------------------


def test_a_commitment_marked_dropped_is_reported():
    report = report_for(ledger_with(
        commitment("write the spec", due=LAST_WEEK, status=CommitmentStatus.DROPPED)
    ))

    assert len(report.dropped) == 1
    assert "Quietly dropped" in report.as_markdown()


def test_long_overdue_and_never_mentioned_counts_as_dropped():
    """You cannot extract silence from a transcript. It is only visible by
    comparing the ledger to itself over time."""
    forgotten = commitment(
        "look into rate limiting", due=TODAY - timedelta(days=30),
        cid="c9", meeting_id="m_old", last_action=TODAY - timedelta(days=30),
    )

    report = report_for(ledger_with(forgotten))

    assert [c.id for c in report.dropped] == ["c9"]
    assert report.overdue == []


def test_recently_overdue_is_overdue_not_dropped():
    """Calling something abandoned while someone is still working on it is
    insulting and wrong."""
    recent = commitment("write the spec", due=TODAY - timedelta(days=3), cid="c2")

    report = report_for(ledger_with(recent))

    assert report.dropped == []
    assert [c.id for c in report.overdue] == ["c2"]


def test_something_chased_this_week_is_not_dropped():
    chased = commitment(
        "write the spec", due=TODAY - timedelta(days=30), cid="c3",
        last_action=TODAY - timedelta(days=1),
    )

    assert report_for(ledger_with(chased)).dropped == []


# --- silent delivery: from outside the conversation --------------------------


def test_delivery_confirmed_by_evidence_is_reported_as_quiet():
    delivered = commitment(
        "the schema migration", due=LAST_WEEK, cid="c4",
        status=CommitmentStatus.VERIFIED_DONE,
        note="github: PR #204", last_action=TODAY - timedelta(days=2),
    )

    report = report_for(ledger_with(delivered))

    assert [c.id for c in report.delivered_quietly] == ["c4"]


def test_delivery_someone_announced_is_not_quiet():
    """"Quietly" means the ledger learned it from evidence rather than from
    someone saying so - which is the whole justification for the evidence
    layer."""
    announced = commitment(
        "the schema migration", due=LAST_WEEK, cid="c5",
        status=CommitmentStatus.VERIFIED_DONE,
        note="claimed complete on 2026-08-18", last_action=TODAY - timedelta(days=2),
    )

    assert report_for(ledger_with(announced)).delivered_quietly == []


# --- slipping: needs the history, not the current value ----------------------


def test_a_date_that_moved_twice_is_reported_with_its_original():
    # Built forwards, as it happens in life: promised for the 8th, moved to the
    # 15th, moved again to the 29th.
    item = commitment("the ingestion spec", due=date(2026, 8, 8), cid="c6")
    item.record_deadline_change(date(2026, 8, 15), on=date(2026, 8, 8))
    item.deadline.resolved = date(2026, 8, 15)
    item.record_deadline_change(date(2026, 8, 29), on=date(2026, 8, 15))
    item.deadline.resolved = date(2026, 8, 29)

    report = report_for(ledger_with(item))

    assert len(report.slipping) == 1
    assert report.slipping[0].times == 2
    markdown = report.as_markdown()
    assert "moved 2x" in markdown


def test_one_move_is_not_yet_a_pattern():
    item = commitment("the spec", due=date(2026, 8, 29), cid="c7")
    item.record_deadline_change(date(2026, 8, 29), on=date(2026, 8, 8))

    assert report_for(ledger_with(item)).slipping == []


def test_pulling_a_deadline_forward_is_not_a_slip():
    """A team that got ahead of schedule must not be reported as in trouble."""
    # Promised for the 10th, slipped to the 30th, then pulled forward to the
    # 20th. One slip, not two movements.
    item = commitment("the spec", due=date(2026, 8, 10), cid="c8")
    item.record_deadline_change(date(2026, 8, 30), on=date(2026, 8, 12))
    item.deadline.resolved = date(2026, 8, 30)
    item.record_deadline_change(date(2026, 8, 20), on=date(2026, 8, 15))
    item.deadline.resolved = date(2026, 8, 20)

    assert len(item.deadline_history) == 2
    assert item.times_slipped == 1


def test_setting_a_first_deadline_is_not_a_slip():
    """Triage filling in a date that was never given is not the date moving."""
    item = commitment("the spec", cid="c10")
    item.record_deadline_change(date(2026, 8, 29), on=TODAY, source="triage")

    assert item.times_slipped == 0


def test_re_recording_the_same_date_records_nothing():
    item = commitment("the spec", due=date(2026, 8, 29), cid="c11")
    item.record_deadline_change(date(2026, 8, 29), on=TODAY)

    assert item.deadline_history == []


# --- balance and rendering ----------------------------------------------------


def test_new_promises_are_reported_too():
    """A report that only lists failures reads as an indictment and gets
    closed."""
    report = report_for(ledger_with(commitment("write the spec", due=TODAY + timedelta(days=5))))

    assert len(report.newly_promised) == 1
    assert "Newly promised" in report.as_markdown()


def test_a_quiet_week_says_so_rather_than_rendering_empty_headings():
    report = report_for(ledger_with())

    assert report.is_quiet
    assert "Nothing changed this week" in report.as_markdown()


def test_the_report_never_calls_a_model(monkeypatch):
    """Free, instant and identical on every run - properties that matter for
    something meant to be read every Monday."""
    def explode(*args, **kwargs):
        raise AssertionError("the weekly report must not call a model")

    monkeypatch.setattr("quorum.llm.router.get_router", explode)
    report_for(ledger_with(commitment("write the spec", due=LAST_WEEK))).as_markdown()
