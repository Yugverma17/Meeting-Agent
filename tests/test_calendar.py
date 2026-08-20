"""Calendar sync.

The fake service below is the whole test strategy: the Google client is an
object with `.events().list/insert/patch/delete().execute()`, so a small stand-in
covers every path without a network, a token, or a mocking framework. What is
actually being tested is the diff - whether a second run is a no-op, whether a
moved deadline patches instead of duplicating, and whether anything the user
wrote themselves can be touched.
"""

from __future__ import annotations

from datetime import date

import pytest

from quorum.execution.calendar import (
    COMMITMENT_KEY,
    PROJECT_KEY,
    CalendarConfig,
    CalendarPlan,
    CalendarSync,
    CalendarTransport,
    ChangeKind,
    reminder_overrides,
)
from quorum.models import (
    Assignee,
    Commitment,
    CommitmentStatus,
    CommitmentStrength,
    Deadline,
    DeadlineResolution,
    Evidence,
)
from quorum.tracking import Ledger

TODAY = date(2026, 3, 16)
DUE = date(2026, 3, 20)


def commit(
    description: str = "send the ingestion spec",
    *,
    due: date | None = DUE,
    strength: CommitmentStrength = CommitmentStrength.FIRM,
    status: CommitmentStatus = CommitmentStatus.OPEN,
    commitment_id: str = "cmt_1",
) -> Commitment:
    return Commitment(
        id=commitment_id,
        description=description,
        meeting_id="mtg_1",
        project_id="proj",
        assignee=Assignee(
            speaker_id="spk_yug", display_name="Yug Verma", email="yug@example.com",
            confidence=0.9,
        ),
        deadline=Deadline(
            raw_text="by Friday", resolved=due,
            method=DeadlineResolution.RELATIVE if due else DeadlineResolution.NONE,
        ),
        strength=strength,
        status=status,
        evidence=[Evidence(utterance_id="utt_1", quote="I'll have the spec to you by Friday")],
    )


def ledger_with(*commitments: Commitment) -> Ledger:
    ledger = Ledger("proj")
    ledger.commitments.extend(commitments)
    ledger.meeting_dates["mtg_1"] = date(2026, 3, 13)
    return ledger


# --- the fake Google client -------------------------------------------------


class FakeRequest:
    def __init__(self, result) -> None:
        self._result = result

    def execute(self):
        return self._result


class FakeEvents:
    def __init__(self, store: dict[str, dict]) -> None:
        self.store = store
        self.calls: list[tuple[str, str]] = []
        self.next_id = len(store) + 1
        self.fail_on: set[str] = set()

    def list(self, **kwargs):
        self.calls.append(("list", kwargs.get("privateExtendedProperty", "")))
        wanted = kwargs.get("privateExtendedProperty", "")
        items = [
            event for event in self.store.values()
            if not wanted or _has_property(event, wanted)
        ]
        return FakeRequest({"items": items})

    def insert(self, calendarId, body):  # noqa: N803 - Google's parameter name
        event_id = f"evt_{self.next_id}"
        self.next_id += 1
        self.calls.append(("insert", body["summary"]))
        self.store[event_id] = {"id": event_id, **body}
        return FakeRequest(self.store[event_id])

    def patch(self, calendarId, eventId, body):  # noqa: N803
        self.calls.append(("patch", eventId))
        if eventId in self.fail_on:
            raise RuntimeError("simulated Google 500")
        self.store[eventId] = {**self.store[eventId], **body}
        return FakeRequest(self.store[eventId])

    def delete(self, calendarId, eventId):  # noqa: N803
        self.calls.append(("delete", eventId))
        self.store.pop(eventId, None)
        return FakeRequest(None)


class FakeService:
    def __init__(self, events: list[dict] | None = None) -> None:
        store = {event["id"]: event for event in (events or [])}
        self._events = FakeEvents(store)

    def events(self):
        return self._events


def _has_property(event: dict, wanted: str) -> bool:
    key, _, value = wanted.partition("=")
    private = (event.get("extendedProperties") or {}).get("private") or {}
    return private.get(key) == value


def quorum_event(
    event_id: str = "evt_1",
    commitment_id: str = "cmt_1",
    start: date = DUE,
    summary: str = "Yug Verma: send the ingestion spec",
) -> dict:
    return {
        "id": event_id,
        "summary": summary,
        "start": {"date": start.isoformat()},
        "extendedProperties": {
            "private": {COMMITMENT_KEY: commitment_id, PROJECT_KEY: "proj"}
        },
    }


# --- reminders --------------------------------------------------------------


def test_reminders_land_in_the_morning_not_at_midnight():
    """All-day events start at 00:00, so a naive 'one day before' fires then.

    Correct, and useless - the point of a lead time is that someone sees it.
    """
    overrides = reminder_overrides((3, 1), hour=9)

    assert [o["minutes"] for o in overrides] == [3 * 1440 - 540, 1440 - 540]
    assert all(o["method"] == "popup" for o in overrides)


def test_reminders_are_ordered_furthest_out_first_and_deduplicated():
    assert [o["minutes"] for o in reminder_overrides((1, 3, 1), hour=9)] == [3780, 900]


def test_reminder_beyond_googles_limit_is_dropped_not_sent():
    """Google rejects anything past four weeks with a 400 that fails the whole
    insert - so an over-long lead time must never reach the API."""
    overrides = reminder_overrides((30, 1), hour=9)

    assert [o["minutes"] for o in overrides] == [900]


def test_same_day_lead_time_is_dropped():
    """A zero-day lead resolves to a negative offset, which Google rejects."""
    assert reminder_overrides((0,), hour=9) == []


# --- planning ---------------------------------------------------------------


def test_open_commitment_with_a_deadline_becomes_an_event():
    sync = CalendarSync(FakeService(), CalendarConfig())
    plan = sync.plan(ledger_with(commit()), TODAY)

    (change,) = plan.of_kind(ChangeKind.CREATE)
    assert change.due == DUE
    assert "send the ingestion spec" in change.title
    assert change.body["start"]["date"] == "2026-03-20"


def test_all_day_event_ends_the_following_day():
    """The Calendar API treats `end` as exclusive. Off by one here puts every
    deadline on the wrong date, and it looks right in the plan output."""
    sync = CalendarSync(FakeService(), CalendarConfig())
    (change,) = sync.plan(ledger_with(commit()), TODAY).of_kind(ChangeKind.CREATE)

    assert change.body["start"]["date"] == "2026-03-20"
    assert change.body["end"]["date"] == "2026-03-21"


def test_deadline_events_do_not_make_you_look_busy():
    sync = CalendarSync(FakeService(), CalendarConfig())
    (change,) = sync.plan(ledger_with(commit()), TODAY).of_kind(ChangeKind.CREATE)

    assert change.body["transparency"] == "transparent"


def test_event_carries_the_quote_that_created_the_obligation():
    sync = CalendarSync(FakeService(), CalendarConfig())
    (change,) = sync.plan(ledger_with(commit()), TODAY).of_kind(ChangeKind.CREATE)

    assert "I'll have the spec to you by Friday" in change.body["description"]
    assert "2026-03-13" in change.body["description"], "the meeting date should be cited"


def test_commitment_without_a_deadline_is_surfaced_not_silently_skipped():
    plan = CalendarSync(FakeService(), CalendarConfig()).plan(
        ledger_with(commit(due=None)), TODAY
    )

    assert plan.is_empty
    assert [c.description for c in plan.undated] == ["send the ingestion spec"]


def test_musings_never_reach_the_calendar():
    plan = CalendarSync(FakeService(), CalendarConfig()).plan(
        ledger_with(commit(strength=CommitmentStrength.MUSING)), TODAY
    )

    assert plan.is_empty and not plan.undated


def test_tentative_commitments_are_excluded_by_default_and_opt_in():
    tentative = commit(strength=CommitmentStrength.TENTATIVE)

    assert CalendarSync(FakeService()).plan(ledger_with(tentative), TODAY).is_empty
    opted_in = CalendarSync(
        FakeService(), CalendarConfig(include_tentative=True)
    ).plan(ledger_with(tentative), TODAY)
    assert len(opted_in.of_kind(ChangeKind.CREATE)) == 1


# --- idempotency ------------------------------------------------------------


def test_second_run_changes_nothing():
    """The property that makes it safe to run from a scheduled task. Without
    it, every morning adds another copy of Friday's deadline."""
    service = FakeService([quorum_event()])
    plan = CalendarSync(service, CalendarConfig()).plan(ledger_with(commit()), TODAY)

    assert plan.is_empty
    assert len(plan.of_kind(ChangeKind.UNCHANGED)) == 1


def test_moved_deadline_patches_the_existing_event():
    service = FakeService([quorum_event(start=date(2026, 3, 18))])
    plan = CalendarSync(service, CalendarConfig()).plan(ledger_with(commit()), TODAY)

    (change,) = plan.of_kind(ChangeKind.UPDATE)
    assert change.event_id == "evt_1"
    assert "2026-03-18" in change.reason
    assert not plan.of_kind(ChangeKind.CREATE), "a moved deadline must not duplicate"


def test_closed_commitment_removes_its_event():
    service = FakeService([quorum_event()])
    ledger = ledger_with(commit(status=CommitmentStatus.VERIFIED_DONE))

    (change,) = CalendarSync(service, CalendarConfig()).plan(ledger, TODAY).of_kind(
        ChangeKind.DELETE
    )
    assert change.event_id == "evt_1"
    assert "verified_done" in change.reason


def test_keeping_resolved_events_is_configurable():
    service = FakeService([quorum_event()])
    ledger = ledger_with(commit(status=CommitmentStatus.VERIFIED_DONE))

    plan = CalendarSync(service, CalendarConfig(delete_resolved=False)).plan(ledger, TODAY)
    assert plan.is_empty


def test_events_the_user_created_are_never_touched():
    """Matching is on the private extended property alone. Matching on title or
    date would eventually delete a real appointment, and that failure is silent."""
    theirs = {
        "id": "evt_theirs",
        "summary": "Yug Verma: send the ingestion spec",  # identical title
        "start": {"date": DUE.isoformat()},
    }
    service = FakeService([theirs])
    plan = CalendarSync(service, CalendarConfig()).plan(ledger_with(commit()), TODAY)

    assert not plan.of_kind(ChangeKind.DELETE)
    assert len(plan.of_kind(ChangeKind.CREATE)) == 1, "ours is created alongside theirs"
    assert "evt_theirs" in service.events().store


def test_lookup_is_scoped_to_the_project():
    service = FakeService()
    CalendarSync(service, CalendarConfig()).plan(ledger_with(commit()), TODAY)

    kind, query = service.events().calls[0]
    assert kind == "list"
    assert query == f"{PROJECT_KEY}=proj"


# --- applying ---------------------------------------------------------------


def test_apply_performs_each_change_once():
    service = FakeService([quorum_event(event_id="evt_stale", commitment_id="cmt_gone")])
    sync = CalendarSync(service, CalendarConfig())
    plan = sync.plan(ledger_with(commit()), TODAY)

    result = sync.apply(plan)

    assert (result.created, result.deleted, result.failed) == (1, 1, [])
    assert [kind for kind, _ in service.events().calls if kind != "list"] == [
        "insert", "delete"
    ]


def test_a_failed_event_does_not_abort_the_rest():
    service = FakeService([
        quorum_event(event_id="evt_1", commitment_id="cmt_1", start=date(2026, 3, 18)),
        quorum_event(event_id="evt_2", commitment_id="cmt_2", start=date(2026, 3, 18),
                     summary="Yug Verma: review the spec"),
    ])
    service.events().fail_on.add("evt_1")
    sync = CalendarSync(service, CalendarConfig())
    plan = sync.plan(
        ledger_with(commit(), commit("review the spec", commitment_id="cmt_2")), TODAY
    )

    result = sync.apply(plan)

    assert result.updated == 1
    assert len(result.failed) == 1 and "simulated Google 500" in result.failed[0]


def test_dry_run_without_a_service_plans_but_cannot_apply():
    sync = CalendarSync(None, CalendarConfig())
    plan = sync.plan(ledger_with(commit()), TODAY)

    assert len(plan.of_kind(ChangeKind.CREATE)) == 1, "a dry run still shows the work"
    with pytest.raises(RuntimeError, match="dry-run"):
        sync.apply(plan)


# --- the gate ---------------------------------------------------------------


def test_calendar_writes_require_an_approval_token():
    """Same invariant as email: there is no argument to `execute` that skips
    the gate, so a caller cannot write to a calendar by setting a flag."""
    from quorum.execution import ApprovalGate
    from quorum.execution.approval import NotApproved
    from quorum.tracking import ActionType, PlannedAction

    service = FakeService()
    sync = CalendarSync(service, CalendarConfig())
    plan = sync.plan(ledger_with(commit()), TODAY)

    gate = ApprovalGate()
    pending = gate.propose(
        PlannedAction("calendar:proj", ActionType.SCHEDULE, "1 to add"), "sync"
    )
    transport = CalendarTransport(sync, plan)

    with pytest.raises(NotApproved):
        gate.execute(pending.id, "guessed-token", transport)
    assert not [c for c in service.events().calls if c[0] == "insert"]

    gate.execute(pending.id, gate.approve(pending.id), transport)
    assert transport.result.created == 1


def test_rendered_plan_names_every_change_it_will_make():
    """What the human reads before approving. If it under-reports the effect,
    the approval is not informed consent."""
    service = FakeService([quorum_event(event_id="evt_stale", commitment_id="cmt_gone")])
    sync = CalendarSync(service, CalendarConfig())
    plan = sync.plan(ledger_with(commit(), commit(due=None, commitment_id="cmt_2")), TODAY)

    rendered = plan.render()

    assert "add" in rendered and "2026-03-20" in rendered
    assert "remove" in rendered
    assert "no deadline" in rendered
    assert plan.summary_line() == "1 to add, 0 to update, 1 to remove"


def test_empty_plan_renders_without_error():
    assert CalendarPlan().render()
