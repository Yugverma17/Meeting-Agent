"""Deadlines as calendar events.

The alternative was a notifier: a background process that wakes up, checks the
ledger and raises a desktop toast. It was rejected, and the reason generalises
past this feature.

A homemade notifier only fires when the machine it lives on is awake, running,
and has the agent installed. A calendar event fires on the phone in your pocket.
Google has already solved delivery, snoozing, timezones and daylight saving; the
same reliability written here would be a background service to keep alive, and
it would still lose to a lock-screen notification. The interesting part of this
project is deciding *what* is worth reminding someone about - not re-implementing
the part of the problem that is already commodity.

Three properties the implementation holds to:

**Idempotent.** Every event carries the commitment id it came from in a private
extended property. Re-running the sync finds its own events and updates them;
it does not accumulate a second copy of Friday's deadline every morning. This is
the property that makes it safe to run from a scheduled task.

**Reversible in one place.** Only events Quorum created are ever touched. The
lookup is by extended property, never by title or date, so nothing the user put
in their own calendar can be matched by accident.

**Gated.** Writing to a real person's calendar is an outbound side effect and
goes through the same approval gate as email. `plan()` is pure and free to call;
`apply()` is reached only via an approved action.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum
from typing import Any

from quorum.models import Commitment, CommitmentStatus, CommitmentStrength
from quorum.tracking.ledger import Ledger

log = logging.getLogger(__name__)

COMMITMENT_KEY = "quorumCommitmentId"
PROJECT_KEY = "quorumProjectId"
MAX_TITLE = 90

# Google rejects reminders further out than four weeks.
MAX_REMINDER_MINUTES = 40320
MAX_REMINDER_OVERRIDES = 5


class ChangeKind(str, Enum):
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    UNCHANGED = "unchanged"


@dataclass
class CalendarConfig:
    calendar_id: str = "primary"

    reminder_days: tuple[int, ...] = (3, 1)
    """Lead times, in days before the deadline. Two by default: far enough out
    to still act on, and close enough to be the last word."""

    reminder_hour: int = 9
    """Local hour the reminder fires. All-day events start at midnight, so a
    naive 'N days before' would pop at 00:00 - technically correct and useless."""

    include_tentative: bool = False
    """Firm commitments only, by default. A calendar filled with "I could
    probably take a look at that" is a calendar people stop reading."""

    delete_resolved: bool = True
    """Remove events for commitments that closed. Only ever removes events this
    tool created."""


@dataclass
class CalendarChange:
    kind: ChangeKind
    commitment_id: str
    title: str
    due: date | None = None
    reason: str = ""
    event_id: str | None = None
    body: dict[str, Any] | None = field(default=None, repr=False)

    @property
    def is_write(self) -> bool:
        return self.kind is not ChangeKind.UNCHANGED


@dataclass
class CalendarPlan:
    """What a sync would do. Produced without writing anything."""

    changes: list[CalendarChange] = field(default_factory=list)
    undated: list[Commitment] = field(default_factory=list)
    """Open commitments with no resolvable deadline. They cannot be scheduled,
    and surfacing them here is what turns "the agent quietly ignored it" into a
    question the user can answer."""

    calendar_id: str = "primary"

    @property
    def writes(self) -> list[CalendarChange]:
        return [c for c in self.changes if c.is_write]

    def of_kind(self, kind: ChangeKind) -> list[CalendarChange]:
        return [c for c in self.changes if c.kind is kind]

    @property
    def is_empty(self) -> bool:
        return not self.writes

    def summary_line(self) -> str:
        counts = {
            kind.value: len(self.of_kind(kind))
            for kind in (ChangeKind.CREATE, ChangeKind.UPDATE, ChangeKind.DELETE)
        }
        return (
            f"{counts['create']} to add, {counts['update']} to update, "
            f"{counts['delete']} to remove"
        )

    def render(self) -> str:
        """What a human reads before approving. If this does not describe the
        effect accurately, the approval is not informed consent."""
        verb = {
            ChangeKind.CREATE: "add", ChangeKind.UPDATE: "move", ChangeKind.DELETE: "remove"
        }
        lines = [f"Calendar: {self.calendar_id}", ""]
        for change in self.writes:
            when = change.due.isoformat() if change.due else "-"
            lines.append(f"  {verb[change.kind]:>6} {when}  {change.title}")
            if change.reason:
                lines.append(f"         {change.reason}")
        if self.undated:
            lines += ["", f"Not scheduled - no deadline ({len(self.undated)}):"]
            lines += [f"  - {c.description}" for c in self.undated]
        return "\n".join(lines)


@dataclass
class SyncResult:
    created: int = 0
    updated: int = 0
    deleted: int = 0
    failed: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.created + self.updated + self.deleted

    def as_dict(self) -> dict[str, Any]:
        return {
            "created": self.created, "updated": self.updated,
            "deleted": self.deleted, "failed": self.failed,
        }


def reminder_overrides(days: tuple[int, ...], hour: int) -> list[dict[str, Any]]:
    """Lead times in days -> Google's minutes-before-event-start.

    All-day events start at midnight on the due date, so `days` alone would fire
    the popup at 00:00. Subtracting `hour` moves it to a time a person is awake:
    one day before at 09:00 is 1440 - 540 = 900 minutes.
    """
    overrides = []
    for day in sorted(set(days), reverse=True):
        minutes = day * 24 * 60 - hour * 60
        if 0 < minutes <= MAX_REMINDER_MINUTES:
            overrides.append({"method": "popup", "minutes": minutes})
    return overrides[:MAX_REMINDER_OVERRIDES]


class CalendarSync:
    """Diffs the ledger against the calendar, then applies the difference."""

    def __init__(self, service: Any = None, config: CalendarConfig | None = None) -> None:
        self.service = service
        """A Google Calendar v3 client, or None for a dry run. Injected rather
        than constructed so tests can drive a fake and never touch a network."""

        self.config = config or CalendarConfig()

    # -- planning ----------------------------------------------------------

    def plan(self, ledger: Ledger, today: date | None = None) -> CalendarPlan:
        """Work out the difference. Reads; never writes."""
        today = today or date.today()
        plan = CalendarPlan(calendar_id=self.config.calendar_id)
        existing = self._existing_events(ledger.project_id)

        wanted: dict[str, Commitment] = {}
        for commitment in ledger.open_commitments():
            if not self._should_schedule(commitment):
                continue
            if commitment.deadline.resolved is None:
                plan.undated.append(commitment)
                continue
            wanted[commitment.id] = commitment

        for commitment_id, commitment in wanted.items():
            body = self._event_body(commitment, ledger)
            event = existing.get(commitment_id)
            title = _title(commitment)
            due = commitment.deadline.resolved

            if event is None:
                plan.changes.append(CalendarChange(
                    ChangeKind.CREATE, commitment_id, title, due,
                    reason=self._why(commitment, today), body=body,
                ))
                continue

            current_start = (event.get("start") or {}).get("date")
            if current_start == body["start"]["date"] and event.get("summary") == title:
                plan.changes.append(CalendarChange(
                    ChangeKind.UNCHANGED, commitment_id, title, due, event_id=event["id"],
                ))
                continue

            moved = current_start and current_start != body["start"]["date"]
            plan.changes.append(CalendarChange(
                ChangeKind.UPDATE, commitment_id, title, due,
                reason=f"deadline moved from {current_start}" if moved else "wording changed",
                event_id=event["id"], body=body,
            ))

        if self.config.delete_resolved:
            for commitment_id, event in existing.items():
                if commitment_id in wanted:
                    continue
                plan.changes.append(CalendarChange(
                    ChangeKind.DELETE, commitment_id,
                    event.get("summary", commitment_id),
                    reason=self._closed_reason(ledger, commitment_id),
                    event_id=event["id"],
                ))

        return plan

    def _should_schedule(self, commitment: Commitment) -> bool:
        if commitment.strength is CommitmentStrength.MUSING:
            return False
        if commitment.strength is CommitmentStrength.TENTATIVE:
            return self.config.include_tentative
        return True

    @staticmethod
    def _why(commitment: Commitment, today: date) -> str:
        due = commitment.deadline.resolved
        if due is None:
            return ""
        days = (due - today).days
        if days < 0:
            return f"already {-days} day(s) overdue"
        return f"due in {days} day(s)"

    @staticmethod
    def _closed_reason(ledger: Ledger, commitment_id: str) -> str:
        commitment = ledger.by_id(commitment_id)
        if commitment is None:
            return "no longer in the ledger"
        return f"now {commitment.status.value}"

    # -- event construction ------------------------------------------------

    def _event_body(self, commitment: Commitment, ledger: Ledger) -> dict[str, Any]:
        due = commitment.deadline.resolved
        assert due is not None  # callers filter undated commitments first
        return {
            "summary": _title(commitment),
            "description": _description(commitment, ledger),
            # All-day. `end` is exclusive in the Calendar API, so a one-day
            # event ends the following day - off by one here puts every
            # deadline on the wrong date.
            "start": {"date": due.isoformat()},
            "end": {"date": (due + timedelta(days=1)).isoformat()},
            # A deadline is not a meeting. Marking it opaque would make the
            # owner look busy all day to anyone checking their availability.
            "transparency": "transparent",
            "reminders": {
                "useDefault": False,
                "overrides": reminder_overrides(
                    self.config.reminder_days, self.config.reminder_hour
                ),
            },
            "extendedProperties": {
                "private": {
                    COMMITMENT_KEY: commitment.id,
                    PROJECT_KEY: commitment.project_id or ledger.project_id or "",
                }
            },
        }

    # -- reading -----------------------------------------------------------

    def _existing_events(self, project_id: str) -> dict[str, dict[str, Any]]:
        """Every event this tool previously created for the project.

        Matched on the extended property alone. Matching on title or date would
        eventually collide with something the user wrote themselves, and the
        failure mode of that is deleting a real appointment.
        """
        if self.service is None:
            return {}

        found: dict[str, dict[str, Any]] = {}
        page_token = None
        while True:
            response = self.service.events().list(
                calendarId=self.config.calendar_id,
                privateExtendedProperty=f"{PROJECT_KEY}={project_id}",
                showDeleted=False,
                singleEvents=True,
                maxResults=250,
                pageToken=page_token,
            ).execute()

            for event in response.get("items", []):
                private = (event.get("extendedProperties") or {}).get("private") or {}
                commitment_id = private.get(COMMITMENT_KEY)
                if commitment_id:
                    found[commitment_id] = event

            page_token = response.get("nextPageToken")
            if not page_token:
                return found

    # -- writing -----------------------------------------------------------

    def apply(self, plan: CalendarPlan) -> SyncResult:
        """Perform the plan. Reached only through the approval gate."""
        result = SyncResult()
        if self.service is None:
            raise RuntimeError("No calendar service - this is a dry-run sync")

        for change in plan.writes:
            try:
                self._apply_one(change, result)
            except Exception as exc:  # noqa: BLE001 - one bad event must not abort the rest
                log.warning("Calendar %s failed for %s: %s",
                            change.kind.value, change.commitment_id, exc)
                result.failed.append(f"{change.kind.value} {change.title}: {exc}")
        return result

    def _apply_one(self, change: CalendarChange, result: SyncResult) -> None:
        events = self.service.events()
        if change.kind is ChangeKind.CREATE:
            events.insert(calendarId=self.config.calendar_id, body=change.body).execute()
            result.created += 1
        elif change.kind is ChangeKind.UPDATE:
            events.patch(
                calendarId=self.config.calendar_id,
                eventId=change.event_id,
                body=change.body,
            ).execute()
            result.updated += 1
        elif change.kind is ChangeKind.DELETE:
            events.delete(
                calendarId=self.config.calendar_id, eventId=change.event_id
            ).execute()
            result.deleted += 1


class CalendarTransport:
    """Adapts a calendar sync to the approval gate's transport protocol.

    The gate calls `send(item)` on an approved action. Holding the plan on the
    transport rather than serialising it into the action's body means the thing
    that gets applied is exactly the thing that was shown and approved - there
    is no re-planning step in between where the world could have changed.
    """

    def __init__(self, sync: CalendarSync, plan: CalendarPlan) -> None:
        self.sync = sync
        self.plan = plan
        self.result: SyncResult | None = None

    def send(self, item) -> bool:  # noqa: ANN001 - PendingAction, kept duck-typed
        self.result = self.sync.apply(self.plan)
        if self.result.failed:
            log.warning("Calendar sync completed with %d failure(s)", len(self.result.failed))
        return self.result.total > 0 or not self.plan.writes


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _title(commitment: Commitment) -> str:
    who = commitment.assignee.display_name
    text = commitment.description.strip()
    prefix = f"{who}: " if who else ""
    title = f"{prefix}{text}"
    if len(title) > MAX_TITLE:
        title = title[: MAX_TITLE - 1].rstrip() + "…"
    return title


def _description(commitment: Commitment, ledger: Ledger) -> str:
    """The event carries its own evidence.

    Same principle as the digests: a reminder you cannot check is a reminder you
    have to take on trust. Six weeks later, "you said this, on this date" is the
    difference between acting on it and wondering where it came from.
    """
    lines = [commitment.description.strip(), ""]

    when = ledger.meeting_dates.get(commitment.meeting_id)
    for evidence in commitment.evidence[:1]:
        said = f'"{evidence.quote.strip()}"'
        lines.append(f"Said{f' on {when.isoformat()}' if when else ''}: {said}")

    if commitment.assignee.display_name:
        lines.append(f"Owner: {commitment.assignee.display_name}")
    if commitment.deadline.raw_text:
        lines.append(f"Deadline as stated: {commitment.deadline.raw_text}")

    lines += [
        "",
        f"Commitment {commitment.id} - tracked by Quorum.",
        "Closing it in Quorum removes this event; deleting the event here does not "
        "close the commitment.",
    ]
    return "\n".join(lines)


def status_is_open(commitment: Commitment) -> bool:
    """Kept next to the sync because 'open' here means 'still worth a reminder',
    which is a slightly narrower question than the ledger's."""
    return commitment.status not in (
        CommitmentStatus.VERIFIED_DONE,
        CommitmentStatus.CANCELLED,
        CommitmentStatus.DROPPED,
    )
