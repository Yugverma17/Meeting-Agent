"""Turning spoken timing into actual dates.

People do not say "2026-03-13" in meetings. They say "Friday", "end of next
week", "before the demo". Every one of those is relative to *when the meeting
happened*, which is why nothing here uses today's date.

Deterministic rules run first and cover the overwhelming majority of phrasings
at zero token cost. Only genuinely ambiguous text falls through to a model.
"""

from __future__ import annotations

import calendar
import logging
import re
from dataclasses import dataclass
from datetime import date, timedelta

from dateutil import parser as dateparser

from quorum.models import DeadlineResolution

log = logging.getLogger(__name__)

WEEKDAYS = {
    "monday": 0, "mon": 0,
    "tuesday": 1, "tues": 1, "tue": 1,
    "wednesday": 2, "weds": 2, "wed": 2,
    "thursday": 3, "thurs": 3, "thu": 3,
    "friday": 4, "fri": 4,
    "saturday": 5, "sat": 5,
    "sunday": 6, "sun": 6,
}

_NUMBER_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

_WS = re.compile(r"\s+")
_FILLER = re.compile(r"^(by|before|on|at|due|no later than|latest|sometime|the)\s+")


@dataclass
class ResolvedDate:
    value: date | None
    method: DeadlineResolution
    confidence: float
    note: str = ""


def _clean(text: str) -> str:
    cleaned = _WS.sub(" ", text.strip().lower()).strip(" .,!?;:")
    previous = None
    # Strip stacked lead-ins: "by no later than the 20th".
    while previous != cleaned:
        previous = cleaned
        cleaned = _FILLER.sub("", cleaned).strip()
    return cleaned


def next_weekday(origin: date, weekday: int, *, allow_same_day: bool = True) -> date:
    """The next occurrence of `weekday` on or after `origin`."""
    delta = (weekday - origin.weekday()) % 7
    if delta == 0 and not allow_same_day:
        delta = 7
    return origin + timedelta(days=delta)


def end_of_week(origin: date) -> date:
    """Friday of the origin's week - the working sense of 'end of the week'."""
    return origin + timedelta(days=(4 - origin.weekday()) % 7)


def end_of_month(origin: date) -> date:
    return origin.replace(day=calendar.monthrange(origin.year, origin.month)[1])


def resolve_deadline(
    raw_text: str | None,
    meeting_date: date,
    known_events: dict[str, date] | None = None,
) -> ResolvedDate:
    """Best-effort deterministic resolution of a spoken deadline.

    `known_events` anchors phrases like "before the demo" to a real calendar
    date. Without it, such phrases resolve to nothing rather than to a guess -
    a wrong date is worse than an admitted gap, because it silently produces a
    calendar reminder on the wrong day.
    """
    if not raw_text or not raw_text.strip():
        return ResolvedDate(None, DeadlineResolution.NONE, 0.0)

    # Two views of the text. `text` has lead-in words stripped for pattern
    # matching; `spoken` keeps them, because for event anchors the lead-in *is*
    # the meaning - "before the demo" and "at the demo" are different days.
    spoken = _WS.sub(" ", raw_text.strip().lower()).strip(" .,!?;:")
    text = _clean(raw_text)
    events = {k.lower(): v for k, v in (known_events or {}).items()}

    # --- anchored to a known event ------------------------------------
    for name, event_date in events.items():
        if name and name in spoken:
            if any(word in spoken for word in ("before", "ahead of", "prior to", "by")):
                return ResolvedDate(
                    event_date - timedelta(days=1), DeadlineResolution.ANCHORED, 0.85,
                    f"day before {name}",
                )
            return ResolvedDate(event_date, DeadlineResolution.ANCHORED, 0.8, f"on {name}")

    # An event was referenced but its date is unknown. Reporting the gap beats
    # falling through to a guess that would schedule a reminder on a wrong day.
    if re.search(r"\b(before|ahead of|prior to|after)\s+(the|our|my|his|her|their)\b", spoken):
        return ResolvedDate(
            None, DeadlineResolution.ANCHORED, 0.0, "references an event with no known date"
        )

    # --- immediate ----------------------------------------------------
    if text in {"today", "eod", "end of day", "cob", "close of business", "tonight"}:
        return ResolvedDate(meeting_date, DeadlineResolution.RELATIVE, 0.95)
    if text in {"tomorrow", "tmrw"}:
        return ResolvedDate(meeting_date + timedelta(days=1), DeadlineResolution.RELATIVE, 0.95)
    if text in {"asap", "immediately", "right away", "now"}:
        return ResolvedDate(meeting_date, DeadlineResolution.RELATIVE, 0.6, "urgency, not a date")

    # --- week-scale ---------------------------------------------------
    # Order matters: "end of next week" must be tested before "next week".
    if re.search(r"\b(end of (the )?next week|next week end)\b", text):
        return ResolvedDate(end_of_week(meeting_date) + timedelta(days=7),
                            DeadlineResolution.RELATIVE, 0.85)
    if re.search(r"\b(end of (the )?week|eow|end of this week)\b", text):
        return ResolvedDate(end_of_week(meeting_date), DeadlineResolution.RELATIVE, 0.85)
    if re.search(r"\bnext week\b", text):
        return ResolvedDate(end_of_week(meeting_date) + timedelta(days=7),
                            DeadlineResolution.RELATIVE, 0.7, "assumed end of next week")
    if re.search(r"\bthis week\b", text):
        return ResolvedDate(end_of_week(meeting_date), DeadlineResolution.RELATIVE, 0.8)

    # --- month-scale ---------------------------------------------------
    if re.search(r"\bend of (the )?month\b", text):
        return ResolvedDate(end_of_month(meeting_date), DeadlineResolution.RELATIVE, 0.85)
    if re.search(r"\bnext month\b", text):
        first_next = end_of_month(meeting_date) + timedelta(days=1)
        return ResolvedDate(end_of_month(first_next), DeadlineResolution.RELATIVE, 0.6,
                            "assumed end of next month")

    # --- "in N days/weeks" ---------------------------------------------
    span = re.search(
        r"\b(?:in|within|after)\s+(\d+|" + "|".join(_NUMBER_WORDS) + r")\s+(day|week|month)s?\b",
        text,
    )
    if span:
        raw_amount = span.group(1)
        amount = int(raw_amount) if raw_amount.isdigit() else _NUMBER_WORDS[raw_amount]
        unit = span.group(2)
        days = {"day": 1, "week": 7, "month": 30}[unit] * amount
        return ResolvedDate(meeting_date + timedelta(days=days),
                            DeadlineResolution.RELATIVE, 0.85)

    # --- weekday names --------------------------------------------------
    weekday_match = re.search(r"\b(next|this|coming)?\s*([a-z]+day|mon|tues?|weds?|thurs?|fri|sat|sun)\b", text)
    if weekday_match and weekday_match.group(2) in WEEKDAYS:
        qualifier = (weekday_match.group(1) or "").strip()
        target = WEEKDAYS[weekday_match.group(2)]
        # A bare weekday means the next one, today included; "next Friday" means
        # the week after, which is what people almost always intend.
        resolved = next_weekday(meeting_date, target, allow_same_day=qualifier != "next")
        if qualifier == "next" and (resolved - meeting_date).days < 7:
            resolved += timedelta(days=7)
        return ResolvedDate(resolved, DeadlineResolution.RELATIVE, 0.9)

    # --- explicit calendar dates ----------------------------------------
    ordinal = re.fullmatch(r"(\d{1,2})(st|nd|rd|th)", text)
    if ordinal:
        day = int(ordinal.group(1))
        try:
            candidate = meeting_date.replace(day=day)
        except ValueError:
            return ResolvedDate(None, DeadlineResolution.NONE, 0.0, "invalid day of month")
        if candidate < meeting_date:  # "the 3rd" said on the 20th means next month
            following = end_of_month(meeting_date) + timedelta(days=1)
            try:
                candidate = following.replace(day=day)
            except ValueError:
                return ResolvedDate(None, DeadlineResolution.NONE, 0.0, "invalid day of month")
        return ResolvedDate(candidate, DeadlineResolution.EXPLICIT, 0.85)

    try:
        # default= supplies the meeting's year/month for partial dates like
        # "March 14", and stops dateutil silently substituting today.
        parsed = dateparser.parse(text, default=_default_for(meeting_date), fuzzy=False)
        if parsed:
            return ResolvedDate(parsed.date(), DeadlineResolution.EXPLICIT, 0.8)
    except (ValueError, OverflowError, TypeError):
        pass

    return ResolvedDate(None, DeadlineResolution.NONE, 0.0, f"unparsed: {raw_text!r}")


def _default_for(meeting_date: date):
    from datetime import datetime

    return datetime(meeting_date.year, meeting_date.month, meeting_date.day)
