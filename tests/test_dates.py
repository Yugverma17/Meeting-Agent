from __future__ import annotations

from datetime import date

import pytest

from quorum.agents.dates import end_of_month, end_of_week, next_weekday, resolve_deadline
from quorum.models import DeadlineResolution

# 2026-03-09 is a Monday. Every expectation below is relative to that meeting,
# never to the day the tests happen to run.
MONDAY = date(2026, 3, 9)
FRIDAY = date(2026, 3, 13)


def resolved(text, meeting=MONDAY, events=None):
    return resolve_deadline(text, meeting, events)


# --- immediate -------------------------------------------------------------


@pytest.mark.parametrize("text", ["today", "EOD", "end of day", "tonight", "by close of business"])
def test_same_day_phrases(text):
    assert resolved(text).value == MONDAY


@pytest.mark.parametrize("text", ["tomorrow", "by tomorrow"])
def test_tomorrow(text):
    assert resolved(text).value == date(2026, 3, 10)


def test_asap_is_dated_but_low_confidence():
    """'ASAP' expresses urgency, not a deadline. Dating it is a guess and the
    confidence must say so, or it becomes a false overdue alert."""
    result = resolved("asap")
    assert result.value == MONDAY
    assert result.confidence < 0.7


# --- weekdays ---------------------------------------------------------------


def test_bare_weekday_is_the_next_one():
    assert resolved("by Friday").value == FRIDAY
    assert resolved("Wednesday").value == date(2026, 3, 11)


def test_next_weekday_means_the_following_week():
    """'next Friday' on a Monday means the week after, not four days away."""
    assert resolved("next Friday").value == date(2026, 3, 20)


def test_weekday_abbreviations():
    assert resolved("by weds").value == date(2026, 3, 11)
    assert resolved("thurs").value == date(2026, 3, 12)


def test_same_weekday_as_the_meeting_resolves_to_today():
    assert resolved("by Monday").value == MONDAY


# --- weeks and months -------------------------------------------------------


def test_end_of_week_is_friday():
    assert resolved("end of the week").value == FRIDAY
    assert resolved("EOW").value == FRIDAY


def test_end_of_next_week_beats_the_next_week_rule():
    """Ordering matters: 'end of next week' contains 'next week' as a substring
    and must not be captured by the looser rule."""
    assert resolved("end of next week").value == date(2026, 3, 20)


def test_next_week_alone_is_lower_confidence():
    result = resolved("next week")
    assert result.value == date(2026, 3, 20)
    assert result.confidence < 0.8, "an assumed interpretation should be flagged as such"


def test_end_of_month():
    assert resolved("by end of month").value == date(2026, 3, 31)


def test_in_n_units():
    assert resolved("in 3 days").value == date(2026, 3, 12)
    assert resolved("in two weeks").value == date(2026, 3, 23)
    assert resolved("within 10 days").value == date(2026, 3, 19)


# --- explicit dates ---------------------------------------------------------


def test_ordinal_day_in_the_current_month():
    result = resolved("by the 20th")
    assert result.value == date(2026, 3, 20)
    assert result.method is DeadlineResolution.EXPLICIT


def test_ordinal_day_already_past_rolls_to_next_month():
    """'the 3rd' said on the 20th means next month's 3rd."""
    assert resolved("the 3rd", meeting=date(2026, 3, 20)).value == date(2026, 4, 3)


def test_explicit_date_uses_the_meeting_year_not_today():
    result = resolved("March 14")
    assert result.value == date(2026, 3, 14)


# --- anchored to events -----------------------------------------------------


def test_before_a_known_event_resolves_to_the_day_before():
    result = resolved("before the demo", events={"demo": date(2026, 3, 20)})
    assert result.value == date(2026, 3, 19)
    assert result.method is DeadlineResolution.ANCHORED


def test_on_a_known_event_resolves_to_the_event_day():
    result = resolved("at the retro", events={"retro": date(2026, 3, 18)})
    assert result.value == date(2026, 3, 18)


def test_unknown_event_resolves_to_nothing_rather_than_a_guess():
    """A wrong date is worse than an admitted gap - it silently produces a
    calendar reminder on the wrong day."""
    result = resolved("before the launch")
    assert result.value is None
    assert result.method is DeadlineResolution.ANCHORED
    assert "no known date" in result.note


# --- absent or unparseable --------------------------------------------------


@pytest.mark.parametrize("text", [None, "", "   "])
def test_no_deadline_text(text):
    result = resolved(text)
    assert result.value is None
    assert result.method is DeadlineResolution.NONE
    assert result.confidence == 0.0


def test_unparseable_text_is_reported_not_guessed():
    result = resolved("when the stars align")
    assert result.value is None
    assert "unparsed" in result.note


def test_filler_words_are_stripped():
    assert resolved("by no later than the 20th").value == date(2026, 3, 20)


# --- helpers ----------------------------------------------------------------


def test_next_weekday_helper():
    assert next_weekday(MONDAY, 4) == FRIDAY
    assert next_weekday(MONDAY, 0) == MONDAY
    assert next_weekday(MONDAY, 0, allow_same_day=False) == date(2026, 3, 16)


def test_end_of_week_helper_from_midweek():
    assert end_of_week(date(2026, 3, 11)) == FRIDAY
    assert end_of_week(FRIDAY) == FRIDAY


def test_end_of_month_helper_handles_february_in_a_non_leap_year():
    assert end_of_month(date(2026, 2, 3)) == date(2026, 2, 28)
