from __future__ import annotations

from datetime import date

import pytest

from quorum.export import Style, parse_time, render, select, stats
from quorum.models import Speaker, Transcript, Utterance


@pytest.fixture
def transcript() -> Transcript:
    priya = Speaker(id="spk_p", display_name="Priya Raghavan", aliases=["Priya"],
                    email="priya@example.com")
    yug = Speaker(id="spk_y", display_name="Yug Verma", aliases=["Yug"])
    lines = [
        (priya.id, "Where are we on the ingestion API?", 0.0, 4.0),
        (yug.id, "Mostly done, I'll send the spec by Friday.", 5.0, 9.0),
        (priya.id, "Good. What about the deadline for the review?", 65.0, 69.0),
        (yug.id, "The review deadline is next Tuesday.", 70.0, 74.0),
        (priya.id, "Right, let's wrap there.", 3665.0, 3668.0),
    ]
    utterances = [
        Utterance(id=f"u{i}", index=i, speaker_id=spk, text=text, start_s=start, end_s=end)
        for i, (spk, text, start, end) in enumerate(lines)
    ]
    return Transcript(
        meeting_id="mtg_1", title="Weekly sync", meeting_date=date(2026, 3, 9),
        speakers=[priya, yug], utterances=utterances, source="live",
    )


# --- time parsing -----------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [("90", 90.0), ("1:30", 90.0), ("12:30", 750.0), ("1:02:03", 3723.0), ("", None),
     (None, None)],
)
def test_parse_time(value, expected):
    assert parse_time(value) == expected


def test_parse_time_rejects_nonsense():
    with pytest.raises(ValueError, match="Cannot read"):
        parse_time("half past two")


# --- styles -----------------------------------------------------------------


def test_speakers_style_is_attributable(transcript):
    text = render(transcript, Style.SPEAKERS)
    assert "Priya Raghavan (00:00): Where are we on the ingestion API?" in text
    assert "Yug Verma (00:05)" in text


def test_timestamped_style_drops_names(transcript):
    text = render(transcript, Style.TIMESTAMPED)
    assert "[00:00] Where are we" in text
    assert "Priya" not in text


def test_plain_style_is_continuous_prose(transcript):
    text = render(transcript, Style.PLAIN)
    assert "\n" not in text
    assert text.startswith("Where are we on the ingestion API?")


def test_markdown_groups_consecutive_lines_by_speaker(transcript):
    text = render(transcript, Style.MARKDOWN)
    assert "# Weekly sync" in text
    assert "**Priya Raghavan**" in text
    assert "> Where are we on the ingestion API?" in text


def test_srt_is_valid_subtitle_format(transcript):
    text = render(transcript, Style.SRT)
    assert text.startswith("1\n00:00:00,000 --> 00:00:04,000\n")
    assert "-->" in text


def test_srt_invents_an_end_time_when_missing():
    """Whisper does not always return an end time, and a subtitle without one
    cannot be displayed."""
    speaker = Speaker(id="s", display_name="A")
    transcript = Transcript(
        meeting_id="m", meeting_date=date(2026, 3, 9), speakers=[speaker],
        utterances=[Utterance(id="u", index=0, speaker_id="s", text="hello", start_s=10.0)],
    )
    text = render(transcript, Style.SRT)
    assert "00:00:10,000 --> 00:00:14,000" in text


def test_hours_appear_only_when_needed(transcript):
    text = render(transcript, Style.TIMESTAMPED)
    assert "[00:00]" in text, "short timestamps stay mm:ss"
    assert "[1:01:05]" in text, "past an hour they gain an hours field"


# --- filtering --------------------------------------------------------------


def test_filter_by_speaker(transcript):
    text = render(transcript, Style.SPEAKERS, speaker="Yug")
    assert "Yug Verma" in text
    assert "Priya" not in text


def test_speaker_filter_accepts_aliases_and_partials(transcript):
    for name in ("Yug", "yug verma", "Verma"):
        assert render(transcript, Style.PLAIN, speaker=name).startswith("Mostly done")


def test_unknown_speaker_lists_who_is_present(transcript):
    with pytest.raises(ValueError, match="Priya Raghavan"):
        render(transcript, Style.PLAIN, speaker="Dmitri")


def test_filter_by_time_window(transcript):
    """A two-hour seminar is unreadable whole; the middle ten minutes is not."""
    text = render(transcript, Style.PLAIN, start_s=60.0, end_s=120.0)
    assert "deadline for the review" in text
    assert "ingestion API" not in text


def test_filter_by_search_text(transcript):
    text = render(transcript, Style.PLAIN, search="deadline")
    assert "deadline" in text
    assert "ingestion API" not in text


def test_search_is_case_insensitive(transcript):
    assert render(transcript, Style.PLAIN, search="DEADLINE")


def test_filters_combine(transcript):
    text = render(transcript, Style.PLAIN, speaker="Yug", search="deadline")
    assert text == "The review deadline is next Tuesday."


def test_no_matches_renders_empty(transcript):
    assert render(transcript, Style.PLAIN, search="quarterly budget") == ""


def test_select_preserves_order(transcript):
    chosen = select(transcript)
    assert [u.index for u in chosen] == [0, 1, 2, 3, 4]


# --- stats ------------------------------------------------------------------


def test_stats_report_speaking_share(transcript):
    summary = stats(transcript)
    assert summary["utterances"] == 5
    assert set(summary["by_speaker"]) == {"Priya Raghavan", "Yug Verma"}
    assert abs(sum(s["share"] for s in summary["by_speaker"].values()) - 1.0) < 0.01


def test_stats_are_ordered_by_volume(transcript):
    names = list(stats(transcript)["by_speaker"])
    assert names[0] == "Priya Raghavan", "the most talkative first"


def test_stats_on_an_empty_transcript():
    empty = Transcript(meeting_id="m", meeting_date=date(2026, 3, 9))
    summary = stats(empty)
    assert summary["utterances"] == 0 and summary["by_speaker"] == {}
