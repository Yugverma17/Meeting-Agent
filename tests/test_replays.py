"""Detecting the parts of a lecture that were played more than once.

The hard case is not detection, it is **precision**. Lecturers repeat
themselves constantly - one real recording says "this is a substring where C is
the last character" four times in a row - and a section claiming those were
struggles would be noise, which is worse than no section at all.

So the tests that matter here are mostly the negatives.
"""

from __future__ import annotations

from datetime import date

from quorum.analysis.replays import MIN_GAP_S, Replay, find_replays
from quorum.models import Speaker, Transcript, Utterance

SPEAKER = Speaker(id="spk_lecturer", display_name="Lecturer")

# Roughly sixty words each - about twenty seconds of speech, which is the
# scale of a real rewind. Anything shorter is below the detector's floor by
# design, because thirty words repeated is rhetoric far more often than it is a
# replay.
BLOCK_A = (
    "so the minimal window ending at this character is determined by the most "
    "recent occurrence of each of the three characters and you take the smallest "
    "of those last seen indices as the start of the window and that gives you "
    "the shortest stretch ending here which still contains every character you "
    "were asked to find in the first place"
)
BLOCK_B = (
    "now the time complexity of the brute force approach is order n squared "
    "because we enumerate every possible substring and check each one for the "
    "presence of all three required characters which means the outer loop runs "
    "n times and the inner loop runs n times again so the total work grows with "
    "the square of the input length as you would expect"
)
BLOCK_C = (
    "let us look at a worked example with the string b b a c and count how many "
    "substrings of it contain all three of the characters a and b and c together "
    "starting from the leftmost position and expanding one character at a time "
    "until we reach the end of the string and have counted every candidate "
    "exactly once without missing any of them"
)


def transcript_of(blocks: list[str], seconds_each: float = 30.0) -> Transcript:
    utterances = []
    clock = 0.0
    for index, text in enumerate(blocks):
        utterances.append(Utterance(
            id=f"u{index}", index=index, speaker_id=SPEAKER.id, text=text,
            start_s=clock, end_s=clock + seconds_each,
        ))
        clock += seconds_each
    return Transcript(
        meeting_id="mtg_x", title="Lecture", meeting_date=date(2026, 8, 15),
        speakers=[SPEAKER], utterances=utterances, source="live",
    )


# --- the negatives, which matter most ---------------------------------------


def test_a_lecture_watched_straight_through_has_no_replays():
    assert find_replays(transcript_of([BLOCK_A, BLOCK_B, BLOCK_C])) == []


def test_a_repeated_sentence_is_not_a_replay():
    """The failure that would make this feature worthless. A speaker saying the
    same line four times is rhetoric, not a rewind - it is short, and a replay
    reproduces a long contiguous stretch."""
    line = "this is a substring where c is the last character"
    blocks = [
        f"{BLOCK_A} {line}",
        f"{line} {line} and again {line} {BLOCK_B}",
        f"{line} {BLOCK_C}",
    ]

    assert find_replays(transcript_of(blocks)) == []


def test_a_speaker_restating_an_idea_in_other_words_is_not_a_replay():
    restated = (
        "in other words you look at where each character last appeared and the "
        "earliest of those positions tells you where the window begins"
    )

    assert find_replays(transcript_of([BLOCK_A, restated, BLOCK_C])) == []


def test_two_occurrences_too_close_together_are_not_counted():
    """Adjacent windows overlap by design, so without a separation requirement
    the detector would recognise its own overlap as a repeat."""
    close = transcript_of([BLOCK_A, BLOCK_A], seconds_each=MIN_GAP_S / 4)

    assert find_replays(close) == []


def test_a_transcript_too_short_to_judge_returns_nothing():
    assert find_replays(transcript_of(["hello everyone"])) == []
    assert find_replays(Transcript(meeting_id="m", meeting_date=date(2026, 8, 15))) == []


# --- the positives ----------------------------------------------------------


def test_a_replayed_stretch_is_found():
    found = find_replays(transcript_of([BLOCK_A, BLOCK_B, BLOCK_A, BLOCK_C]))

    assert len(found) == 1
    assert found[0].count == 2
    assert "minimal window" in found[0].text


def test_the_count_is_how_many_times_it_was_played():
    """The count is the entire claim. An earlier version derived it from how
    many alignments happened to be found and reported a stretch watched three
    times as seven."""
    found = find_replays(transcript_of([BLOCK_A, BLOCK_B, BLOCK_A, BLOCK_C, BLOCK_A]))

    assert len(found) == 1, "one stretch, however many alignments found it"
    assert found[0].count == 3


def test_the_most_replayed_stretch_comes_first():
    blocks = [BLOCK_A, BLOCK_B, BLOCK_A, BLOCK_C, BLOCK_A, BLOCK_B]
    found = find_replays(transcript_of(blocks))

    assert [r.count for r in found] == sorted([r.count for r in found], reverse=True)
    assert found[0].count == 3


def test_a_replay_reports_when_it_was_first_played():
    """Approximate by design: windows overlap block boundaries, so the reported
    moment can sit a few seconds before the stretch itself. Close enough to
    scrub to; not a precise offset, and not claimed as one."""
    found = find_replays(transcript_of([BLOCK_B, BLOCK_A, BLOCK_C, BLOCK_A]))

    assert 25.0 <= found[0].times[0] <= 35.0, "the first play, not the rewind"
    assert found[0].times[-1] > 60.0, "the later play is recorded too"


# --- rendering --------------------------------------------------------------


def test_replays_appear_in_the_notes():
    from quorum.analysis.lecture import LectureNotes

    notes = LectureNotes(title="Substrings")
    notes.replays = [Replay(text="the minimal window formula", times=[10.0, 200.0, 400.0])]

    markdown = notes.as_markdown()

    assert "## You replayed these" in markdown
    assert "**3x**" in markdown
    assert "the minimal window formula" in markdown


def test_notes_without_replays_have_no_such_section():
    from quorum.analysis.lecture import LectureNotes

    assert "You replayed" not in LectureNotes(title="x", summary="y").as_markdown()


def test_a_long_replay_is_summarised_rather_than_dumped():
    replay = Replay(text=" ".join(["word"] * 400), times=[0.0, 100.0])

    assert len(replay.summary()) <= 161
    assert replay.summary().endswith("…")
