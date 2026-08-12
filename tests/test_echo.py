"""Echo suppression tests.

The fixtures are taken verbatim from the first real recording, where a laptop
microphone picked up the laptop's own speakers and the remote audio was
attributed to the user.
"""

from __future__ import annotations

import pytest

from quorum.capture.audio import MIC, SYSTEM
from quorum.capture.echo import suppress_echo
from quorum.capture.transcribe import TranscriptSegment

# Observed output, lightly trimmed. The mic copy is quieter and transcribes less
# cleanly - note "index C" where the system channel heard "index table" - which
# is exactly why exact matching would not catch it.
REAL_REMOTE = (
    "you have TOC, table of content, right? Table of content is just like index "
    "table, right? So let's say there is one introduction. The first chapter is "
    "something called as introduction. The page number is mentioned P1, okay?"
)
REAL_ECHO = "You have TOC table of content. Table of content is just like index C."
REAL_SPEECH = "I will finish the ingestion API spec and send it to Priya by Friday."
REAL_MUSING = "We should probably think about rate limiting at some point."


def seg(channel, start, text, end=None):
    return TranscriptSegment(
        channel=channel, start_s=start, end_s=end if end is not None else start + 5.0, text=text
    )


# --- the observed failure ---------------------------------------------------


def test_the_real_recording_echo_is_removed():
    """Regression for the first live run: remote audio arrived on both channels
    and the echoed copy was labelled as the user."""
    kept, report = suppress_echo([
        seg(MIC, 0.0, REAL_SPEECH),
        seg(SYSTEM, 0.0, REAL_REMOTE),
        seg(MIC, 7.0, REAL_MUSING),
        seg(MIC, 18.0, REAL_ECHO),
    ])

    texts = [s.text for s in kept]
    assert REAL_SPEECH in texts
    assert REAL_MUSING in texts
    assert REAL_ECHO not in texts, "the echoed copy must not survive as user speech"
    assert report.removed == 1


def test_the_users_own_words_are_never_removed():
    """The cost of a false positive here is losing a real commitment."""
    kept, _ = suppress_echo([
        seg(MIC, 0.0, REAL_SPEECH),
        seg(SYSTEM, 2.0, REAL_REMOTE),
    ])
    assert [s.text for s in kept if s.channel == MIC] == [REAL_SPEECH]


def test_system_channel_is_never_filtered():
    """Speakers cannot hear the microphone, so suppression is one-directional."""
    kept, report = suppress_echo([
        seg(SYSTEM, 0.0, REAL_REMOTE),
        seg(SYSTEM, 6.0, REAL_REMOTE),
    ])
    assert len(kept) == 2 and report.removed == 0


# --- thresholds and edges ---------------------------------------------------


def test_imperfect_transcription_of_the_echo_still_matches():
    """The echoed copy is quieter, so words get dropped or misheard. Exact
    matching would miss nearly all of them."""
    kept, report = suppress_echo([
        seg(SYSTEM, 0.0, "So whenever you have this kind of table of content, right?"),
        seg(MIC, 3.0, "So whenever you have this kind of table of content right"),
    ])
    assert report.removed == 1
    assert len(kept) == 1


def test_short_utterances_are_kept_even_if_they_match():
    """"Yeah" and "okay" appear on both channels constantly. Deleting them would
    remove genuine acknowledgements, and a duplicated filler word costs nothing."""
    kept, report = suppress_echo([
        seg(SYSTEM, 0.0, "Yeah okay right"),
        seg(MIC, 1.0, "Yeah okay"),
    ])
    assert report.removed == 0
    assert len(kept) == 2


def test_matching_text_far_apart_in_time_is_not_an_echo():
    """The same topic can genuinely recur later in a meeting."""
    kept, report = suppress_echo([
        seg(SYSTEM, 0.0, REAL_REMOTE),
        seg(MIC, 600.0, REAL_ECHO),
    ])
    assert report.removed == 0
    assert len(kept) == 2


def test_no_system_audio_means_nothing_to_suppress():
    kept, report = suppress_echo([seg(MIC, 0.0, REAL_SPEECH)])
    assert len(kept) == 1 and report.removed == 0


def test_empty_input():
    kept, report = suppress_echo([])
    assert kept == [] and report.removed == 0


# --- the headphones signal --------------------------------------------------


def test_heavy_echo_is_flagged_as_a_headphones_problem():
    """A high echo rate is diagnostic: the user is on open speakers."""
    segments = [seg(SYSTEM, i * 6.0, REAL_REMOTE) for i in range(4)]
    segments += [seg(MIC, i * 6.0 + 1.0, REAL_ECHO) for i in range(4)]
    segments.append(seg(MIC, 30.0, REAL_SPEECH))

    _, report = suppress_echo(segments)
    assert report.removed == 4
    assert report.likely_no_headphones


def test_clean_recording_is_not_flagged():
    _, report = suppress_echo([
        seg(SYSTEM, 0.0, REAL_REMOTE),
        seg(MIC, 1.0, REAL_SPEECH),
        seg(MIC, 8.0, REAL_MUSING),
    ])
    assert not report.likely_no_headphones
    assert report.echo_rate == 0.0


def test_report_serialises():
    _, report = suppress_echo([
        seg(SYSTEM, 0.0, REAL_REMOTE),
        seg(MIC, 2.0, REAL_ECHO),
    ])
    payload = report.as_dict()
    assert payload["removed"] == 1
    assert 0.0 <= payload["echo_rate"] <= 1.0


def test_examples_are_captured_for_diagnosis():
    _, report = suppress_echo([
        seg(SYSTEM, 0.0, REAL_REMOTE),
        seg(MIC, 2.0, REAL_ECHO),
    ])
    assert report.examples and REAL_ECHO[:20] in report.examples[0]


@pytest.mark.parametrize("threshold,expected_removed", [(50.0, 1), (99.0, 0)])
def test_threshold_is_configurable(threshold, expected_removed):
    _, report = suppress_echo(
        [seg(SYSTEM, 0.0, REAL_REMOTE), seg(MIC, 2.0, REAL_ECHO)],
        similarity_threshold=threshold,
    )
    assert report.removed == expected_removed
