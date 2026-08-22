"""The Streamlit interface.

Two kinds of test here, and the first kind is the one that matters.

`RecordingSession` is plain logic with a clock, and it holds the rules that
Streamlit's re-run model makes easy to get wrong: start once, stop is separate
from finish, and elapsed time is the wall clock rather than the recorder's
`captured_seconds` (which sums both channels and reads as double).

The second is a smoke test that actually executes the page. A Streamlit script
only fails at render time - a plain HTTP request gets the shell and reports 200
however broken the script is - so nothing short of running it catches a typo in
a dict key. The first version of this app read `found['system']` where the
recorder returns `found['loopback']`, and only `AppTest` found it.
"""

from __future__ import annotations

import time

import pytest

from quorum.ui.session import AlreadyRecording, RecordingSession


class FakeRecorder:
    def __init__(self) -> None:
        self.started = False
        self.stopped = False
        self.skipped_silent = 3
        self.drained: list = []

    def start(self, announced: bool = True) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def drain(self):
        return list(self.drained)


def live_session(**kwargs) -> RecordingSession:
    session = RecordingSession(title="Weekly sync", **kwargs)
    session.recorder = FakeRecorder()
    session.started_at = time.time()
    return session


# --- the rules Streamlit's rerun model makes easy to break -------------------


def test_a_second_recording_cannot_start_while_one_is_live():
    """A double-click, or a rerun fired while the callback was still running,
    would otherwise open a second pair of streams on the same devices - and the
    two then fight over the microphone."""
    session = live_session()

    with pytest.raises(AlreadyRecording):
        session.begin()


def test_elapsed_is_the_wall_clock_not_the_captured_seconds():
    """`captured_seconds` sums the microphone and loopback streams, so it reads
    as roughly double: a nine-minute lecture reported as sixteen minutes, which
    looks like half of it went missing."""
    session = live_session()
    session.started_at = time.time() - 65

    assert 64 <= session.elapsed_s <= 67
    assert session.elapsed_label == "01:05"


def test_the_clock_stops_when_the_recording_does():
    session = live_session()
    session.started_at = time.time() - 30
    session.stop()
    frozen = session.elapsed_s
    time.sleep(0.05)

    assert session.elapsed_s == frozen


def test_stopping_twice_is_harmless():
    session = live_session()
    session.stop()
    first = session.stopped_at
    session.stop()

    assert session.stopped_at == first


def test_a_failure_to_stop_does_not_strand_the_page():
    """The session must leave the live state even if the audio layer errors, or
    the UI shows a Stop button that can never be satisfied."""
    class Stubborn(FakeRecorder):
        def stop(self):
            raise RuntimeError("device already closed")

    session = live_session()
    session.recorder = Stubborn()
    session.stop()

    assert not session.live
    assert "device already closed" in session.error


def test_a_session_that_never_started_reports_no_time():
    assert RecordingSession(title="x").elapsed_s == 0.0
    assert RecordingSession(title="x").elapsed_label == "00:00"


# --- which audio is kept ------------------------------------------------------


class Chunk:
    def __init__(self, channel: str) -> None:
        self.channel = channel


def test_a_video_lecture_keeps_only_the_speaker_channel():
    from quorum.capture.audio import MIC, SYSTEM

    session = live_session(system_only=True)
    session.recorder.drained = [Chunk(SYSTEM), Chunk(MIC), Chunk(SYSTEM)]

    assert [c.channel for c in session.chunks()] == [SYSTEM, SYSTEM]


def test_a_meeting_keeps_both_channels():
    from quorum.capture.audio import MIC, SYSTEM

    session = live_session(system_only=False)
    session.recorder.drained = [Chunk(SYSTEM), Chunk(MIC)]

    assert len(session.chunks()) == 2


def test_no_recorder_means_no_chunks():
    assert RecordingSession(title="x").chunks() == []


# --- the page actually renders ------------------------------------------------


@pytest.mark.slow
def test_the_page_renders_without_raising(tmp_path, monkeypatch):
    """A Streamlit script fails at render time, not import time. A plain request
    for the page returns 200 however broken it is, so only executing it catches
    a wrong dict key - which is exactly what this found on its first run."""
    import pathlib

    import quorum.ui
    from streamlit.testing.v1 import AppTest

    # Located from the installed package, not from the working directory -
    # pytest does not promise to run from the repository root.
    script = pathlib.Path(quorum.ui.__file__).parent / "app.py"

    monkeypatch.setenv("QUORUM_LOG_LEVEL", "ERROR")
    page = AppTest.from_file(str(script), default_timeout=120)
    page.run()

    assert not page.exception, [str(e.value) for e in page.exception]
    assert [tab.label for tab in page.tabs][:1] == ["Record"]


# --- outcomes must outlive the rerun that erases the page --------------------


def test_a_recording_outcome_survives_a_rerun(monkeypatch):
    """Streamlit draws a message once and forgets it. A recording finished, said
    why it had failed, and the message was wiped before it could be read - the
    user saw an empty Library and a Record tab offering a fresh recording, with
    no sign anything had happened."""
    import streamlit as st

    from quorum.ui import app

    store: dict = {}
    monkeypatch.setattr(st, "session_state", store, raising=False)

    app._remember("error", "No audio was captured.", "Check the microphone.")

    assert store["last_outcome"]["state"] == "error"
    assert "No audio" in store["last_outcome"]["headline"]
    assert store["last_outcome"]["detail"]


def test_the_newest_outcome_replaces_the_last(monkeypatch):
    import streamlit as st

    from quorum.ui import app

    store: dict = {}
    monkeypatch.setattr(st, "session_state", store, raising=False)

    app._remember("error", "first")
    app._remember("ok", "second")

    assert store["last_outcome"]["headline"] == "second"


def test_every_ending_of_a_recording_reports_something():
    """Each early return in `_finish_recording` is a path a user can hit after
    waiting two minutes. None of them may end silently."""
    import inspect

    from quorum.ui import app

    source = inspect.getsource(app._finish_recording)
    endings = [line for line in source.splitlines() if line.strip() == "return"]

    assert len(endings) >= 3
    assert source.count("_remember(") >= len(endings)


@pytest.mark.slow
def test_the_page_renders_with_an_outcome_showing(monkeypatch):
    """The smoke test above passes with no outcome stored, because
    `_show_last_outcome` returns before creating anything. That hid a real
    crash: the outcome is drawn on Record *and* Library, Streamlit renders every
    tab in one pass, and two buttons sharing a key replaces the whole page with
    a traceback. Seeding the state is what exercises it."""
    import pathlib

    import quorum.ui
    from streamlit.testing.v1 import AppTest

    script = pathlib.Path(quorum.ui.__file__).parent / "app.py"
    monkeypatch.setenv("QUORUM_LOG_LEVEL", "ERROR")

    page = AppTest.from_file(str(script), default_timeout=120)
    page.session_state["last_outcome"] = {
        "state": "error", "headline": "No audio was captured.", "detail": "Check the mode.",
    }
    page.run()

    assert not page.exception, [str(e.value) for e in page.exception]
    dismiss = [b for b in page.button if b.label == "Dismiss"]
    assert len(dismiss) == 2, "shown on Record and Library, with distinct keys"


def test_no_two_widgets_share_a_key():
    """A duplicate key is a hard error that replaces the page with a traceback,
    and it only appears once the relevant branch renders - so it is worth
    checking statically rather than waiting to hit it."""
    import ast
    import pathlib
    from collections import Counter

    import quorum.ui

    source = pathlib.Path(quorum.ui.__file__).parent / "app.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))

    literal_keys = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg == "key" and isinstance(keyword.value, ast.Constant):
                literal_keys.append(keyword.value.value)

    repeated = [key for key, count in Counter(literal_keys).items() if count > 1]
    assert repeated == [], f"these literal keys appear more than once: {repeated}"
