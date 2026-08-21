"""Holding a live recording across Streamlit's re-runs.

Streamlit re-executes the entire script on every interaction. That is fine for a
form and hostile to a forty-minute recording: anything held in a local variable
is gone the moment you click something, and starting a recorder inside the
script body would start a *new* one on every rerun.

So the recorder lives in `st.session_state`, which survives reruns, and this
module is the only thing that touches it. Two rules follow:

**Start once, and prove it.** `begin` refuses if a session is already live.
Without that, a double-click - or a rerun triggered while the button's callback
was still running - opens a second pair of audio streams on the same devices,
and the two fight over the microphone.

**Stop is not the same as finish.** Stopping the audio streams is instant;
transcribing and extracting is not. They are separate steps here so the page can
show "stopped, now transcribing" rather than freezing on a click with no
explanation for a minute.

Nothing in here knows about Streamlit. It is a plain object with a clock, which
also means it can be tested without a browser.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date

log = logging.getLogger(__name__)


class AlreadyRecording(RuntimeError):
    """A second recording was started while one was live."""


@dataclass
class RecordingSession:
    """One live capture, and everything needed to finish it afterwards."""

    title: str
    kind: str = "meeting"
    """"meeting" or "lecture" - decides which analysis runs at the end."""

    project_id: str | None = None
    system_only: bool = True
    """Capture the speakers (a video) rather than the microphone (a room)."""

    speed: float = 1.0
    me: str = "You"
    my_email: str = ""
    roster: str = ""

    recorder: object | None = field(default=None, repr=False)
    started_at: float = 0.0
    stopped_at: float = 0.0
    error: str = ""

    @property
    def live(self) -> bool:
        return self.recorder is not None and not self.stopped_at

    @property
    def elapsed_s(self) -> float:
        """Wall clock, not the recorder's captured seconds.

        `captured_seconds` sums both audio channels, so it reads as roughly
        double the real duration - a nine-minute lecture reported as sixteen
        minutes, which looks like half of it went missing.
        """
        if not self.started_at:
            return 0.0
        end = self.stopped_at or time.time()
        return max(0.0, end - self.started_at)

    @property
    def elapsed_label(self) -> str:
        total = int(self.elapsed_s)
        return f"{total // 60:02d}:{total % 60:02d}"

    @property
    def skipped_silent(self) -> int:
        return getattr(self.recorder, "skipped_silent", 0)

    @property
    def queued_chunks(self) -> int:
        queue = getattr(self.recorder, "chunks", None)
        return queue.qsize() if queue is not None else 0

    # -- lifecycle ---------------------------------------------------------

    def begin(self) -> None:
        """Open the audio streams. Raises rather than starting a second one."""
        if self.live:
            raise AlreadyRecording("A recording is already running")

        from quorum.capture.audio import DualRecorder, RecorderConfig

        recorder = DualRecorder(RecorderConfig())
        recorder.start(announced=True)
        self.recorder = recorder
        self.started_at = time.time()
        self.stopped_at = 0.0
        self.error = ""

    def stop(self) -> None:
        """Close the streams. Fast, and separate from processing on purpose."""
        if self.recorder is None or self.stopped_at:
            return
        try:
            self.recorder.stop()
        except Exception as exc:  # noqa: BLE001 - a failed stop must not strand the UI
            log.warning("Stopping the recorder failed (%s)", exc)
            self.error = str(exc)
        self.stopped_at = time.time()

    def chunks(self) -> list:
        """Everything captured, filtered to the channels this kind needs."""
        if self.recorder is None:
            return []
        from quorum.capture.audio import SYSTEM

        drained = self.recorder.drain()
        if self.system_only:
            # A lecture is the speaker, not you. Dropping the microphone halves
            # the audio-seconds spent and removes echo as a concern.
            drained = [c for c in drained if c.channel == SYSTEM]
        return drained

    # -- turning audio into a transcript ------------------------------------

    def transcribe(self, chunks: list):
        """Audio to transcript. Returns `(transcript, notes)` for the caller."""
        from quorum.capture.echo import suppress_echo
        from quorum.capture.speakers import SpeakerRoster, build_transcript
        from quorum.capture.transcribe import WhisperTranscriber, rescale
        from quorum.models import Speaker

        transcriber = WhisperTranscriber()
        segments = transcriber.transcribe_all(chunks)
        echo = None
        if not self.system_only:
            segments, echo = suppress_echo(segments)
        if self.speed != 1.0:
            segments = rescale(segments, self.speed)

        if not segments:
            return None, transcriber.stats, echo

        if self.kind == "lecture":
            people = SpeakerRoster.solo(self.me)
        else:
            others = []
            for index, entry in enumerate(e for e in self.roster.split(",") if e.strip()):
                name, _, email = entry.partition(":")
                others.append(Speaker(
                    id=f"spk_r{index}", display_name=name.strip(),
                    email=email.strip() or None, aliases=[name.strip().split()[0]],
                ))
            people = SpeakerRoster(
                you=Speaker(id="spk_you", display_name=self.me,
                            email=self.my_email or None, aliases=["I", "me"]),
                others=others,
            )

        transcript = build_transcript(
            segments, people, meeting_date=date.today(), title=self.title,
            project_id=self.project_id,
        )
        return transcript, transcriber.stats, echo
