"""Removing your own speakers from your own microphone.

The two-channel design assumes the microphone hears only you. On a laptop with
open speakers that is false: the remote audio comes out of the speakers, crosses
a few centimetres of air, and re-enters the microphone. The same words then
appear on *both* channels, and the echoed copy gets attributed to you.

Observed in the first real recording:

    [1] Remote participant (00:00): you have TOC, table of content, right?...
    [4] Yug Verma        (00:18): You have TOC table of content...

Same words, both channels. Downstream this is worse than untidy - the resolver
maps first-person speech to whoever spoke it, so an echoed *"I'll have that by
Friday"* from someone else becomes a commitment owned by you.

**Headphones remove the problem entirely** and are the real fix; the recorder
now says so. But a tool that only works with headphones is a tool that will
silently produce wrong data the first time someone forgets, so this suppresses
the echo as well.

Suppression happens on transcribed text, not on the waveform. Proper acoustic
echo cancellation means adaptive filtering against a reference signal, which is
sensitive to clock drift between two independently-started streams. Comparing
what was *said* is far more robust: an echo is a near-copy of remote text at
roughly the same moment, and that survives drift, volume differences and
timestamp jitter.

Suppression is deliberately one-directional. Speakers cannot hear the
microphone, so only the microphone channel is ever filtered.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from rapidfuzz import fuzz

from quorum.capture.audio import MIC, SYSTEM
from quorum.capture.transcribe import TranscriptSegment

log = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = 70.0
"""How alike two texts must be to call one an echo. Transcription of the quieter
echoed copy is imperfect, so exact matching would miss most of them."""

TIME_WINDOW_S = 40.0
"""How far apart the two copies may sit. Generous, because the channels are
chunked independently and their timestamps drift relative to each other."""

MIN_ECHO_CHARS = 25
"""Short utterances are never treated as echoes. "Yeah" and "okay" are common
on both channels and matching them would delete genuine speech - and losing a
real acknowledgement costs more than keeping a duplicated filler word."""


@dataclass
class EchoReport:
    removed: int = 0
    kept: int = 0
    examples: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.examples is None:
            self.examples = []

    @property
    def echo_rate(self) -> float:
        total = self.removed + self.kept
        return self.removed / total if total else 0.0

    @property
    def likely_no_headphones(self) -> bool:
        """A high echo rate means the microphone is hearing the speakers."""
        return self.echo_rate > 0.25

    def as_dict(self) -> dict:
        return {
            "removed": self.removed,
            "kept": self.kept,
            "echo_rate": round(self.echo_rate, 4),
            "likely_no_headphones": self.likely_no_headphones,
        }


def suppress_echo(
    segments: list[TranscriptSegment],
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    window_s: float = TIME_WINDOW_S,
    min_chars: int = MIN_ECHO_CHARS,
) -> tuple[list[TranscriptSegment], EchoReport]:
    """Drop microphone segments that merely repeat remote audio.

    The system channel is always kept: it is the clean original. Only the
    microphone's echoed copy is discarded.
    """
    report = EchoReport()
    remote = [s for s in segments if s.channel == SYSTEM]
    if not remote:
        return list(segments), report

    kept: list[TranscriptSegment] = []
    for segment in segments:
        if segment.channel != MIC:
            kept.append(segment)
            continue

        text = segment.text.strip()
        if len(text) < min_chars:
            kept.append(segment)
            report.kept += 1
            continue

        if _echoes_any(text, segment, remote, similarity_threshold, window_s):
            report.removed += 1
            if len(report.examples) < 3:
                report.examples.append(text[:80])
            log.debug("Dropped echoed mic segment: %s", text[:60])
        else:
            kept.append(segment)
            report.kept += 1

    if report.likely_no_headphones:
        log.warning(
            "%.0f%% of microphone speech was echoed system audio - use headphones",
            report.echo_rate * 100,
        )
    return kept, report


def _echoes_any(
    text: str,
    segment: TranscriptSegment,
    remote: list[TranscriptSegment],
    threshold: float,
    window_s: float,
) -> bool:
    lowered = text.lower()
    for other in remote:
        if abs(other.start_s - segment.start_s) > window_s:
            continue
        # token_set_ratio ignores word order and repetition, which matters
        # because the echoed copy is quieter and transcribes less cleanly - words
        # get dropped, merged or misheard.
        if fuzz.token_set_ratio(lowered, other.text.lower()) >= threshold:
            return True
        if fuzz.partial_ratio(lowered, other.text.lower()) >= threshold + 10:
            return True
    return False
