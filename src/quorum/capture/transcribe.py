"""Speech to text via Groq's free Whisper tier.

28,800 audio-seconds a day, free — eight hours of meetings. That single quota is
what makes live capture viable on a laptop that cannot host a Whisper model
locally without swapping.

The budget is spent carefully: silent chunks are dropped before they reach here
(most of any meeting is one side not talking), and the recorder already
downsamples to 16 kHz mono, so nothing uploads more data than Whisper uses.
"""

from __future__ import annotations

import io
import logging
import time
import wave
from dataclasses import dataclass, field

import numpy as np

from quorum.capture.audio import TARGET_RATE, AudioChunk
from quorum.config import get_settings

log = logging.getLogger(__name__)

MODEL = "whisper-large-v3-turbo"

DAILY_AUDIO_SECONDS = 28_800
HOURLY_AUDIO_SECONDS = 7_200


@dataclass
class TranscriptSegment:
    channel: str
    start_s: float
    end_s: float
    text: str

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)


def rescale(segments: list[TranscriptSegment], factor: float) -> list[TranscriptSegment]:
    """Map capture-clock times onto source-media times, at a constant rate.

    Recording a video played at 2x captures nine minutes of audio for an
    eighteen-minute lecture, and every timestamp then points at half its real
    position - a note saying a concept was explained at 06:24 sends you to 06:24
    in a video where it happens at 12:48.

    **This only holds if the whole thing was watched straight through at one
    speed.** Capture time is monotonic; video position is not. Skipping forward
    advances the video while the clock runs normally, changing speed part-way
    bends the mapping at a point nothing here can see, and rewatching a section
    puts the *same* video position at two different capture times - so the
    mapping stops being a function at all, and no scalar can express it.

    A single factor is therefore a convenience for the common case, not a
    general solution. When the session was not linear the honest answer is that
    these are positions in the recording, and `quorum transcript --search` is
    the way to find a moment again.

    The text is untouched - only when things were said.
    """
    if factor <= 0 or factor == 1.0:
        return segments
    return [
        TranscriptSegment(
            channel=segment.channel,
            start_s=segment.start_s * factor,
            end_s=segment.end_s * factor,
            text=segment.text,
        )
        for segment in segments
    ]


@dataclass
class TranscriptionStats:
    chunks: int = 0
    audio_seconds: float = 0.0
    api_calls: int = 0
    failures: int = 0
    segments: int = 0

    @property
    def daily_budget_used(self) -> float:
        return self.audio_seconds / DAILY_AUDIO_SECONDS

    def as_dict(self) -> dict:
        return {
            "chunks": self.chunks,
            "audio_seconds": round(self.audio_seconds, 1),
            "api_calls": self.api_calls,
            "failures": self.failures,
            "segments": self.segments,
            "daily_budget_used": round(self.daily_budget_used, 4),
        }


def to_wav_bytes(samples: np.ndarray, rate: int = TARGET_RATE) -> bytes:
    """In-memory WAV. Avoids touching disk for audio we do not keep."""
    buffer = io.BytesIO()
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm.tobytes())
    return buffer.getvalue()


class WhisperTranscriber:
    def __init__(self, client=None, model: str = MODEL, max_retries: int = 2) -> None:
        self._client = client
        self.model = model
        self.max_retries = max_retries
        self.stats = TranscriptionStats()

    @property
    def client(self):
        if self._client is None:
            from groq import Groq

            settings = get_settings()
            if not settings.groq_api_key:
                raise RuntimeError("GROQ_API_KEY is required for transcription")
            self._client = Groq(api_key=settings.groq_api_key)
        return self._client

    def transcribe_chunk(self, chunk: AudioChunk) -> list[TranscriptSegment]:
        """Transcribe one chunk, returning timestamped segments.

        Timestamps come back relative to the chunk, so they are shifted by the
        chunk's own offset. Without that every chunk would claim to start at
        zero and the whole meeting would collapse onto itself.
        """
        self.stats.chunks += 1
        self.stats.audio_seconds += chunk.duration_s
        audio = to_wav_bytes(chunk.samples)

        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.audio.transcriptions.create(
                    file=(f"{chunk.channel}.wav", audio),
                    model=self.model,
                    response_format="verbose_json",
                    timestamp_granularities=["segment"],
                    temperature=0.0,
                )
                self.stats.api_calls += 1
                return self._to_segments(response, chunk)
            except Exception as exc:  # noqa: BLE001 - SDK raises varied types
                if attempt >= self.max_retries:
                    log.warning("Transcription failed for %s chunk: %s", chunk.channel, exc)
                    self.stats.failures += 1
                    return []
                # Rate limits here are audio-seconds per hour, so backing off
                # briefly is usually enough to clear the window.
                time.sleep(2.0 * (attempt + 1))
        return []

    def _to_segments(self, response, chunk: AudioChunk) -> list[TranscriptSegment]:
        raw = getattr(response, "segments", None) or []
        segments: list[TranscriptSegment] = []

        for item in raw:
            text = (item.get("text") if isinstance(item, dict) else getattr(item, "text", "")) or ""
            if not text.strip():
                continue
            start = float(
                item.get("start") if isinstance(item, dict) else getattr(item, "start", 0.0) or 0.0
            )
            end = float(
                item.get("end") if isinstance(item, dict) else getattr(item, "end", 0.0) or 0.0
            )
            segments.append(
                TranscriptSegment(
                    channel=chunk.channel,
                    start_s=chunk.start_s + start,
                    end_s=chunk.start_s + max(end, start),
                    text=text.strip(),
                )
            )

        if not segments:
            # Some responses carry only a flat `text` field with no segments.
            whole = (getattr(response, "text", "") or "").strip()
            if whole:
                segments.append(
                    TranscriptSegment(
                        channel=chunk.channel, start_s=chunk.start_s,
                        end_s=chunk.end_s, text=whole,
                    )
                )

        self.stats.segments += len(segments)
        return segments

    def transcribe_all(self, chunks: list[AudioChunk]) -> list[TranscriptSegment]:
        segments: list[TranscriptSegment] = []
        for chunk in chunks:
            segments.extend(self.transcribe_chunk(chunk))
        segments.sort(key=lambda s: s.start_s)
        return segments
