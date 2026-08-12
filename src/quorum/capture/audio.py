"""Capturing a live meeting from the machine, without joining it.

**Why not a bot in the call.** Google's Meet Media API only works if every
participant is enrolled in its developer preview, which makes it useless for a
real meeting. Zoom's RTMS needs account credits. A headless-browser bot that
joins as a participant is fragile, ToS-grey, and would want more RAM than this
laptop has spare.

Capturing the machine's own audio sidesteps all of it. It works identically for
Meet, Zoom, Teams or a phone call on speaker, needs no API, no bot in the
participant list, and no permission from the platform.

**Two channels, and that is the trick.** The microphone is you. The WASAPI
loopback - what Windows is about to send to the speakers - is everyone else. So
speaker separation between you and the room is free and exact, with no
diarisation model, which matters on a machine that cannot host one.

What it does *not* give you is separation *among* the remote participants; they
share one channel. That is handled downstream in `speakers.py`, and it is the
weakest link in the live path.

**Consent.** Recording other people has legal requirements that vary by
jurisdiction, and two-party-consent rules are common. `DualRecorder` announces
itself on start and refuses to run with `announced=False` unless explicitly
overridden.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

TARGET_RATE = 16_000
"""Whisper works at 16 kHz mono. Sending 48 kHz stereo wastes upload bandwidth
and the free tier's audio-seconds budget without improving accuracy."""

MIC = "mic"
SYSTEM = "system"


@dataclass
class AudioChunk:
    """One recorded slice of a single channel."""

    channel: str
    start_s: float
    duration_s: float
    samples: np.ndarray
    path: Path | None = None

    @property
    def end_s(self) -> float:
        return self.start_s + self.duration_s


@dataclass
class RecorderConfig:
    chunk_seconds: float = 30.0
    """Long enough that Whisper has sentence context, short enough that a
    30-second chunk can be transcribed while the next one records."""

    silence_rms: float = 0.004
    """Below this a chunk is treated as silence and never uploaded. Most of any
    meeting is one side not talking, so this is the single biggest saving
    against the 28,800 audio-seconds/day free budget."""

    device_rate: int = 48_000
    channels: int = 2
    frames_per_buffer: int = 1024
    output_dir: Path | None = None
    keep_wav: bool = False


# --- signal helpers (pure, and therefore testable without a sound card) -----


def downmix_to_mono(samples: np.ndarray, channels: int) -> np.ndarray:
    """Average interleaved channels into one."""
    if channels <= 1:
        return samples.astype(np.float32)
    usable = (len(samples) // channels) * channels
    return samples[:usable].astype(np.float32).reshape(-1, channels).mean(axis=1)


def resample_to_16k(samples: np.ndarray, source_rate: int) -> np.ndarray:
    """Downsample to 16 kHz.

    When the ratio is a whole number (48k -> 16k is 3) it averages each group of
    samples rather than picking every third one. Naive decimation aliases higher
    frequencies down into the speech band, which sounds fine to a human but
    measurably degrades transcription.
    """
    if source_rate == TARGET_RATE or len(samples) == 0:
        return samples.astype(np.float32)

    if source_rate % TARGET_RATE == 0:
        factor = source_rate // TARGET_RATE
        usable = (len(samples) // factor) * factor
        if usable == 0:
            return np.zeros(0, dtype=np.float32)
        return samples[:usable].reshape(-1, factor).mean(axis=1).astype(np.float32)

    target_length = int(len(samples) * TARGET_RATE / source_rate)
    if target_length <= 0:
        return np.zeros(0, dtype=np.float32)
    positions = np.linspace(0, len(samples) - 1, target_length)
    return np.interp(positions, np.arange(len(samples)), samples).astype(np.float32)


def rms(samples: np.ndarray) -> float:
    if len(samples) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))


def is_silent(samples: np.ndarray, threshold: float) -> bool:
    return rms(samples) < threshold


def write_wav(path: Path, samples: np.ndarray, rate: int = TARGET_RATE) -> Path:
    """Write mono 16-bit PCM. Groq accepts WAV directly, so no ffmpeg needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm.tobytes())
    return path


# --- recording --------------------------------------------------------------


class DualRecorder:
    """Records the microphone and the system output at the same time.

    Each channel runs on its own thread and pushes finished chunks onto a shared
    queue, so transcription of chunk N overlaps with the recording of chunk N+1
    instead of waiting for the meeting to end.
    """

    def __init__(self, config: RecorderConfig | None = None) -> None:
        self.config = config or RecorderConfig()
        self.chunks: queue.Queue[AudioChunk] = queue.Queue()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._started_at: float = 0.0
        self.skipped_silent = 0
        self.captured_seconds = 0.0

    # -- device discovery ---------------------------------------------------

    @staticmethod
    def devices() -> dict:
        """Resolve the default mic and the speaker loopback."""
        import pyaudiowpatch as pyaudio

        audio = pyaudio.PyAudio()
        try:
            wasapi = audio.get_host_api_info_by_type(pyaudio.paWASAPI)
            speakers = audio.get_device_info_by_index(wasapi["defaultOutputDevice"])
            loopback = None
            for candidate in audio.get_loopback_device_info_generator():
                if speakers["name"] in candidate["name"]:
                    loopback = candidate
                    break
            if loopback is None:
                loopback = next(audio.get_loopback_device_info_generator(), None)
            mic = audio.get_device_info_by_index(wasapi["defaultInputDevice"])
            return {"mic": mic, "loopback": loopback}
        finally:
            audio.terminate()

    # -- lifecycle ----------------------------------------------------------

    def start(self, announced: bool = True) -> None:
        """Begin recording both channels.

        `announced` must be True unless the caller has taken responsibility for
        telling participants. Recording people without their knowledge is a
        legal problem in many places, not merely an impolite one.
        """
        if not announced:
            raise PermissionError(
                "Refusing to record without announcement. Tell participants they are "
                "being recorded, then pass announced=True."
            )

        import pyaudiowpatch as pyaudio

        devices = self.devices()
        if devices["loopback"] is None:
            raise RuntimeError(
                "No WASAPI loopback device found. System audio cannot be captured; "
                "only your microphone would be recorded."
            )

        self._stop.clear()
        self._started_at = time.time()
        audio = pyaudio.PyAudio()

        for channel, info in ((MIC, devices["mic"]), (SYSTEM, devices["loopback"])):
            thread = threading.Thread(
                target=self._record_channel, args=(audio, channel, info), daemon=True,
                name=f"quorum-{channel}",
            )
            thread.start()
            self._threads.append(thread)

        log.info("Recording started: mic + system audio")

    def stop(self) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=5.0)
        self._threads.clear()

    def _record_channel(self, audio, channel: str, info: dict) -> None:
        cfg = self.config
        rate = int(info.get("defaultSampleRate", cfg.device_rate))
        channels = min(int(info.get("maxInputChannels", cfg.channels)), 2)
        frames_per_chunk = int(rate * cfg.chunk_seconds)

        try:
            stream = audio.open(
                format=__import__("pyaudiowpatch").paFloat32,
                channels=channels, rate=rate, input=True,
                input_device_index=info["index"], frames_per_buffer=cfg.frames_per_buffer,
            )
        except Exception as exc:  # noqa: BLE001 - device errors vary wildly
            log.error("Could not open %s device: %s", channel, exc)
            return

        buffer: list[np.ndarray] = []
        collected = 0
        chunk_start = 0.0

        try:
            while not self._stop.is_set():
                try:
                    raw = stream.read(cfg.frames_per_buffer, exception_on_overflow=False)
                except OSError as exc:
                    log.warning("%s read error: %s", channel, exc)
                    continue

                block = np.frombuffer(raw, dtype=np.float32)
                buffer.append(block)
                collected += len(block) // max(channels, 1)

                if collected >= frames_per_chunk:
                    self._emit(channel, buffer, rate, channels, chunk_start)
                    chunk_start += cfg.chunk_seconds
                    buffer, collected = [], 0

            if buffer:
                self._emit(channel, buffer, rate, channels, chunk_start)
        finally:
            stream.stop_stream()
            stream.close()

    def _emit(
        self, channel: str, buffer: list[np.ndarray], rate: int, channels: int, start: float
    ) -> None:
        joined = np.concatenate(buffer) if buffer else np.zeros(0, dtype=np.float32)
        mono = downmix_to_mono(joined, channels)
        samples = resample_to_16k(mono, rate)
        duration = len(samples) / TARGET_RATE
        if duration <= 0:
            return

        self.captured_seconds += duration
        if is_silent(samples, self.config.silence_rms):
            # Never upload silence: it costs audio-seconds and returns nothing.
            self.skipped_silent += 1
            log.debug("%s chunk at %.0fs is silent; skipped", channel, start)
            return

        chunk = AudioChunk(channel=channel, start_s=start, duration_s=duration, samples=samples)
        if self.config.output_dir:
            chunk.path = write_wav(
                Path(self.config.output_dir) / f"{channel}_{int(start):06d}.wav", samples
            )
        self.chunks.put(chunk)

    def drain(self) -> list[AudioChunk]:
        """Everything recorded so far, in time order."""
        out: list[AudioChunk] = []
        while True:
            try:
                out.append(self.chunks.get_nowait())
            except queue.Empty:
                break
        out.sort(key=lambda c: c.start_s)
        return out
