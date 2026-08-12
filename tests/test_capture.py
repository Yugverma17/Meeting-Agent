"""Live-capture tests.

The signal processing and transcript assembly are pure functions and tested
directly. Actual device recording is not testable in CI - it needs a sound card
and a meeting - so `DualRecorder.start` is exercised only for its refusal paths.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from quorum.capture.audio import (
    MIC,
    SYSTEM,
    TARGET_RATE,
    AudioChunk,
    DualRecorder,
    RecorderConfig,
    downmix_to_mono,
    is_silent,
    resample_to_16k,
    rms,
    write_wav,
)
from quorum.capture.speakers import (
    REMOTE_SPEAKER_ID,
    RemoteSpeakerAttributor,
    SpeakerRoster,
    build_transcript,
    merge_segments,
)
from quorum.capture.transcribe import TranscriptSegment, WhisperTranscriber, to_wav_bytes
from quorum.models import Speaker


def tone(seconds: float, rate: int = 48_000, freq: float = 220.0, amp: float = 0.5):
    t = np.linspace(0, seconds, int(rate * seconds), endpoint=False)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# --- signal handling --------------------------------------------------------


def test_downmix_averages_interleaved_channels():
    interleaved = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)  # L,R,L,R
    assert downmix_to_mono(interleaved, 2).tolist() == [0.5, 0.5]


def test_downmix_handles_a_truncated_final_frame():
    """Audio buffers do not always end on a frame boundary."""
    assert len(downmix_to_mono(np.zeros(5, dtype=np.float32), 2)) == 2


def test_downmix_passes_mono_through():
    mono = np.array([0.1, 0.2], dtype=np.float32)
    assert downmix_to_mono(mono, 1).tolist() == pytest.approx([0.1, 0.2])


def test_resample_48k_to_16k_gives_a_third_of_the_samples():
    resampled = resample_to_16k(tone(1.0, 48_000), 48_000)
    assert len(resampled) == pytest.approx(TARGET_RATE, rel=0.01)


def test_resample_averages_rather_than_decimating():
    """Naive decimation aliases high frequencies into the speech band. It sounds
    fine to a human and measurably degrades transcription."""
    samples = np.array([0.0, 3.0, 0.0, 0.0, 3.0, 0.0], dtype=np.float32)
    assert resample_to_16k(samples, 48_000).tolist() == pytest.approx([1.0, 1.0])


def test_resample_handles_a_non_integer_ratio():
    resampled = resample_to_16k(tone(1.0, 44_100), 44_100)
    assert len(resampled) == pytest.approx(TARGET_RATE, rel=0.02)


def test_resample_is_a_noop_at_target_rate():
    samples = tone(0.1, TARGET_RATE)
    assert np.array_equal(resample_to_16k(samples, TARGET_RATE), samples)


def test_resample_of_empty_input_is_empty():
    assert len(resample_to_16k(np.zeros(0, dtype=np.float32), 48_000)) == 0


def test_silence_detection():
    """Most of any meeting is one side not talking. Skipping those chunks is the
    biggest saving against the 28,800 audio-seconds/day budget."""
    assert is_silent(np.zeros(1000, dtype=np.float32), 0.004)
    assert is_silent(np.full(1000, 0.0001, dtype=np.float32), 0.004)
    assert not is_silent(tone(0.5, TARGET_RATE), 0.004)


def test_rms_of_empty_is_zero():
    assert rms(np.zeros(0, dtype=np.float32)) == 0.0


# --- wav output -------------------------------------------------------------


def test_write_wav_is_mono_16bit_16k(tmp_path):
    path = write_wav(tmp_path / "a.wav", tone(0.25, TARGET_RATE))
    import wave

    with wave.open(str(path), "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        assert handle.getframerate() == TARGET_RATE
        assert handle.getnframes() == pytest.approx(TARGET_RATE * 0.25, rel=0.01)


def test_wav_bytes_have_a_riff_header():
    payload = to_wav_bytes(tone(0.1, TARGET_RATE))
    assert payload[:4] == b"RIFF" and payload[8:12] == b"WAVE"


def test_wav_clips_rather_than_wrapping():
    """Without clipping, an over-driven sample wraps to full-scale negative -
    an audible click that Whisper hears as a consonant."""
    loud = np.array([5.0, -5.0], dtype=np.float32)
    payload = to_wav_bytes(loud)
    pcm = np.frombuffer(payload[44:], dtype=np.int16)
    assert pcm.max() <= 32767 and pcm.min() >= -32767
    assert pcm[0] > 0 and pcm[1] < 0, "sign must be preserved"


# --- consent ----------------------------------------------------------------


def test_recording_without_announcement_is_refused():
    """Recording people without their knowledge is a legal problem in many
    jurisdictions, not merely an impolite one."""
    with pytest.raises(PermissionError, match="announcement"):
        DualRecorder().start(announced=False)


# --- segment merging --------------------------------------------------------


def seg(channel, start, end, text):
    return TranscriptSegment(channel=channel, start_s=start, end_s=end, text=text)


def test_breath_pauses_are_merged_into_one_utterance():
    """Whisper splits on breaths, which are not turn boundaries."""
    merged = merge_segments([
        seg(MIC, 0.0, 2.0, "I'll have the spec"),
        seg(MIC, 2.5, 4.0, "to you by Friday"),
    ])
    assert len(merged) == 1
    assert merged[0].text == "I'll have the spec to you by Friday"
    assert merged[0].end_s == 4.0


def test_a_real_pause_is_kept_separate():
    merged = merge_segments([
        seg(MIC, 0.0, 2.0, "First point"),
        seg(MIC, 10.0, 11.0, "Second point"),
    ])
    assert len(merged) == 2


def test_channel_change_always_splits():
    """A different channel is a different speaker, however tight the timing."""
    merged = merge_segments([
        seg(MIC, 0.0, 2.0, "Can you take that?"),
        seg(SYSTEM, 2.1, 3.0, "Sure, I'll do it."),
    ])
    assert len(merged) == 2


def test_merge_of_nothing_is_nothing():
    assert merge_segments([]) == []


# --- transcript assembly ----------------------------------------------------


def test_mic_becomes_you_and_system_becomes_remote():
    """The channel split gives speaker separation with no diarisation model."""
    roster = SpeakerRoster.solo("Yug Verma", "yug@example.com")
    transcript = build_transcript(
        [seg(MIC, 0.0, 2.0, "Where are we?"), seg(SYSTEM, 3.0, 5.0, "Nearly done.")],
        roster,
    )
    assert transcript.utterances[0].speaker_id == roster.you.id
    assert transcript.utterances[1].speaker_id == REMOTE_SPEAKER_ID


def test_transcript_is_ordered_and_indexed():
    roster = SpeakerRoster.solo()
    transcript = build_transcript(
        [seg(SYSTEM, 9.0, 10.0, "Third"), seg(MIC, 0.0, 1.0, "First"),
         seg(SYSTEM, 4.0, 5.0, "Second")],
        roster,
    )
    assert [u.text for u in transcript.utterances] == ["First", "Second", "Third"]
    assert [u.index for u in transcript.utterances] == [0, 1, 2]


def test_transcript_is_marked_live_sourced():
    """Metrics must never pool live, synthetic and AMI results."""
    transcript = build_transcript([seg(MIC, 0.0, 1.0, "Hello")], SpeakerRoster.solo())
    assert transcript.source == "live"


def test_transcript_carries_the_meeting_date():
    transcript = build_transcript(
        [seg(MIC, 0.0, 1.0, "Hello")], SpeakerRoster.solo(), meeting_date=date(2026, 3, 9)
    )
    assert transcript.meeting_date == date(2026, 3, 9)


def test_single_remote_participant_needs_no_model():
    """One remote person makes the shared channel unambiguous."""
    roster = SpeakerRoster(
        you=Speaker(id="spk_you", display_name="Yug"),
        others=[Speaker(id="spk_p", display_name="Priya", email="priya@x.com")],
    )
    transcript = build_transcript([seg(SYSTEM, 0.0, 2.0, "I'll review it")], roster)

    attributor = RemoteSpeakerAttributor(router=None)
    attributor.attribute(transcript, roster)

    assert transcript.utterances[0].speaker_id == "spk_p"
    assert attributor.calls == 0, "no tokens should be spent on an unambiguous case"


def test_attribution_abstains_rather_than_guessing():
    """A wrongly attributed commitment silently nags the wrong colleague. An
    unattributed one gets surfaced to a human, which is the safer failure."""
    from quorum.capture.speakers import Attribution, AttributionBatch
    from quorum.llm.router import LLMResponse

    class LowConfidence:
        def structured(self, *args, **kwargs):
            return (
                AttributionBatch(
                    attributions=[Attribution(utterance_index=0, speaker_name="Priya",
                                              confidence=0.2)]
                ),
                LLMResponse(text="{}", model="fake", provider="fake"),
            )

    roster = SpeakerRoster(
        you=Speaker(id="spk_you", display_name="Yug"),
        others=[
            Speaker(id="spk_p", display_name="Priya"),
            Speaker(id="spk_s", display_name="Sam"),
        ],
    )
    transcript = build_transcript([seg(SYSTEM, 0.0, 2.0, "I'll review it")], roster)
    RemoteSpeakerAttributor(router=LowConfidence()).attribute(transcript, roster)

    assert transcript.utterances[0].speaker_id == REMOTE_SPEAKER_ID


def test_attribution_failure_leaves_the_transcript_usable():
    class Broken:
        def structured(self, *args, **kwargs):
            raise RuntimeError("provider down")

    roster = SpeakerRoster(
        you=Speaker(id="spk_you", display_name="Yug"),
        others=[Speaker(id="spk_p", display_name="Priya"),
                Speaker(id="spk_s", display_name="Sam")],
    )
    transcript = build_transcript([seg(SYSTEM, 0.0, 2.0, "I'll review it")], roster)
    RemoteSpeakerAttributor(router=Broken()).attribute(transcript, roster)

    assert transcript.utterances[0].text == "I'll review it"


# --- transcription ----------------------------------------------------------


class FakeGroq:
    def __init__(self, segments, fail_times: int = 0) -> None:
        self.segments = segments
        self.fail_times = fail_times
        self.calls = 0
        self.audio = self

    @property
    def transcriptions(self):
        return self

    def create(self, **kwargs):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("rate limited")
        return type("R", (), {"segments": self.segments, "text": ""})()


def chunk(channel=MIC, start=60.0, seconds=30.0):
    return AudioChunk(
        channel=channel, start_s=start, duration_s=seconds,
        samples=tone(0.1, TARGET_RATE),
    )


def test_timestamps_are_shifted_by_the_chunk_offset():
    """Whisper timestamps are relative to the chunk. Without shifting, every
    chunk claims to start at zero and the meeting collapses onto itself."""
    fake = FakeGroq([{"start": 2.0, "end": 5.0, "text": "I'll send it Friday"}])
    segments = WhisperTranscriber(client=fake).transcribe_chunk(chunk(start=60.0))

    assert len(segments) == 1
    assert segments[0].start_s == 62.0
    assert segments[0].end_s == 65.0


def test_blank_segments_are_dropped():
    fake = FakeGroq([{"start": 0.0, "end": 1.0, "text": "   "}])
    assert WhisperTranscriber(client=fake).transcribe_chunk(chunk()) == []


def test_falls_back_to_the_flat_text_field():
    """Some responses carry no segments, only `text`."""

    class NoSegments(FakeGroq):
        def create(self, **kwargs):
            self.calls += 1
            return type("R", (), {"segments": [], "text": "Hello everyone"})()

    segments = WhisperTranscriber(client=NoSegments([])).transcribe_chunk(chunk(start=30.0))
    assert len(segments) == 1 and segments[0].start_s == 30.0


def test_transient_failure_is_retried(monkeypatch):
    monkeypatch.setattr("quorum.capture.transcribe.time.sleep", lambda _: None)
    fake = FakeGroq([{"start": 0.0, "end": 1.0, "text": "ok"}], fail_times=1)
    transcriber = WhisperTranscriber(client=fake)

    assert len(transcriber.transcribe_chunk(chunk())) == 1
    assert fake.calls == 2


def test_permanent_failure_returns_nothing_rather_than_raising(monkeypatch):
    """One bad chunk must not lose the meeting."""
    monkeypatch.setattr("quorum.capture.transcribe.time.sleep", lambda _: None)
    transcriber = WhisperTranscriber(client=FakeGroq([], fail_times=99), max_retries=1)

    assert transcriber.transcribe_chunk(chunk()) == []
    assert transcriber.stats.failures == 1


def test_stats_track_the_daily_audio_budget():
    fake = FakeGroq([{"start": 0.0, "end": 1.0, "text": "hi"}])
    transcriber = WhisperTranscriber(client=fake)
    transcriber.transcribe_all([chunk(seconds=1440.0), chunk(seconds=1440.0)])

    assert transcriber.stats.audio_seconds == 2880.0
    assert transcriber.stats.daily_budget_used == pytest.approx(0.1)


def test_transcribe_all_returns_time_ordered_segments():
    fake = FakeGroq([{"start": 0.0, "end": 1.0, "text": "x"}])
    segments = WhisperTranscriber(client=fake).transcribe_all(
        [chunk(start=90.0), chunk(start=0.0), chunk(start=30.0)]
    )
    assert [s.start_s for s in segments] == [0.0, 30.0, 90.0]
