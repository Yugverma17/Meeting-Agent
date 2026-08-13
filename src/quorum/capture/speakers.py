"""Turning two audio channels into a speaker-labelled transcript.

The microphone/loopback split gives **you vs everyone else** for free and exactly
right. What it cannot give is separation *among* the remote participants — they
all arrive mixed on one channel.

This is the weakest link in the live path, and worth being blunt about. Three
options were considered:

1. **Proper diarisation** (pyannote). Correct, and needs more RAM than this
   machine has spare while also running a browser and a meeting. Ruled out by
   hardware, not by preference.
2. **Leave everyone as "Remote".** Honest, and downstream the resolver simply
   cannot attribute a commitment to a person — so half the product stops working.
3. **Attribute from the invite roster using conversational cues.** Imperfect,
   cheap, and degrades safely.

Option 3 is implemented, with a deliberate bias: when attribution is unsure it
returns *unknown* rather than guessing. An unattributed commitment gets surfaced
for a human to assign; a wrongly attributed one silently nags the wrong
colleague, which is the worse failure and the harder one to notice.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

from pydantic import BaseModel, Field

from quorum.capture.audio import MIC, SYSTEM
from quorum.capture.transcribe import TranscriptSegment
from quorum.llm.providers import ModelTier
from quorum.llm.router import Router, get_router
from quorum.models import Speaker, Transcript, Utterance

log = logging.getLogger(__name__)

REMOTE_SPEAKER_ID = "spk_remote"
YOU_SPEAKER_ID = "spk_you"

MERGE_GAP_S = 1.2
"""Consecutive segments on one channel closer than this are one utterance.
Whisper splits on breath pauses, which are not turn boundaries."""

MAX_MERGED_S = 25.0
"""Ceiling on how long a merged utterance may grow.

Without it, merging runs away. Recording chunks are contiguous - chunk N ends at
30.0s and chunk N+1 begins at 30.0s, a gap of zero - so an uninterrupted speaker
merges across every chunk boundary in the recording. A 19-minute lecture
collapsed into a single utterance, which the segmenter then could not split
(nothing divides one utterance), so the entire talk went to the model in one
call and came back summarised rather than extracted: two key points for
nineteen minutes.

Nobody speaks in 19-minute turns. Capping the merge restores real utterance
boundaries and lets segmentation and extraction work as intended."""


class Attribution(BaseModel):
    utterance_index: int
    speaker_name: str | None = Field(
        default=None, description="Exact roster name, or null if genuinely unclear"
    )
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class AttributionBatch(BaseModel):
    attributions: list[Attribution] = Field(default_factory=list)


@dataclass
class SpeakerRoster:
    """Who is in the meeting. Normally lifted from the calendar invite."""

    you: Speaker
    others: list[Speaker]

    @property
    def all(self) -> list[Speaker]:
        return [self.you, *self.others]

    @classmethod
    def solo(cls, your_name: str = "You", your_email: str | None = None) -> SpeakerRoster:
        return cls(
            you=Speaker(id=YOU_SPEAKER_ID, display_name=your_name, email=your_email,
                        aliases=["I", "me"]),
            others=[],
        )


def merge_segments(
    segments: list[TranscriptSegment],
    gap: float = MERGE_GAP_S,
    max_merged_s: float = MAX_MERGED_S,
) -> list[TranscriptSegment]:
    """Join same-channel segments separated by only a breath, up to a ceiling."""
    if not segments:
        return []

    ordered = sorted(segments, key=lambda s: s.start_s)
    merged = [ordered[0]]
    for segment in ordered[1:]:
        previous = merged[-1]
        close_enough = segment.start_s - previous.end_s <= gap
        would_stay_short = (segment.end_s - previous.start_s) <= max_merged_s

        if segment.channel == previous.channel and close_enough and would_stay_short:
            merged[-1] = TranscriptSegment(
                channel=previous.channel, start_s=previous.start_s, end_s=segment.end_s,
                text=f"{previous.text} {segment.text}".strip(),
            )
        else:
            merged.append(segment)
    return merged


def build_transcript(
    segments: list[TranscriptSegment],
    roster: SpeakerRoster,
    meeting_date: date | None = None,
    title: str = "Live meeting",
    project_id: str | None = None,
    meeting_id: str | None = None,
) -> Transcript:
    """Assemble a Transcript the rest of the pipeline can consume unchanged.

    That is the point of this module: everything downstream — segmenter,
    extractor, verifier, resolver, ledger — was built against `Transcript` and
    needs no knowledge that the source was live audio rather than a file.
    """
    merged = merge_segments(segments)
    remote = Speaker(
        id=REMOTE_SPEAKER_ID, display_name="Remote participant", aliases=["they", "them"]
    )
    speakers = [*roster.all]
    if not any(s.id == REMOTE_SPEAKER_ID for s in speakers):
        speakers.append(remote)

    utterances = [
        Utterance(
            id=f"live_u{index}", index=index,
            speaker_id=roster.you.id if segment.channel == MIC else REMOTE_SPEAKER_ID,
            text=segment.text, start_s=segment.start_s, end_s=segment.end_s,
        )
        for index, segment in enumerate(merged)
    ]

    return Transcript(
        meeting_id=meeting_id or f"live_{int(merged[0].start_s) if merged else 0}",
        title=title,
        meeting_date=meeting_date or date.today(),
        speakers=speakers,
        utterances=utterances,
        source="live",
        project_id=project_id,
    )


class RemoteSpeakerAttributor:
    """Best-effort naming of the remote participants sharing one channel."""

    SYSTEM_PROMPT = """\
You assign speaker names to lines of a meeting transcript.

All the lines below came from remote participants sharing one audio channel, so
the speaker of each is unknown. Use conversational evidence to attribute them:
people addressing each other by name, self-introduction, replies to a direct
question, or a consistent topic one person clearly owns.

Return null for any line you are not confident about. An unattributed line is
handled by a human; a wrongly attributed one silently assigns work to the wrong
person and nobody notices. Guessing is worse than abstaining."""

    def __init__(
        self,
        router: Router | None = None,
        tier: ModelTier = ModelTier.BALANCED,
        min_confidence: float = 0.6,
        batch_size: int = 25,
    ) -> None:
        self._router = router
        self.tier = tier
        self.min_confidence = min_confidence
        self.batch_size = batch_size
        self.calls = 0
        self.attributed = 0
        self.abstained = 0

    @property
    def router(self) -> Router:
        if self._router is None:
            self._router = get_router()
        return self._router

    def attribute(self, transcript: Transcript, roster: SpeakerRoster) -> Transcript:
        """Reassign remote utterances to named people where confident."""
        if len(roster.others) < 2:
            # One remote person means the channel is unambiguous; no model needed.
            if len(roster.others) == 1:
                for utterance in transcript.utterances:
                    if utterance.speaker_id == REMOTE_SPEAKER_ID:
                        utterance.speaker_id = roster.others[0].id
                        self.attributed += 1
            return transcript

        targets = [u for u in transcript.utterances if u.speaker_id == REMOTE_SPEAKER_ID]
        by_name = {s.display_name.lower(): s for s in roster.others}

        for start in range(0, len(targets), self.batch_size):
            batch = targets[start : start + self.batch_size]
            for attribution in self._ask(batch, transcript, roster):
                if attribution.speaker_name is None:
                    continue
                if attribution.confidence < self.min_confidence:
                    self.abstained += 1
                    continue
                speaker = by_name.get(attribution.speaker_name.strip().lower())
                match = next(
                    (u for u in batch if u.index == attribution.utterance_index), None
                )
                if speaker and match:
                    match.speaker_id = speaker.id
                    self.attributed += 1

        self.abstained += sum(
            1 for u in transcript.utterances if u.speaker_id == REMOTE_SPEAKER_ID
        )
        return transcript

    def _ask(
        self, batch: list[Utterance], transcript: Transcript, roster: SpeakerRoster
    ) -> list[Attribution]:
        names = "\n".join(f"- {s.display_name}" for s in roster.others)
        lines = "\n".join(f"[{u.index}] {u.text}" for u in batch)
        prompt = (
            f"Remote participants:\n{names}\n\n"
            f"You are {roster.you.display_name}; your own lines are not shown.\n\n"
            f"Lines to attribute:\n{lines}\n\n"
            "Give a speaker for each line index, or null where unclear."
        )
        try:
            result, _ = self.router.structured(
                prompt, AttributionBatch, system=self.SYSTEM_PROMPT,
                tier=self.tier, max_tokens=1024, purpose="attribute_speakers",
            )
        except Exception as exc:  # noqa: BLE001 - never sink a recording over this
            log.warning("Speaker attribution failed: %s", exc)
            return []
        self.calls += 1
        return result.attributions

    def stats(self) -> dict:
        total = self.attributed + self.abstained
        return {
            "llm_calls": self.calls,
            "attributed": self.attributed,
            "abstained": self.abstained,
            "attribution_rate": round(self.attributed / total, 4) if total else 0.0,
        }
