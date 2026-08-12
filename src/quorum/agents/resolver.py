"""Resolving who owns a commitment and when it is due.

This is the stage that decides whether the product is usable. An extractor that
finds "someone should send the spec" has done nothing actionable; the value is
in turning *"I'll get you that by Friday"* into
*(Yug Verma, yug@example.com, 2026-03-13)*.

Deterministic rules run first and settle the large majority of cases at zero
token cost:

- **"I", "me", "I'll"** - the owner is whoever spoke the cited line. This is the
  most common phrasing in real meetings and needs no model at all.
- **A name or nickname** - matched against the attendee roster.
- **"you"** - resolved from who was addressed in the line, or failing that, who
  answered next.
- **"we", "someone", "the team"** - deliberately left unresolved. A collective
  is not an owner, and inventing one produces a nag aimed at the wrong person.

Only a genuinely ambiguous mention reaches a model.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date

from pydantic import BaseModel, Field

from quorum.agents.dates import resolve_deadline
from quorum.llm.providers import ModelTier
from quorum.llm.router import Router, get_router
from quorum.models import Commitment, Deadline, Speaker, Transcript

log = logging.getLogger(__name__)

FIRST_PERSON = {"i", "i'll", "ill", "me", "my", "myself", "i will", "i can", "i've", "i am"}
SECOND_PERSON = {"you", "your", "you'll", "youll", "u"}
COLLECTIVE = {
    "we", "us", "our", "the team", "team", "everyone", "everybody", "all of us",
    "someone", "somebody", "anyone", "anybody", "no one", "whoever", "people",
    "the frontend team", "the backend team", "engineering", "ops", "they",
}

_WS = re.compile(r"\s+")


def _norm(text: str | None) -> str:
    return _WS.sub(" ", (text or "").strip().lower()).strip(" .,!?;:'\"")


class AssigneeGuess(BaseModel):
    """Model-side shape for the ambiguous-mention fallback."""

    speaker_name: str | None = Field(
        default=None, description="Exact display name from the roster, or null if unclear"
    )
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reasoning: str = ""


@dataclass
class ResolverConfig:
    use_llm_fallback: bool = True
    llm_tier: ModelTier = ModelTier.FAST
    min_llm_confidence: float = 0.6
    """Below this the model's guess is discarded. An unowned commitment is
    surfaced for a human; a wrongly-owned one silently nags an innocent
    colleague, which is much worse."""


@dataclass
class ResolutionStats:
    total: int = 0
    assignee_deterministic: int = 0
    assignee_llm: int = 0
    assignee_unresolved: int = 0
    deadline_deterministic: int = 0
    deadline_unresolved: int = 0
    llm_calls: int = 0
    unresolved_reasons: dict[str, int] = field(default_factory=dict)

    @property
    def deterministic_rate(self) -> float:
        """Share of resolved assignees that needed no model call at all."""
        resolved = self.assignee_deterministic + self.assignee_llm
        return self.assignee_deterministic / resolved if resolved else 0.0

    def as_dict(self) -> dict:
        return {
            "total": self.total,
            "assignee_deterministic": self.assignee_deterministic,
            "assignee_llm": self.assignee_llm,
            "assignee_unresolved": self.assignee_unresolved,
            "deadline_deterministic": self.deadline_deterministic,
            "deadline_unresolved": self.deadline_unresolved,
            "llm_calls": self.llm_calls,
            "deterministic_rate": round(self.deterministic_rate, 4),
            "unresolved_reasons": self.unresolved_reasons,
        }


class Resolver:
    def __init__(self, router: Router | None = None, config: ResolverConfig | None = None) -> None:
        self.config = config or ResolverConfig()
        self._router = router

    @property
    def router(self) -> Router:
        if self._router is None:
            self._router = get_router()
        return self._router

    # -- public ------------------------------------------------------------

    def resolve(
        self,
        commitments: list[Commitment],
        transcript: Transcript,
        known_events: dict[str, date] | None = None,
    ) -> ResolutionStats:
        """Resolve in place. Returns statistics for the metrics report."""
        stats = ResolutionStats(total=len(commitments))
        for commitment in commitments:
            self._resolve_assignee(commitment, transcript, stats)
            self._resolve_deadline(commitment, transcript, known_events, stats)
        return stats

    # -- assignee ----------------------------------------------------------

    def _resolve_assignee(
        self, commitment: Commitment, transcript: Transcript, stats: ResolutionStats
    ) -> None:
        mention = _norm(commitment.assignee.raw_mention)
        anchor = self._anchor_utterance(commitment, transcript)

        # 1. First person: the speaker of the cited line owns it.
        if mention in FIRST_PERSON or (not mention and anchor is not None):
            if anchor is not None:
                speaker = transcript.speaker(anchor.speaker_id)
                if speaker:
                    self._assign(commitment, speaker, 0.95, stats, deterministic=True)
                    return

        # 2. A collective is not an owner.
        if mention in COLLECTIVE:
            self._unresolved(commitment, stats, "collective or unowned mention")
            return

        # 3. Named person in the roster.
        if mention:
            speaker = transcript.resolve_mention(mention)
            if speaker:
                self._assign(commitment, speaker, 0.9, stats, deterministic=True)
                return

        # 4. Second person: who was being addressed?
        if mention in SECOND_PERSON and anchor is not None:
            addressed = self._addressee(anchor, transcript)
            if addressed:
                self._assign(commitment, addressed, 0.75, stats, deterministic=True)
                return

        # 5. Ambiguous - ask a model, but only if there is something to ask about.
        if self.config.use_llm_fallback and mention:
            speaker = self._ask_model(commitment, transcript, stats)
            if speaker:
                self._assign(commitment, speaker, 0.65, stats, deterministic=False)
                return

        self._unresolved(commitment, stats, f"unrecognised mention {mention!r}" if mention
                         else "no owner mentioned")

    def _anchor_utterance(self, commitment: Commitment, transcript: Transcript):
        """The utterance the commitment was spoken in."""
        for evidence in commitment.evidence:
            utterance = transcript.utterance(evidence.utterance_id)
            if utterance is not None:
                return utterance
        return None

    def _addressee(self, anchor, transcript: Transcript) -> Speaker | None:
        """Who "you" refers to in a line like "Sam, can you review it?".

        First look for a name inside the line itself. Failing that, take the
        next person to speak - in practice the question gets answered by the
        person it was aimed at.
        """
        text = _norm(anchor.text)
        for speaker in transcript.speakers:
            if speaker.id == anchor.speaker_id:
                continue
            for label in [speaker.display_name, *speaker.aliases]:
                if label and re.search(rf"\b{re.escape(label.lower())}\b", text):
                    return speaker

        for utterance in transcript.utterances[anchor.index + 1 :]:
            if utterance.speaker_id != anchor.speaker_id:
                return transcript.speaker(utterance.speaker_id)
        return None

    def _ask_model(
        self, commitment: Commitment, transcript: Transcript, stats: ResolutionStats
    ) -> Speaker | None:
        anchor = self._anchor_utterance(commitment, transcript)
        if anchor is None:
            return None

        lo = max(0, anchor.index - 3)
        hi = min(len(transcript.utterances), anchor.index + 3)
        roster = "\n".join(
            f"- {s.display_name}" + (f" (also called: {', '.join(s.aliases)})" if s.aliases else "")
            for s in transcript.speakers
        )
        prompt = (
            f"Meeting participants:\n{roster}\n\n"
            f"Conversation around the commitment:\n{transcript.as_dialogue(lo, hi)}\n\n"
            f'Commitment: "{commitment.description}"\n'
            f'Referred to the owner as: "{commitment.assignee.raw_mention}"\n\n'
            "Which participant owns this commitment? Use their exact display name. "
            "If the text does not make the owner clear, return null - guessing wrongly "
            "is worse than admitting it is unclear."
        )

        try:
            guess, _ = self.router.structured(
                prompt, AssigneeGuess, tier=self.config.llm_tier,
                max_tokens=256, purpose="resolve_assignee",
            )
        except Exception as exc:  # noqa: BLE001 - resolution failure must not sink the run
            log.warning("Assignee fallback failed: %s", exc)
            return None

        stats.llm_calls += 1
        if not guess.speaker_name or guess.confidence < self.config.min_llm_confidence:
            return None
        return transcript.resolve_mention(guess.speaker_name) or next(
            (s for s in transcript.speakers if _norm(s.display_name) == _norm(guess.speaker_name)),
            None,
        )

    @staticmethod
    def _assign(
        commitment: Commitment,
        speaker: Speaker,
        confidence: float,
        stats: ResolutionStats,
        *,
        deterministic: bool,
    ) -> None:
        commitment.assignee.speaker_id = speaker.id
        commitment.assignee.display_name = speaker.display_name
        commitment.assignee.email = speaker.email
        commitment.assignee.github_login = speaker.github_login
        commitment.assignee.confidence = confidence
        commitment.assignee.unresolved_reason = None
        if deterministic:
            stats.assignee_deterministic += 1
        else:
            stats.assignee_llm += 1

    @staticmethod
    def _unresolved(commitment: Commitment, stats: ResolutionStats, reason: str) -> None:
        commitment.assignee.unresolved_reason = reason
        commitment.assignee.confidence = 0.0
        stats.assignee_unresolved += 1
        stats.unresolved_reasons[reason] = stats.unresolved_reasons.get(reason, 0) + 1

    # -- deadline ----------------------------------------------------------

    def _resolve_deadline(
        self,
        commitment: Commitment,
        transcript: Transcript,
        known_events: dict[str, date] | None,
        stats: ResolutionStats,
    ) -> None:
        resolved = resolve_deadline(
            commitment.deadline.raw_text, transcript.meeting_date, known_events
        )
        commitment.deadline = Deadline(
            raw_text=commitment.deadline.raw_text,
            resolved=resolved.value,
            method=resolved.method,
            confidence=resolved.confidence,
        )
        if resolved.value is not None:
            stats.deadline_deterministic += 1
        else:
            stats.deadline_unresolved += 1
