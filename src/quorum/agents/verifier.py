"""Grounding verifier - the gate that makes "no hallucinated commitments" real.

Every extracted item claims a verbatim quote from a specific utterance. This
stage checks that claim mechanically and deletes whatever fails. Nothing here
asks a model whether it was telling the truth; the check is string matching
against the source transcript, so it cannot itself be talked out of its job.

Three outcomes per citation:

- **verified**   the quote matches the cited utterance
- **repaired**   the quote is genuinely in the transcript but the model pointed
                 at the wrong utterance. Models get the text right and the index
                 wrong often enough that discarding these would throw away real
                 findings, so the citation is re-pointed instead.
- **rejected**   the quote appears nowhere. The item is deleted.

The rejected count is the numerator of the hallucinated-commitment rate, which
is reported before and after this gate.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TypeVar

from rapidfuzz import fuzz

from quorum.models import Grounded, Transcript

log = logging.getLogger(__name__)

T = TypeVar("T", bound=Grounded)

_WS = re.compile(r"\s+")


def normalise(text: str) -> str:
    """Lowercase and collapse whitespace.

    Deliberately light. Stripping punctuation entirely would let a model invent
    clause boundaries that change meaning ("we will not ship" vs "we will,
    not ship") and still match.
    """
    return _WS.sub(" ", text.strip().lower())


@dataclass
class VerificationConfig:
    min_match_score: float = 85.0
    """Fuzzy threshold out of 100. Below ~80 paraphrases start passing, which
    defeats the point of demanding a verbatim quote."""

    min_quote_chars: int = 12
    """Quotes shorter than this are rejected outright. A two-word quote like
    "we will" partial-matches almost any transcript, so a short quote is not
    evidence - it is a loophole."""

    allow_repair: bool = True
    search_window: int = 0
    """0 searches the whole transcript for a misattributed quote. A positive
    value restricts repair to nearby utterances."""


@dataclass
class VerificationReport:
    proposed: int = 0
    accepted: int = 0
    rejected: int = 0
    repaired: int = 0
    rejections: list[tuple[str, str]] = field(default_factory=list)
    """(item description, reason) - kept for failure analysis, not just counting."""

    @property
    def hallucination_rate(self) -> float:
        """Share of proposed items that could not be grounded."""
        return self.rejected / self.proposed if self.proposed else 0.0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "proposed": self.proposed,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "repaired": self.repaired,
            "hallucination_rate": round(self.hallucination_rate, 4),
        }


class GroundingVerifier:
    def __init__(self, config: VerificationConfig | None = None) -> None:
        self.config = config or VerificationConfig()

    # -- public ------------------------------------------------------------

    def verify(self, items: list[T], transcript: Transcript) -> tuple[list[T], VerificationReport]:
        """Return only the items that survive grounding, plus a report."""
        report = VerificationReport(proposed=len(items))
        survivors: list[T] = []

        for item in items:
            kept, reason = self._verify_item(item, transcript, report)
            if kept:
                survivors.append(item)
                report.accepted += 1
            else:
                report.rejected += 1
                report.rejections.append((self._describe(item), reason))
                log.debug("Rejected %s: %s", self._describe(item)[:60], reason)

        return survivors, report

    # -- internals ---------------------------------------------------------

    def _verify_item(
        self, item: Grounded, transcript: Transcript, report: VerificationReport
    ) -> tuple[bool, str]:
        verified_any = False

        for evidence in item.evidence:
            quote = normalise(evidence.quote)
            if len(quote) < self.config.min_quote_chars:
                evidence.verified = False
                evidence.match_score = 0.0
                continue

            utterance = transcript.utterance(evidence.utterance_id)
            if utterance is not None:
                score = self._score(quote, utterance.text)
                if score >= self.config.min_match_score:
                    self._accept(evidence, utterance.speaker_id, utterance.timestamp, score)
                    verified_any = True
                    continue

            if self.config.allow_repair:
                match = self._find_quote(quote, transcript)
                if match is not None:
                    found, score = match
                    evidence.utterance_id = found.id
                    self._accept(evidence, found.speaker_id, found.timestamp, score)
                    report.repaired += 1
                    verified_any = True
                    continue

            evidence.verified = False
            evidence.match_score = 0.0

        if not verified_any:
            return False, self._reason(item)

        # Drop the individual citations that failed, keeping the ones that held.
        item.evidence = [e for e in item.evidence if e.verified]
        return True, ""

    @staticmethod
    def _accept(evidence, speaker_id: str, timestamp: str, score: float) -> None:
        evidence.verified = True
        evidence.match_score = round(score, 2)
        evidence.speaker_id = speaker_id
        evidence.timestamp = timestamp

    @staticmethod
    def _score(quote: str, source: str) -> float:
        """Partial ratio: the quote should appear *within* the utterance."""
        return float(fuzz.partial_ratio(quote, normalise(source)))

    def _find_quote(self, quote: str, transcript: Transcript):
        """Locate a quote anywhere in the transcript, for citation repair."""
        best, best_score = None, 0.0
        for utterance in transcript.utterances:
            score = self._score(quote, utterance.text)
            if score > best_score:
                best, best_score = utterance, score
        if best is not None and best_score >= self.config.min_match_score:
            return best, best_score
        return None

    def _reason(self, item: Grounded) -> str:
        shortest = min((len(normalise(e.quote)) for e in item.evidence), default=0)
        if shortest < self.config.min_quote_chars:
            return f"quote shorter than {self.config.min_quote_chars} chars"
        return "quote not found in transcript"

    @staticmethod
    def _describe(item: Grounded) -> str:
        for attr in ("description", "statement", "question"):
            value = getattr(item, attr, None)
            if value:
                return str(value)
        return item.id
