"""Aligning predicted commitments to ground truth.

Every downstream number depends on this step, and it is easy to get quietly
wrong. Two rules keep it honest:

**Match on the quote, not the description.** A model writes "send the ingestion
spec" where the manifest says "the ingestion API spec". Comparing those
paraphrases measures the model's phrasing, not whether it found the right
commitment. The cited transcript span is the same string in both, so alignment
uses that.

**One-to-one, best-first.** Greedy nearest-match lets a single prediction
satisfy two ground-truth items, which silently inflates recall. Pairs are
scored, sorted, and consumed exclusively.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rapidfuzz import fuzz

from quorum.models import Commitment
from quorum.synth.project import GroundTruthCommitment

MATCH_THRESHOLD = 70.0
"""Below this two items are different commitments. Deliberately lenient on
wording and strict on identity - a prediction only has to point at the same
utterance content, not phrase it the same way."""


@dataclass
class MatchedPair:
    predicted: Commitment
    truth: GroundTruthCommitment
    score: float

    @property
    def assignee_correct(self) -> bool:
        return (
            self.predicted.assignee.display_name is not None
            and self.predicted.assignee.display_name == self.truth.owner_name
        )

    @property
    def deadline_correct(self) -> bool:
        return self.predicted.deadline.resolved == self.truth.deadline_date

    @property
    def strength_correct(self) -> bool:
        return self.predicted.strength.value == self.truth.strength


@dataclass
class Alignment:
    matched: list[MatchedPair] = field(default_factory=list)
    missed: list[GroundTruthCommitment] = field(default_factory=list)
    """Real commitments the system never found - false negatives."""

    spurious: list[Commitment] = field(default_factory=list)
    """Predictions with no ground-truth counterpart - false positives."""

    @property
    def true_positives(self) -> int:
        return len(self.matched)


def _best_quote_score(predicted: Commitment, truth: GroundTruthCommitment) -> float:
    """How well any of the prediction's citations matches the true spoken line."""
    target = truth.spoken_quote.lower()
    best = 0.0
    for evidence in predicted.evidence:
        best = max(best, float(fuzz.partial_ratio(evidence.quote.lower(), target)))
    # Fall back to the description when a prediction cites loosely; weighted
    # down so a description-only match cannot outrank a real quote match.
    description = float(fuzz.token_set_ratio(predicted.description.lower(), target)) * 0.75
    return max(best, description)


def align_commitments(
    predicted: list[Commitment],
    truth: list[GroundTruthCommitment],
    threshold: float = MATCH_THRESHOLD,
) -> Alignment:
    """Pair predictions with ground truth, best matches first, one-to-one."""
    candidates = [
        (_best_quote_score(p, t), i, j)
        for i, p in enumerate(predicted)
        for j, t in enumerate(truth)
    ]
    candidates.sort(key=lambda c: -c[0])

    used_predictions: set[int] = set()
    used_truths: set[int] = set()
    alignment = Alignment()

    for score, i, j in candidates:
        if score < threshold:
            break
        if i in used_predictions or j in used_truths:
            continue
        used_predictions.add(i)
        used_truths.add(j)
        alignment.matched.append(MatchedPair(predicted[i], truth[j], score))

    alignment.missed = [t for j, t in enumerate(truth) if j not in used_truths]
    alignment.spurious = [p for i, p in enumerate(predicted) if i not in used_predictions]
    return alignment
