"""Metric definitions.

Two families, and the second is the point of the project:

**Extraction** - precision/recall on finding commitments in a single meeting.
Comparable in spirit to what AMI or QMSum can score.

**Tracking** - dropped-commitment recall, false-nag rate, contradiction
detection. None of these are scoreable on any public corpus, because no public
corpus follows a commitment across weeks. They exist here only because the
synthetic benchmark knows the outcome of every commitment by construction.

On reporting: `false_nag_rate` deserves more weight than it looks. Missing a
dropped commitment is a silent failure; chasing someone who already delivered is
a visible, irritating one, and it is what gets a tool switched off.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PrecisionRecall:
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    @property
    def precision(self) -> float:
        denominator = self.true_positives + self.false_positives
        return self.true_positives / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        denominator = self.true_positives + self.false_negatives
        return self.true_positives / denominator if denominator else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def __add__(self, other: PrecisionRecall) -> PrecisionRecall:
        return PrecisionRecall(
            self.true_positives + other.true_positives,
            self.false_positives + other.false_positives,
            self.false_negatives + other.false_negatives,
        )

    def as_dict(self) -> dict[str, float | int]:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "tp": self.true_positives,
            "fp": self.false_positives,
            "fn": self.false_negatives,
        }


@dataclass
class ExtractionScores:
    commitments: PrecisionRecall = field(default_factory=PrecisionRecall)

    assignee_correct: int = 0
    assignee_total: int = 0
    deadline_correct: int = 0
    deadline_total: int = 0
    strength_correct: int = 0
    strength_total: int = 0

    musings_promoted: int = 0
    musings_total: int = 0
    """Idle talk wrongly turned into a commitment. The precision failure that
    makes these tools unusable in practice."""

    hallucinated: int = 0
    proposed: int = 0

    def _ratio(self, correct: int, total: int) -> float:
        return correct / total if total else 0.0

    @property
    def assignee_accuracy(self) -> float:
        return self._ratio(self.assignee_correct, self.assignee_total)

    @property
    def deadline_accuracy(self) -> float:
        return self._ratio(self.deadline_correct, self.deadline_total)

    @property
    def strength_accuracy(self) -> float:
        return self._ratio(self.strength_correct, self.strength_total)

    @property
    def musing_promotion_rate(self) -> float:
        return self._ratio(self.musings_promoted, self.musings_total)

    @property
    def hallucination_rate(self) -> float:
        return self._ratio(self.hallucinated, self.proposed)

    def as_dict(self) -> dict:
        return {
            "commitments": self.commitments.as_dict(),
            "assignee_accuracy": round(self.assignee_accuracy, 4),
            "deadline_accuracy": round(self.deadline_accuracy, 4),
            "strength_accuracy": round(self.strength_accuracy, 4),
            "musing_promotion_rate": round(self.musing_promotion_rate, 4),
            "hallucination_rate": round(self.hallucination_rate, 4),
            "counts": {
                "assignee_total": self.assignee_total,
                "deadline_total": self.deadline_total,
                "musings_total": self.musings_total,
                "proposed": self.proposed,
            },
        }


@dataclass
class TrackingScores:
    """Longitudinal metrics. No public benchmark can produce these."""

    dropped_caught: int = 0
    dropped_total: int = 0

    false_nags: int = 0
    nag_targets_total: int = 0

    contradictions_caught: int = 0
    contradictions_total: int = 0
    contradiction_false_positives: int = 0
    """Reversals reported that were not real. Tracked separately because
    recall alone can be gamed by flagging every pair of decisions."""

    silent_deliveries_verified: int = 0
    silent_deliveries_total: int = 0

    blocked_propagated: int = 0
    blocked_total: int = 0

    def _ratio(self, correct: int, total: int) -> float:
        return correct / total if total else 0.0

    @property
    def dropped_recall(self) -> float:
        """Of the commitments everyone forgot, how many did the agent still chase?"""
        return self._ratio(self.dropped_caught, self.dropped_total)

    @property
    def false_nag_rate(self) -> float:
        """Of the commitments already delivered or cancelled, how many did it
        chase anyway? Lower is better; this is the usability metric."""
        return self._ratio(self.false_nags, self.nag_targets_total)

    @property
    def contradiction_recall(self) -> float:
        return self._ratio(self.contradictions_caught, self.contradictions_total)

    @property
    def contradiction_precision(self) -> float:
        detected = self.contradictions_caught + self.contradiction_false_positives
        return self._ratio(self.contradictions_caught, detected)

    @property
    def silent_delivery_recall(self) -> float:
        """Delivered but never discussed - provable only from external evidence."""
        return self._ratio(self.silent_deliveries_verified, self.silent_deliveries_total)

    @property
    def blocked_propagation_recall(self) -> float:
        return self._ratio(self.blocked_propagated, self.blocked_total)

    def as_dict(self) -> dict:
        return {
            "dropped_recall": round(self.dropped_recall, 4),
            "false_nag_rate": round(self.false_nag_rate, 4),
            "contradiction_recall": round(self.contradiction_recall, 4),
            "contradiction_precision": round(self.contradiction_precision, 4),
            "silent_delivery_recall": round(self.silent_delivery_recall, 4),
            "blocked_propagation_recall": round(self.blocked_propagation_recall, 4),
            "counts": {
                "dropped_total": self.dropped_total,
                "nag_targets_total": self.nag_targets_total,
                "contradictions_total": self.contradictions_total,
                "silent_deliveries_total": self.silent_deliveries_total,
                "blocked_total": self.blocked_total,
            },
        }
