from __future__ import annotations

import pytest

from quorum.eval.metrics import ExtractionScores, PrecisionRecall, TrackingScores


# --- the bug this file exists for -------------------------------------------


def test_no_rate_can_exceed_one():
    """Regression: contradiction recall once reported 1.167.

    The harness counted raw detections rather than *correct* detections, so
    false positives landed in the numerator. A rate above 1.0 is arithmetically
    impossible and was the visible symptom of a metric that could be gamed by
    flagging every pair of decisions.
    """
    scores = TrackingScores(
        dropped_caught=3, dropped_total=3,
        false_nags=2, nag_targets_total=4,
        contradictions_caught=6, contradictions_total=6,
        contradiction_false_positives=5,
        silent_deliveries_verified=1, silent_deliveries_total=2,
        blocked_propagated=1, blocked_total=1,
    )
    payload = scores.as_dict()
    for key, value in payload.items():
        if isinstance(value, float):
            assert 0.0 <= value <= 1.0, f"{key} = {value} is not a valid rate"


def test_false_positives_lower_precision_without_touching_recall():
    scores = TrackingScores(
        contradictions_caught=3, contradictions_total=6, contradiction_false_positives=3
    )
    assert scores.contradiction_recall == 0.5
    assert scores.contradiction_precision == 0.5


def test_perfect_detection_scores_one_on_both():
    scores = TrackingScores(
        contradictions_caught=6, contradictions_total=6, contradiction_false_positives=0
    )
    assert scores.contradiction_recall == 1.0
    assert scores.contradiction_precision == 1.0


def test_flagging_everything_is_punished_by_precision():
    """Recall alone rewards over-flagging; precision is what stops it."""
    scores = TrackingScores(
        contradictions_caught=6, contradictions_total=6, contradiction_false_positives=40
    )
    assert scores.contradiction_recall == 1.0
    assert scores.contradiction_precision < 0.2


# --- precision / recall basics ----------------------------------------------


def test_precision_recall_f1():
    pr = PrecisionRecall(true_positives=8, false_positives=2, false_negatives=2)
    assert pr.precision == 0.8
    assert pr.recall == 0.8
    assert pr.f1 == pytest.approx(0.8)


def test_empty_counts_do_not_divide_by_zero():
    pr = PrecisionRecall()
    assert (pr.precision, pr.recall, pr.f1) == (0.0, 0.0, 0.0)


def test_precision_recall_adds_for_pooling():
    """Aggregation pools counts rather than averaging rates, so a project with
    two commitments does not weigh the same as one with twelve."""
    total = PrecisionRecall(4, 1, 1) + PrecisionRecall(6, 1, 3)
    assert (total.true_positives, total.false_positives, total.false_negatives) == (10, 2, 4)


def test_extraction_rates_stay_in_range():
    scores = ExtractionScores(
        assignee_correct=9, assignee_total=10,
        deadline_correct=8, deadline_total=10,
        strength_correct=10, strength_total=10,
        musings_promoted=0, musings_total=15,
        hallucinated=1, proposed=48,
    )
    assert scores.assignee_accuracy == 0.9
    assert scores.musing_promotion_rate == 0.0
    assert 0.0 <= scores.hallucination_rate <= 1.0


def test_zero_musings_does_not_crash_the_rate():
    assert ExtractionScores(musings_total=0).musing_promotion_rate == 0.0
