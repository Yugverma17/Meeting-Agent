"""Scoring extraction against the AMI corpus.

Different in kind from the synthetic benchmark, and the difference matters.

The synthetic manifest stores the *exact sentence* a commitment was made in, so
alignment is essentially exact. AMI's ACTIONS are **abstractive** - an annotator
wrote "The project manager will send the specification" after the fact, and that
sentence appears nowhere in the transcript. Matching therefore has to be fuzzy
and semantic-ish, and the threshold is a judgement call rather than a fact.

Two consequences to state plainly whenever these numbers are quoted:

1. **AMI scores are not comparable to synthetic scores.** Different matching
   procedure, different notion of a hit. Report them side by side, never pooled.
2. **Human agreement on this task is around kappa 0.36.** Annotators barely agree
   with each other on what counts as an action item, so a system "underperforming"
   against one annotator may be disagreeing no more than a second annotator would.
   Perfect agreement is not the target and would be evidence of overfitting to one
   person's habits.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from rapidfuzz import fuzz

from quorum.data.ami import AmiMeeting
from quorum.eval.metrics import PrecisionRecall
from quorum.models import Commitment, CommitmentStrength

log = logging.getLogger(__name__)

MATCH_THRESHOLD = 55.0
"""Deliberately lower than the synthetic benchmark's 70. An abstractive summary
sentence and an extracted commitment share meaning, rarely wording."""

INTER_ANNOTATOR_KAPPA = 0.36
"""Published agreement between human annotators on action-item labelling in this
family of corpora. The honest ceiling for any score reported here."""


@dataclass
class AmiMeetingResult:
    meeting_id: str
    scores: PrecisionRecall = field(default_factory=PrecisionRecall)
    matched: list[tuple[str, str, float]] = field(default_factory=list)
    """(predicted description, annotator sentence, score) - for failure analysis."""

    missed: list[str] = field(default_factory=list)
    spurious: list[str] = field(default_factory=list)


def score_meeting(
    meeting: AmiMeeting,
    predicted: list[Commitment],
    threshold: float = MATCH_THRESHOLD,
) -> AmiMeetingResult:
    """Align extracted commitments with the annotator's ACTIONS, one-to-one."""
    result = AmiMeetingResult(meeting_id=meeting.meeting_id)
    firm = [c for c in predicted if c.strength is CommitmentStrength.FIRM]

    candidates = []
    for i, commitment in enumerate(firm):
        # Include the resolved owner: annotators name a role ("the project
        # manager"), and the description alone often omits who is acting.
        surface = commitment.description
        if commitment.assignee.display_name:
            surface = f"{commitment.assignee.display_name} {surface}"
        for j, action in enumerate(meeting.actions):
            candidates.append((fuzz.token_set_ratio(surface.lower(), action.lower()), i, j))

    candidates.sort(key=lambda row: -row[0])
    used_predictions: set[int] = set()
    used_actions: set[int] = set()

    for score, i, j in candidates:
        if score < threshold:
            break
        if i in used_predictions or j in used_actions:
            continue
        used_predictions.add(i)
        used_actions.add(j)
        result.matched.append((firm[i].description, meeting.actions[j], score))

    result.missed = [a for j, a in enumerate(meeting.actions) if j not in used_actions]
    result.spurious = [
        c.description for i, c in enumerate(firm) if i not in used_predictions
    ]
    result.scores = PrecisionRecall(
        true_positives=len(result.matched),
        false_positives=len(result.spurious),
        false_negatives=len(result.missed),
    )
    return result


def summarise(results: list[AmiMeetingResult]) -> dict:
    pooled = PrecisionRecall()
    for result in results:
        pooled = pooled + result.scores

    return {
        "corpus": "ami",
        "meetings": len(results),
        "commitments": pooled.as_dict(),
        "match_threshold": MATCH_THRESHOLD,
        "inter_annotator_kappa": INTER_ANNOTATOR_KAPPA,
        "note": (
            "Abstractive ground truth, fuzzy alignment - not comparable to the "
            "synthetic benchmark's exact-span scores. Human annotators agree with "
            f"each other at kappa~{INTER_ANNOTATOR_KAPPA} on this task, so perfect "
            "agreement is not the target."
        ),
    }
