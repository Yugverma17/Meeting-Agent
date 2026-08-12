"""End-to-end evaluation over synthetic projects.

Runs the real pipeline - segment, extract, verify, resolve, ledger, plan - across
a whole generated project, then scores the result against the manifest that
produced it.

The tracking metrics need one honest simplification: the reality-verification
layer talks to GitHub, which has nothing in it for a project that never existed.
`ManifestEvidenceProvider` stands in, answering from the manifest's known
outcomes through the exact same `EvidenceProvider` interface the real GitHub
client implements. It simulates the *data source*, never the decision - the
planner still has to ask, and still has to act correctly on the answer.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date, timedelta

from rapidfuzz import fuzz

from quorum.agents.contradictions import ContradictionDetector
from quorum.agents.embedding import Embedder
from quorum.agents.extractor import Extractor
from quorum.agents.resolver import Resolver
from quorum.agents.segmenter import Segmenter
from quorum.agents.verifier import GroundingVerifier
from quorum.eval.matching import align_commitments
from quorum.eval.metrics import ExtractionScores, PrecisionRecall, TrackingScores
from quorum.models import Commitment, CommitmentStrength, MeetingRecord, Transcript
from quorum.synth.project import CommitmentFate, ProjectGenerator, ProjectManifest
from quorum.tracking import ActionType, Ledger, Planner, PlannerConfig
from quorum.tracking.planner import DeliveryEvidence

log = logging.getLogger(__name__)


class ManifestEvidenceProvider:
    """Stands in for GitHub during evaluation, using known outcomes.

    Matches on description because that is all a real GitHub search has to go
    on either - a PR title, not a commitment id.
    """

    MATCH_THRESHOLD = 80.0

    def __init__(self, manifest: ProjectManifest, as_of: date) -> None:
        self.manifest = manifest
        self.as_of = as_of
        self.lookups = 0

    def find_evidence(self, commitment: Commitment) -> DeliveryEvidence | None:
        self.lookups += 1
        best, best_score = None, 0.0
        for truth in self.manifest.commitments:
            if not truth.github_evidence or not truth.delivered_on:
                continue
            if truth.delivered_on > self.as_of:
                continue
            score = fuzz.token_set_ratio(
                truth.description.lower(), commitment.description.lower()
            )
            if score > best_score:
                best, best_score = truth, score

        if best is None or best_score < self.MATCH_THRESHOLD:
            return None
        return DeliveryEvidence(
            source="github", reference=best.github_evidence,
            found_on=best.delivered_on, confidence=best_score / 100.0,
            detail=best.description,
        )


@dataclass
class ProjectResult:
    project_id: str
    extraction: ExtractionScores = field(default_factory=ExtractionScores)
    tracking: TrackingScores = field(default_factory=TrackingScores)
    llm_calls: int = 0
    total_tokens: int = 0
    wall_seconds: float = 0.0
    meetings: int = 0
    degraded_calls: int = 0
    failed_segments: int = 0

    def as_dict(self) -> dict:
        return {
            "project_id": self.project_id,
            "extraction": self.extraction.as_dict(),
            "tracking": self.tracking.as_dict(),
            "cost": {
                "llm_calls": self.llm_calls,
                "total_tokens": self.total_tokens,
                "wall_seconds": round(self.wall_seconds, 2),
                "tokens_per_meeting": (
                    round(self.total_tokens / self.meetings) if self.meetings else 0
                ),
                "degraded_calls": self.degraded_calls,
                "failed_segments": self.failed_segments,
            },
        }


class EvaluationHarness:
    def __init__(
        self,
        extractor: Extractor | None = None,
        resolver: Resolver | None = None,
        segmenter: Segmenter | None = None,
        verifier: GroundingVerifier | None = None,
        planner: Planner | None = None,
        embedder: Embedder | None = None,
        use_evidence: bool = True,
        detect_contradictions: bool = True,
    ) -> None:
        self.extractor = extractor or Extractor()
        self.resolver = resolver or Resolver()
        self.segmenter = segmenter or Segmenter(embedder=embedder)
        self.verifier = verifier or GroundingVerifier()
        self.planner = planner or Planner()
        self.use_evidence = use_evidence
        self.contradiction_detector = ContradictionDetector() if detect_contradictions else None

    # -- one project -------------------------------------------------------

    def run_project(
        self, transcripts: list[Transcript], manifest: ProjectManifest
    ) -> ProjectResult:
        started = time.time()
        result = ProjectResult(project_id=manifest.project_id, meetings=len(transcripts))
        ledger = Ledger(manifest.project_id)
        known_events = {
            name: date.fromisoformat(value) for name, value in manifest.known_events.items()
        }
        truth_to_prediction: dict[str, str] = {}

        for index, transcript in enumerate(transcripts):
            record = self._process_meeting(transcript, known_events, result)
            ledger.ingest(record)
            self._score_meeting(index, record, manifest, result, truth_to_prediction)

        # Contradiction detection runs once at the end, over the whole decision
        # history - a reversal can only be recognised against everything before
        # it, so doing this per meeting would waste calls re-checking old pairs.
        if self.contradiction_detector is not None and len(ledger.decisions) > 1:
            self.contradiction_detector.scan(ledger.decisions)
            result.llm_calls += self.contradiction_detector.calls

        self._score_tracking(ledger, manifest, transcripts, result, truth_to_prediction)
        result.wall_seconds = time.time() - started
        return result

    def _process_meeting(
        self, transcript: Transcript, known_events: dict[str, date], result: ProjectResult
    ) -> MeetingRecord:
        segments = self.segmenter.segment(transcript)
        extraction = self.extractor.extract(transcript, segments)

        commitments, report = self.verifier.verify(extraction.commitments, transcript)
        decisions, _ = self.verifier.verify(extraction.decisions, transcript)
        self.resolver.resolve(commitments, transcript, known_events)

        result.extraction.hallucinated += report.rejected
        result.extraction.proposed += report.proposed
        result.llm_calls += extraction.stats.llm_calls
        result.total_tokens += extraction.stats.total_tokens
        result.degraded_calls += extraction.stats.degraded_calls
        result.failed_segments += extraction.stats.failed_segments

        updates, _ = self.verifier.verify(extraction.status_updates, transcript)

        return MeetingRecord(
            meeting_id=transcript.meeting_id, project_id=transcript.project_id,
            meeting_date=transcript.meeting_date, title=transcript.title,
            commitments=commitments, decisions=decisions, status_updates=updates,
            rejected_items=report.rejected,
        )

    # -- scoring -----------------------------------------------------------

    def _score_meeting(
        self,
        index: int,
        record: MeetingRecord,
        manifest: ProjectManifest,
        result: ProjectResult,
        truth_to_prediction: dict[str, str],
    ) -> None:
        truth = [c for c in manifest.firm() if c.meeting_index == index]
        predicted_firm = [
            c for c in record.commitments if c.strength is CommitmentStrength.FIRM
        ]
        alignment = align_commitments(predicted_firm, truth)

        result.extraction.commitments += PrecisionRecall(
            true_positives=len(alignment.matched),
            false_positives=len(alignment.spurious),
            false_negatives=len(alignment.missed),
        )

        for pair in alignment.matched:
            truth_to_prediction[pair.truth.id] = pair.predicted.id
            result.extraction.assignee_total += 1
            result.extraction.assignee_correct += int(pair.assignee_correct)
            result.extraction.strength_total += 1
            result.extraction.strength_correct += int(pair.strength_correct)
            if pair.truth.deadline_date is not None:
                result.extraction.deadline_total += 1
                result.extraction.deadline_correct += int(pair.deadline_correct)

        # Musings must never be promoted to firm commitments.
        musings = [m for m in manifest.musings if m.meeting_index == index]
        result.extraction.musings_total += len(musings)
        for musing in musings:
            promoted = any(
                any(
                    fuzz.partial_ratio(e.quote.lower(), musing.spoken_quote.lower()) >= 85
                    for e in commitment.evidence
                )
                for commitment in predicted_firm
            )
            result.extraction.musings_promoted += int(promoted)

    def _score_tracking(
        self,
        ledger: Ledger,
        manifest: ProjectManifest,
        transcripts: list[Transcript],
        result: ProjectResult,
        truth_to_prediction: dict[str, str],
    ) -> None:
        """Run the planner as of just after the project, then grade its choices."""
        as_of = transcripts[-1].meeting_date + timedelta(days=7)
        evidence = (
            ManifestEvidenceProvider(manifest, as_of) if self.use_evidence else None
        )
        config = PlannerConfig(project_end=as_of + timedelta(days=30))
        report = Planner(config).plan(ledger, as_of, evidence)
        by_commitment = {action.commitment_id: action for action in report.actions}

        chase_actions = {ActionType.NUDGE, ActionType.ESCALATE, ActionType.MARK_DROPPED}

        for truth in manifest.commitments:
            action = self._action_for(truth, by_commitment, ledger, truth_to_prediction)

            if truth.should_be_chased:
                result.tracking.dropped_total += 1
                if action is not None and action.action in chase_actions:
                    result.tracking.dropped_caught += 1

            if truth.is_false_nag_target:
                result.tracking.nag_targets_total += 1
                if action is not None and action.is_outbound:
                    result.tracking.false_nags += 1

            if truth.fate is CommitmentFate.DELIVERED_SILENTLY:
                result.tracking.silent_deliveries_total += 1
                if action is not None and action.action is ActionType.MARK_DONE:
                    result.tracking.silent_deliveries_verified += 1

            if truth.fate is CommitmentFate.BLOCKED:
                result.tracking.blocked_total += 1
                if action is not None and action.action is ActionType.PROPAGATE_SLIP:
                    result.tracking.blocked_propagated += 1

        self._score_contradictions(ledger, manifest, result)

    @staticmethod
    def _score_contradictions(ledger: Ledger, manifest: ProjectManifest, result: ProjectResult):
        """Score detected reversals against the true ones.

        Counting raw detections here produced a recall of 1.167 - impossible,
        and a giveaway that false positives were landing in the numerator. A
        detected pair only counts if it names the same two decisions the
        manifest says reverse each other, and each true pair can be claimed
        once. Everything else is a false positive, reported separately: telling
        a team its plan changed when it did not is its own kind of failure.
        """
        by_id = {d.id: d for d in manifest.decisions}
        truth_pairs = [
            (d.statement, by_id[d.reversed_by].statement)
            for d in manifest.decisions
            if d.reversed_by and d.reversed_by in by_id
        ]
        result.tracking.contradictions_total += len(truth_pairs)

        claimed: set[int] = set()
        for earlier, later in ledger.contradictions():
            match = None
            for index, (true_earlier, true_later) in enumerate(truth_pairs):
                if index in claimed:
                    continue
                if (
                    fuzz.token_set_ratio(earlier.statement.lower(), true_earlier.lower()) >= 70
                    and fuzz.token_set_ratio(later.statement.lower(), true_later.lower()) >= 70
                ):
                    match = index
                    break
            if match is None:
                result.tracking.contradiction_false_positives += 1
            else:
                claimed.add(match)
                result.tracking.contradictions_caught += 1

    @staticmethod
    def _action_for(truth, by_commitment, ledger: Ledger, truth_to_prediction):
        """Find the planned action for a ground-truth commitment, if any.

        Ledger merging can fold a restated commitment into an earlier one, so
        the originally-matched prediction id may no longer be in the ledger. Fall
        back to locating the surviving entry by owner and description.
        """
        predicted_id = truth_to_prediction.get(truth.id)
        if predicted_id and predicted_id in by_commitment:
            return by_commitment[predicted_id]

        best, best_score = None, 0.0
        for commitment in ledger.commitments:
            if commitment.assignee.display_name != truth.owner_name:
                continue
            score = fuzz.token_set_ratio(
                commitment.description.lower(), truth.description.lower()
            )
            if score > best_score:
                best, best_score = commitment, score
        if best is not None and best_score >= 80:
            return by_commitment.get(best.id)
        return None

    # -- many projects -----------------------------------------------------

    def run(self, seeds: list[int], weeks: int = 8) -> tuple[list[ProjectResult], dict]:
        results = []
        for seed in seeds:
            transcripts, manifest = ProjectGenerator(seed=seed, weeks=weeks).generate()
            log.info("Evaluating %s", manifest.project_id)
            results.append(self.run_project(transcripts, manifest))
        return results, aggregate(results)


def aggregate(results: list[ProjectResult]) -> dict:
    """Pool counts across projects rather than averaging per-project rates.

    Averaging rates would weight a project with two dropped commitments the same
    as one with twelve.
    """
    extraction = ExtractionScores()
    tracking = TrackingScores()
    calls = tokens = seconds = meetings = degraded = 0

    for result in results:
        extraction.commitments += result.extraction.commitments
        for attribute in (
            "assignee_correct", "assignee_total", "deadline_correct", "deadline_total",
            "strength_correct", "strength_total", "musings_promoted", "musings_total",
            "hallucinated", "proposed",
        ):
            setattr(
                extraction, attribute,
                getattr(extraction, attribute) + getattr(result.extraction, attribute),
            )
        for attribute in (
            "dropped_caught", "dropped_total", "false_nags", "nag_targets_total",
            "contradictions_caught", "contradictions_total",
            "contradiction_false_positives",
            "silent_deliveries_verified", "silent_deliveries_total",
            "blocked_propagated", "blocked_total",
        ):
            setattr(
                tracking, attribute,
                getattr(tracking, attribute) + getattr(result.tracking, attribute),
            )
        calls += result.llm_calls
        tokens += result.total_tokens
        seconds += result.wall_seconds
        meetings += result.meetings
        degraded += result.degraded_calls

    return {
        "projects": len(results),
        "meetings": meetings,
        "extraction": extraction.as_dict(),
        "tracking": tracking.as_dict(),
        "cost": {
            "llm_calls": calls,
            "total_tokens": tokens,
            "tokens_per_meeting": round(tokens / meetings) if meetings else 0,
            "wall_seconds": round(seconds, 1),
            "degraded_calls": degraded,
            "usd": 0.0,
        },
    }
