"""Detecting when a decision reverses an earlier one.

Teams change their minds and rarely announce it as a reversal. Week 2 settles on
Postgres; week 6 says "we're switching to Mongo" without anyone noting that a
prior decision just died. Nobody owns noticing this, which is exactly the kind
of gap an agent with persistent memory can fill.

Cost shape: one call per new decision, with all prior decisions in the prompt,
rather than a call per pair. Pairwise would be quadratic and would exhaust a
free-tier daily budget on a single project.

Reasoning is enabled here. It is one of only two places in the pipeline where
that is true - the judgement is genuinely multi-step (same topic? actually
incompatible? or just a refinement?) and the call volume is low enough to
afford it.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from quorum.llm.providers import ModelTier
from quorum.llm.router import Router, get_router
from quorum.models import Decision

log = logging.getLogger(__name__)


class ContradictionVerdict(BaseModel):
    reverses_index: int | None = Field(
        default=None,
        description="Index of the earlier decision this reverses, or null if none",
    )
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reasoning: str = ""


SYSTEM_PROMPT = """\
You decide whether a new decision reverses an earlier one.

A reversal means the two cannot both be true: they concern the SAME question and
choose DIFFERENT answers. "We're using Postgres" then "we're switching to Mongo"
is a reversal.

These are NOT reversals:
- decisions about different questions, however similar the wording
- a refinement that narrows an earlier choice without contradicting it
- restating or reaffirming an earlier decision

Return null unless the incompatibility is clear. A false reversal tells a team
its plan changed when it did not, which erodes trust in everything else the
system reports."""


class ContradictionDetector:
    def __init__(
        self,
        router: Router | None = None,
        tier: ModelTier = ModelTier.DEEP,
        min_confidence: float = 0.6,
    ) -> None:
        self._router = router
        self.tier = tier
        self.min_confidence = min_confidence
        self.calls = 0

    @property
    def router(self) -> Router:
        if self._router is None:
            self._router = get_router()
        return self._router

    def scan(self, decisions: list[Decision]) -> int:
        """Set `reverses` on any decision that contradicts an earlier one.

        Assumes chronological order. Returns how many reversals were found.
        """
        found = 0
        for index, decision in enumerate(decisions):
            prior = decisions[:index]
            if not prior:
                continue
            reversed_id = self._check(decision, prior)
            if reversed_id:
                decision.reverses = reversed_id
                found += 1
        return found

    def _check(self, decision: Decision, prior: list[Decision]) -> str | None:
        listing = "\n".join(
            f"[{i}] {earlier.statement}" for i, earlier in enumerate(prior)
        )
        prompt = (
            f"Earlier decisions:\n{listing}\n\n"
            f"New decision:\n{decision.statement}\n\n"
            "Does the new decision reverse any of the earlier ones? "
            "Give the index, or null."
        )

        try:
            verdict, _ = self.router.structured(
                prompt, ContradictionVerdict, system=SYSTEM_PROMPT,
                tier=self.tier, max_tokens=512, thinking=True,
                purpose="detect_contradiction",
            )
        except Exception as exc:  # noqa: BLE001 - never sink a run over this
            log.warning("Contradiction check failed: %s", exc)
            return None

        self.calls += 1
        if verdict.reverses_index is None or verdict.confidence < self.min_confidence:
            return None
        if not 0 <= verdict.reverses_index < len(prior):
            log.debug("Model returned out-of-range index %s", verdict.reverses_index)
            return None
        return prior[verdict.reverses_index].id
