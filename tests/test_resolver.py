from __future__ import annotations

from datetime import date

from quorum.agents.resolver import AssigneeGuess, Resolver, ResolverConfig
from quorum.llm.router import LLMResponse
from quorum.models import Assignee, Commitment, Deadline, Evidence


class FakeRouter:
    def __init__(self, guesses: list[AssigneeGuess]) -> None:
        self.guesses = list(guesses)
        self.calls = 0

    def structured(self, prompt, schema, **kwargs):
        self.calls += 1
        return self.guesses.pop(0), LLMResponse(text="{}", model="fake", provider="fake")


def make(mention: str | None, utterance_id: str, deadline: str | None = None) -> Commitment:
    return Commitment(
        description="do the thing",
        assignee=Assignee(raw_mention=mention),
        deadline=Deadline(raw_text=deadline),
        evidence=[Evidence(utterance_id=utterance_id, quote="some verbatim quote here")],
    )


def offline() -> Resolver:
    return Resolver(config=ResolverConfig(use_llm_fallback=False))


# --- first person: the common case, and free ------------------------------


def test_first_person_resolves_to_the_speaker_of_the_cited_line(transcript):
    """"I'll have the spec to you by Friday" - said by Yug, so owned by Yug.
    No model call needed, which is the point."""
    commitment = make("I", "utt_1")
    stats = offline().resolve([commitment], transcript)

    assert commitment.assignee.display_name == "Yug Verma"
    assert commitment.assignee.email == "yug@example.com"
    assert commitment.assignee.confidence >= 0.9
    assert stats.assignee_deterministic == 1
    assert stats.llm_calls == 0


def test_missing_mention_falls_back_to_the_speaker(transcript):
    commitment = make(None, "utt_3")
    offline().resolve([commitment], transcript)
    assert commitment.assignee.display_name == "Sam Okafor"


# --- named people ----------------------------------------------------------


def test_named_mention_matches_the_roster(transcript):
    commitment = make("Sam", "utt_2")
    offline().resolve([commitment], transcript)
    assert commitment.assignee.display_name == "Sam Okafor"


def test_full_name_matches(transcript):
    commitment = make("Priya Raghavan", "utt_0")
    offline().resolve([commitment], transcript)
    assert commitment.assignee.speaker_id == "spk_priya"


# --- second person ---------------------------------------------------------


def test_you_resolves_to_the_person_named_in_the_line(transcript):
    """"Sam, can you review it once it lands?" - "you" is Sam."""
    commitment = make("you", "utt_2")
    offline().resolve([commitment], transcript)
    assert commitment.assignee.display_name == "Sam Okafor"


def test_you_without_a_name_resolves_to_whoever_answered(transcript):
    """"Where are we on the ingestion API?" - Yug answers, so Yug is addressed."""
    commitment = make("you", "utt_0")
    offline().resolve([commitment], transcript)
    assert commitment.assignee.display_name == "Yug Verma"


# --- deliberate non-resolution ---------------------------------------------


def test_collective_mentions_stay_unowned(transcript):
    """A collective is not an owner. Picking someone would nag the wrong person."""
    for mention in ("we", "someone", "the team", "everybody"):
        commitment = make(mention, "utt_4")
        stats = offline().resolve([commitment], transcript)

        assert commitment.assignee.speaker_id is None
        assert "collective" in commitment.assignee.unresolved_reason
        assert stats.assignee_unresolved == 1


def test_unowned_commitment_is_not_actionable(transcript):
    commitment = make("someone", "utt_5")
    offline().resolve([commitment], transcript)
    assert not commitment.is_actionable


def test_unknown_name_without_fallback_is_unresolved(transcript):
    commitment = make("Dmitri", "utt_0")
    stats = offline().resolve([commitment], transcript)

    assert commitment.assignee.speaker_id is None
    assert stats.assignee_unresolved == 1


# --- model fallback --------------------------------------------------------


def test_ambiguous_mention_falls_back_to_a_model(transcript):
    router = FakeRouter([AssigneeGuess(speaker_name="Sam Okafor", confidence=0.9)])
    resolver = Resolver(router=router, config=ResolverConfig(use_llm_fallback=True))
    commitment = make("the reviewer", "utt_2")

    stats = resolver.resolve([commitment], transcript)

    assert commitment.assignee.display_name == "Sam Okafor"
    assert stats.assignee_llm == 1 and router.calls == 1


def test_low_confidence_guess_is_discarded(transcript):
    """An unowned commitment gets surfaced to a human; a wrongly-owned one
    silently nags an innocent colleague."""
    router = FakeRouter([AssigneeGuess(speaker_name="Sam Okafor", confidence=0.2)])
    resolver = Resolver(router=router, config=ResolverConfig(use_llm_fallback=True))
    commitment = make("the reviewer", "utt_2")

    resolver.resolve([commitment], transcript)
    assert commitment.assignee.speaker_id is None


def test_null_guess_is_respected(transcript):
    router = FakeRouter([AssigneeGuess(speaker_name=None, confidence=0.0)])
    resolver = Resolver(router=router, config=ResolverConfig(use_llm_fallback=True))
    commitment = make("that person", "utt_2")

    resolver.resolve([commitment], transcript)
    assert commitment.assignee.speaker_id is None


def test_first_person_never_reaches_the_model(transcript):
    """Cost control: the most common phrasing must not spend a token."""
    router = FakeRouter([])
    resolver = Resolver(router=router, config=ResolverConfig(use_llm_fallback=True))
    resolver.resolve([make("I", "utt_1"), make("Sam", "utt_2")], transcript)
    assert router.calls == 0


def test_model_failure_does_not_sink_the_run(transcript):
    class Broken:
        def structured(self, *a, **k):
            raise RuntimeError("provider down")

    resolver = Resolver(router=Broken(), config=ResolverConfig(use_llm_fallback=True))
    stats = resolver.resolve([make("the reviewer", "utt_2")], transcript)
    assert stats.assignee_unresolved == 1


# --- deadlines wired through -----------------------------------------------


def test_deadline_is_resolved_against_the_meeting_date(transcript):
    commitment = make("I", "utt_1", deadline="by Friday")
    offline().resolve([commitment], transcript)

    assert commitment.deadline.resolved == date(2026, 3, 13)
    assert commitment.deadline.raw_text == "by Friday", "the spoken form is kept for scoring"


def test_known_events_anchor_deadlines(transcript):
    commitment = make("I", "utt_1", deadline="before the demo")
    offline().resolve([commitment], transcript, known_events={"demo": date(2026, 3, 20)})
    assert commitment.deadline.resolved == date(2026, 3, 19)


def test_missing_deadline_counts_as_unresolved(transcript):
    commitment = make("I", "utt_1", deadline=None)
    stats = offline().resolve([commitment], transcript)

    assert commitment.deadline.resolved is None
    assert stats.deadline_unresolved == 1


# --- statistics -------------------------------------------------------------


def test_deterministic_rate_is_reported(transcript):
    router = FakeRouter([AssigneeGuess(speaker_name="Sam Okafor", confidence=0.9)])
    resolver = Resolver(router=router, config=ResolverConfig(use_llm_fallback=True))

    stats = resolver.resolve(
        [make("I", "utt_1"), make("Sam", "utt_2"), make("the reviewer", "utt_2")], transcript
    )

    assert stats.assignee_deterministic == 2 and stats.assignee_llm == 1
    assert abs(stats.deterministic_rate - 2 / 3) < 1e-6
    assert stats.as_dict()["total"] == 3
