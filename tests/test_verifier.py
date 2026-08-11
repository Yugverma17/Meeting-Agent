from __future__ import annotations

from quorum.agents.verifier import GroundingVerifier, VerificationConfig, normalise
from quorum.models import Commitment, Evidence


def commitment(quote: str, utterance_id: str = "utt_1", description: str = "x") -> Commitment:
    return Commitment(
        description=description,
        evidence=[Evidence(utterance_id=utterance_id, quote=quote)],
    )


# --- the happy path --------------------------------------------------------


def test_exact_quote_is_verified(transcript):
    item = commitment("I'll have the spec document to you by Friday.")
    kept, report = GroundingVerifier().verify([item], transcript)

    assert len(kept) == 1
    assert report.accepted == 1 and report.rejected == 0
    assert kept[0].evidence[0].verified
    assert kept[0].evidence[0].match_score >= 85


def test_verification_backfills_speaker_and_timestamp(transcript):
    """Provenance the model never sees is filled in from the transcript, so it
    cannot be fabricated."""
    item = commitment("I'll have the spec document to you by Friday.")
    kept, _ = GroundingVerifier().verify([item], transcript)

    evidence = kept[0].evidence[0]
    assert evidence.speaker_id == "spk_yug"
    assert evidence.timestamp == "00:15"


def test_partial_quote_within_an_utterance_verifies(transcript):
    item = commitment("spec document to you by Friday")
    kept, _ = GroundingVerifier().verify([item], transcript)
    assert len(kept) == 1


def test_casing_and_whitespace_differences_are_tolerated(transcript):
    item = commitment("i'll   have  THE spec document to you by friday")
    kept, _ = GroundingVerifier().verify([item], transcript)
    assert len(kept) == 1


# --- the gate ---------------------------------------------------------------


def test_invented_quote_is_rejected(transcript):
    """The whole point: a commitment nobody made does not survive."""
    item = commitment(
        "I will personally rewrite the billing system by Tuesday.",
        description="rewrite billing",
    )
    kept, report = GroundingVerifier().verify([item], transcript)

    assert kept == []
    assert report.rejected == 1
    assert report.hallucination_rate == 1.0
    assert report.rejections[0][0] == "rewrite billing"
    assert "not found" in report.rejections[0][1]


def test_short_quote_is_rejected_as_a_loophole(transcript):
    """"I'll" partial-matches almost anything; a short quote is not evidence."""
    item = commitment("I'll")
    kept, report = GroundingVerifier().verify([item], transcript)

    assert kept == []
    assert "shorter than" in report.rejections[0][1]


def test_paraphrase_is_rejected(transcript):
    """Verbatim means verbatim - a plausible restatement must not pass."""
    item = commitment("Yug agreed to deliver the specification before the weekend")
    kept, _ = GroundingVerifier().verify([item], transcript)
    assert kept == []


def test_hallucination_rate_across_a_mixed_batch(transcript):
    items = [
        commitment("I'll have the spec document to you by Friday."),
        commitment("Sure, I'll take a look over the weekend."),
        commitment("We will migrate everything to Kubernetes next sprint."),
        commitment("I promise to double the budget immediately."),
    ]
    kept, report = GroundingVerifier().verify(items, transcript)

    assert len(kept) == 2
    assert report.proposed == 4 and report.accepted == 2 and report.rejected == 2
    assert report.hallucination_rate == 0.5


# --- citation repair --------------------------------------------------------


def test_correct_quote_with_wrong_utterance_id_is_repaired(transcript):
    """Models often quote accurately but index wrongly. Throwing those away
    would discard real findings."""
    item = commitment("I'll have the spec document to you by Friday.", utterance_id="utt_5")
    kept, report = GroundingVerifier().verify([item], transcript)

    assert len(kept) == 1
    assert report.repaired == 1
    assert kept[0].evidence[0].utterance_id == "utt_1", "citation re-pointed to the real source"


def test_unknown_utterance_id_with_real_quote_is_repaired(transcript):
    item = commitment("I'll have the spec document to you by Friday.", utterance_id="utt_nope")
    kept, report = GroundingVerifier().verify([item], transcript)
    assert len(kept) == 1 and report.repaired == 1


def test_repair_can_be_disabled(transcript):
    item = commitment("I'll have the spec document to you by Friday.", utterance_id="utt_5")
    verifier = GroundingVerifier(VerificationConfig(allow_repair=False))
    kept, report = verifier.verify([item], transcript)

    assert kept == []
    assert report.repaired == 0


# --- multi-citation items ---------------------------------------------------


def test_item_survives_if_any_citation_holds_but_bad_ones_are_dropped(transcript):
    item = Commitment(
        description="review the spec",
        evidence=[
            Evidence(utterance_id="utt_3", quote="Sure, I'll take a look over the weekend."),
            Evidence(utterance_id="utt_3", quote="and I'll also rewrite the entire backend"),
        ],
    )
    kept, report = GroundingVerifier().verify([item], transcript)

    assert len(kept) == 1
    assert len(kept[0].evidence) == 1, "the unsupported citation must be stripped"
    assert kept[0].is_grounded


def test_item_dies_when_every_citation_fails(transcript):
    item = Commitment(
        description="fictional work",
        evidence=[
            Evidence(utterance_id="utt_0", quote="we agreed to acquire a competitor"),
            Evidence(utterance_id="utt_1", quote="and to relocate the office to Berlin"),
        ],
    )
    kept, _ = GroundingVerifier().verify([item], transcript)
    assert kept == []


# --- misc -------------------------------------------------------------------


def test_empty_batch_reports_zero_rate(transcript):
    kept, report = GroundingVerifier().verify([], transcript)
    assert kept == [] and report.hallucination_rate == 0.0


def test_threshold_is_configurable(transcript):
    loose = GroundingVerifier(VerificationConfig(min_match_score=40.0))
    item = commitment("Yug said he would send the specification by Friday")
    kept, _ = loose.verify([item], transcript)
    assert len(kept) == 1, "a permissive threshold should admit near-paraphrases"


def test_normalise_collapses_whitespace_and_case():
    assert normalise("  Hello   WORLD\n\tagain ") == "hello world again"


def test_report_serialises_for_metrics(transcript):
    items = [commitment("I'll have the spec document to you by Friday."), commitment("nonsense")]
    _, report = GroundingVerifier().verify(items, transcript)
    payload = report.as_dict()
    assert payload["proposed"] == 2 and payload["accepted"] == 1
    assert 0.0 <= payload["hallucination_rate"] <= 1.0
