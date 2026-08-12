"""AMI parser tests against synthetic NXT-format XML.

The real corpus needs a manually accepted licence, so it cannot live in CI.
These fixtures reproduce the format instead - word files with timings, segment
files that reference words by *range* pointers, and an abstractive summary with
the ABSTRACT/DECISIONS/PROBLEMS/ACTIONS sections. That covers the parsing that
actually goes wrong; a real download then only has to confirm the shape.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from quorum.data.ami import AmiCorpus, _detokenise, find_corpus_root

NITE = 'xmlns:nite="http://nite.sourceforge.net/"'


def write_words(root: Path, meeting: str, speaker: str, words: list[tuple[str, float]]) -> None:
    lines = [f'<nite:root {NITE} nite:id="{meeting}.{speaker}.words">']
    for index, (text, start) in enumerate(words, start=1):
        lines.append(
            f'  <w nite:id="{meeting}.{speaker}.words{index}" '
            f'starttime="{start}" endtime="{start + 0.4}">{text}</w>'
        )
    lines.append("</nite:root>")
    path = root / "words" / f"{meeting}.{speaker}.words.xml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_segments(
    root: Path, meeting: str, speaker: str, spans: list[tuple[int, int, float]]
) -> None:
    lines = [f'<nite:root {NITE} nite:id="{meeting}.{speaker}.segments">']
    for index, (first, last, start) in enumerate(spans, start=1):
        href = (
            f"{meeting}.{speaker}.words.xml#id({meeting}.{speaker}.words{first})"
            f"..id({meeting}.{speaker}.words{last})"
        )
        lines.append(
            f'  <segment nite:id="{meeting}.{speaker}.seg{index}" starttime="{start}">'
            f'<nite:child href="{href}"/></segment>'
        )
    lines.append("</nite:root>")
    path = root / "segments" / f"{meeting}.{speaker}.segments.xml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_abstractive(root: Path, meeting: str) -> None:
    xml = f"""<nite:root {NITE} nite:id="{meeting}.abssumm">
  <abstract>
    <sentence nite:id="s1">The team discussed the ingestion API.</sentence>
  </abstract>
  <decisions>
    <sentence nite:id="s2">They decided to use Postgres.</sentence>
  </decisions>
  <problems>
    <sentence nite:id="s3">The staging cluster runs out of memory.</sentence>
  </problems>
  <actions>
    <sentence nite:id="s4">The project manager will send the specification.</sentence>
    <sentence nite:id="s5">The engineer will raise the pod memory limits.</sentence>
  </actions>
</nite:root>"""
    path = root / "abstractive" / f"{meeting}.abssumm.xml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(xml, encoding="utf-8")


@pytest.fixture
def corpus(tmp_path) -> Path:
    root = tmp_path / "ami_public_manual_1.6.2"
    write_words(
        root, "ES2002a", "A",
        [("Where", 1.0), ("are", 1.5), ("we", 2.0), ("on", 2.5), ("the", 3.0),
         ("spec", 3.5), ("?", 3.9), ("Okay", 20.0), ("thanks", 20.5), (".", 20.9)],
    )
    write_segments(root, "ES2002a", "A", [(1, 7, 1.0), (8, 10, 20.0)])
    write_words(
        root, "ES2002a", "B",
        [("I", 10.0), ("'ll", 10.2), ("send", 10.5), ("it", 11.0), ("Friday", 11.5), (".", 11.9)],
    )
    write_segments(root, "ES2002a", "B", [(1, 6, 10.0)])
    write_abstractive(root, "ES2002a")
    return root


# --- locating the corpus ----------------------------------------------------


def test_find_corpus_root_at_the_top(corpus):
    assert find_corpus_root(corpus) == corpus


def test_find_corpus_root_when_nested(corpus, tmp_path):
    """People unzip into inconsistent shapes; the parser should search."""
    assert find_corpus_root(tmp_path) == corpus


def test_missing_corpus_raises_a_useful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="words"):
        AmiCorpus(tmp_path / "nothing-here")


# --- discovery --------------------------------------------------------------


def test_lists_meetings_and_speakers(corpus):
    ami = AmiCorpus(corpus)
    assert ami.meeting_ids() == ["ES2002a"]
    assert ami.speakers_for("ES2002a") == ["A", "B"]


# --- transcript assembly ----------------------------------------------------


def test_speakers_are_interleaved_by_time(corpus):
    """Each speaker lives in a separate file. A meeting is their time-ordered
    merge - getting this wrong yields one speaker's whole monologue, then the
    next, which destroys every commitment's context."""
    meeting = AmiCorpus(corpus).load("ES2002a")
    texts = [u.text for u in meeting.transcript.utterances]

    assert len(texts) == 3
    assert texts[0].startswith("Where are we")
    assert texts[1].startswith("I'll send it Friday")
    assert texts[2].startswith("Okay thanks")


def test_speaker_attribution_is_preserved(corpus):
    meeting = AmiCorpus(corpus).load("ES2002a")
    utterances = meeting.transcript.utterances
    assert utterances[0].speaker_id == "spk_A"
    assert utterances[1].speaker_id == "spk_B"


def test_range_pointers_expand_to_every_word(corpus):
    """A segment names only its first and last word. Naive parsing keeps two
    words and silently drops everything between them."""
    meeting = AmiCorpus(corpus).load("ES2002a")
    first = meeting.transcript.utterances[0].text
    for word in ("Where", "are", "we", "on", "the", "spec"):
        assert word in first


def test_indices_are_contiguous_after_the_merge(corpus):
    meeting = AmiCorpus(corpus).load("ES2002a")
    indices = [u.index for u in meeting.transcript.utterances]
    assert indices == list(range(len(indices)))


def test_transcript_is_marked_as_ami_sourced(corpus):
    """Metrics must never pool synthetic and real results into one number."""
    meeting = AmiCorpus(corpus).load("ES2002a")
    assert meeting.transcript.source == "ami"


def test_missing_meeting_raises(corpus):
    with pytest.raises(FileNotFoundError):
        AmiCorpus(corpus).load("XX9999z")


# --- fallback when segments are absent --------------------------------------


def test_words_are_grouped_by_pauses_when_segments_are_missing(tmp_path):
    """A partial download should still yield usable transcripts."""
    root = tmp_path / "ami"
    write_words(
        root, "IS1000a", "A",
        [("Hello", 1.0), ("everyone", 1.4), ("Right", 30.0), ("let", 30.4), ("us", 30.8)],
    )
    meeting = AmiCorpus(root).load("IS1000a")

    assert len(meeting.transcript.utterances) == 2, "a 29s gap is an utterance boundary"
    assert meeting.transcript.utterances[0].text == "Hello everyone"


# --- abstractive summaries (the ground truth) -------------------------------


def test_action_items_are_extracted(corpus):
    meeting = AmiCorpus(corpus).load("ES2002a")
    assert meeting.has_ground_truth
    assert len(meeting.actions) == 2
    assert "send the specification" in meeting.actions[0]


def test_all_four_sections_are_separated(corpus):
    meeting = AmiCorpus(corpus).load("ES2002a")
    assert meeting.abstract and meeting.decisions and meeting.problems and meeting.actions
    assert "Postgres" in meeting.decisions[0]
    assert "memory" in meeting.problems[0]


def test_sections_do_not_bleed_into_each_other(corpus):
    """The HuggingFace mirror of AMI collapses these into one summary field,
    which is why it cannot be used as ground truth here."""
    meeting = AmiCorpus(corpus).load("ES2002a")
    assert not any("Postgres" in action for action in meeting.actions)


def test_alternative_section_tag_names_are_accepted(tmp_path):
    """Tag naming varies across releases; matching one exact string is fragile."""
    root = tmp_path / "ami"
    write_words(root, "TS3003a", "A", [("Hello", 1.0), ("there", 1.4)])
    path = root / "abstractive" / "TS3003a.abssumm.xml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'<nite:root {NITE}><actionitems><sentence>Someone will do the thing.</sentence>'
        "</actionitems></nite:root>",
        encoding="utf-8",
    )
    assert AmiCorpus(root).load("TS3003a").actions == ["Someone will do the thing."]


def test_meeting_without_a_summary_still_loads(tmp_path):
    root = tmp_path / "ami"
    write_words(root, "EN2001a", "A", [("Hello", 1.0), ("there", 1.4)])
    meeting = AmiCorpus(root).load("EN2001a")

    assert meeting.transcript.utterances
    assert not meeting.has_ground_truth


def test_load_all_skips_meetings_without_ground_truth(corpus, tmp_path):
    write_words(corpus, "EN2001a", "A", [("Hello", 1.0), ("there", 1.4)])
    ami = AmiCorpus(corpus)

    assert [m.meeting_id for m in ami.load_all()] == ["ES2002a"]
    assert len(ami.load_all(require_actions=False)) == 2


def test_load_all_respects_the_limit(corpus):
    assert len(AmiCorpus(corpus).load_all(limit=1)) == 1


# --- detokenisation ---------------------------------------------------------


@pytest.mark.parametrize(
    "tokens,expected",
    [
        (["Hello", "world", "."], "Hello world."),
        (["So", ",", "what", "do", "you", "think", "?"], "So, what do you think?"),
        (["I", "'ll", "send", "it"], "I'll send it"),
        (["It", "'s", "done", "n't", "worry"], "It's donen't worry"),
        ([], ""),
    ],
)
def test_detokenise(tokens, expected):
    """AMI stores punctuation as separate tokens. A naive space-join produces
    "So , what do you think ?", which changes how a model reads the sentence."""
    assert _detokenise(tokens) == expected
