"""AMI Meeting Corpus ingestion.

Why this matters: every extraction number in this project currently comes from
generated dialogue. Synthetic meetings are clean - no crosstalk, no
half-sentences, no disfluencies - so those scores are an upper bound. AMI is real
speech with real annotators, and it is the only way to know how much of the
performance survives contact with reality.

The corpus ships as NXT-format XML, one file per speaker per meeting:

    ami_public_manual_1.6.2/
      words/       ES2002a.A.words.xml       word-level tokens with timings
      segments/    ES2002a.A.segments.xml    utterance boundaries
      abstractive/ ES2002a.abssumm.xml       ABSTRACT / DECISIONS / PROBLEMS / ACTIONS

The ACTIONS section is the ground truth we care about.

**A caveat that shapes how the numbers should be read.** Inter-annotator
agreement on action-item labelling is around kappa 0.36 - annotators barely agree
with each other on what counts as one. Raw accuracy against a single annotator is
therefore close to meaningless as a target, and a system scoring "poorly" may be
disagreeing no more than two humans would. Results should be reported alongside
that ceiling, not against an implied 1.0.

AMI actions are also *abstractive* - "The project manager will send the minutes"
rather than a verbatim quote - so alignment against them is necessarily fuzzy in
a way the synthetic benchmark's exact spans are not.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from xml.etree import ElementTree

from quorum.models import Speaker, Transcript, Utterance

log = logging.getLogger(__name__)

NITE_NS = "http://nite.sourceforge.net/"
_MEETING_RE = re.compile(r"^([A-Za-z]{2}\d{4}[a-z])\.([A-Z])\.")

# Section headings inside an abstractive summary. Tag names vary a little across
# releases, so each is matched by several aliases rather than one exact string.
SECTION_ALIASES = {
    "abstract": ("abstract", "abstractsummary", "summary"),
    "decisions": ("decisions", "decision"),
    "problems": ("problems", "problemsissues", "problems_issues", "issues"),
    "actions": ("actions", "actionitems", "action_items", "action"),
}


def _local(tag: str) -> str:
    """Strip any XML namespace, lowercased."""
    return tag.rsplit("}", 1)[-1].lower()


def _nite_id(element: ElementTree.Element) -> str | None:
    return element.get(f"{{{NITE_NS}}}id") or element.get("id")


def _float_or_none(value: str | None) -> float | None:
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


@dataclass
class AmiMeeting:
    meeting_id: str
    transcript: Transcript
    abstract: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    """The ground truth for action-item extraction."""

    @property
    def has_ground_truth(self) -> bool:
        return bool(self.actions)


def find_corpus_root(start: Path) -> Path | None:
    """Locate the directory containing `words/`, wherever the zip was unpacked.

    People unzip into inconsistent shapes - sometimes the archive's own folder,
    sometimes flat - so this searches rather than demanding an exact path.
    """
    start = Path(start)
    if (start / "words").is_dir():
        return start
    for candidate in sorted(start.rglob("words")):
        if candidate.is_dir() and any(candidate.glob("*.words.xml")):
            return candidate.parent
    return None


class AmiCorpus:
    def __init__(self, root: Path, meeting_date: date | None = None) -> None:
        resolved = find_corpus_root(root)
        if resolved is None:
            raise FileNotFoundError(
                f"No AMI corpus under {root}. Expected a 'words/' directory containing "
                "*.words.xml. See `python -m quorum.cli ami --help` for download steps."
            )
        self.root = resolved
        self.meeting_date = meeting_date or date(2005, 1, 1)
        """AMI carries no real meeting dates. A fixed placeholder keeps deadline
        resolution deterministic across runs; relative phrases resolve against
        it consistently, and absolute-date accuracy is simply not scoreable on
        this corpus."""

    # -- discovery ---------------------------------------------------------

    def meeting_ids(self, annotated_only: bool = False) -> list[str]:
        """Meetings in the corpus, optionally only the scoreable ones.

        Not every meeting with a transcript has an annotator's summary: the
        `EN*` series ships words and no `abstractive/` file at all, and it sorts
        first, so an evaluation that took "the first N meetings" evaluated
        against zero ground truth and reported precision 0.000 - a number that
        looks like a devastating result and is actually an empty comparison.
        """
        found = set()
        for path in (self.root / "words").glob("*.words.xml"):
            match = _MEETING_RE.match(path.name)
            if match:
                found.add(match.group(1))
        if annotated_only:
            found = {mid for mid in found if self.has_annotations(mid)}
        return sorted(found)

    def has_annotations(self, meeting_id: str) -> bool:
        """Whether an annotator wrote a summary for this meeting."""
        folder = self.root / "abstractive"
        return any(folder.glob(f"{meeting_id}.*abssumm.xml")) if folder.exists() else False

    def speakers_for(self, meeting_id: str) -> list[str]:
        letters = set()
        for path in (self.root / "words").glob(f"{meeting_id}.*.words.xml"):
            match = _MEETING_RE.match(path.name)
            if match:
                letters.add(match.group(2))
        return sorted(letters)

    # -- loading -----------------------------------------------------------

    def load(self, meeting_id: str) -> AmiMeeting:
        speaker_letters = self.speakers_for(meeting_id)
        if not speaker_letters:
            raise FileNotFoundError(f"No word files for meeting {meeting_id}")

        speakers = [
            Speaker(id=f"spk_{letter}", display_name=f"Speaker {letter}", aliases=[letter])
            for letter in speaker_letters
        ]

        collected: list[tuple[float, str, str]] = []
        for letter in speaker_letters:
            collected.extend(self._utterances_for(meeting_id, letter))

        # Speakers are stored in separate files; a meeting is the time-ordered
        # interleaving of all of them.
        collected.sort(key=lambda row: row[0])
        utterances = [
            Utterance(
                id=f"{meeting_id}_u{index}", index=index, speaker_id=f"spk_{letter}",
                text=text, start_s=start,
            )
            for index, (start, letter, text) in enumerate(collected)
        ]

        transcript = Transcript(
            meeting_id=meeting_id, title=f"AMI {meeting_id}",
            meeting_date=self.meeting_date, speakers=speakers,
            utterances=utterances, source="ami",
        )
        sections = self._abstractive(meeting_id)
        return AmiMeeting(
            meeting_id=meeting_id, transcript=transcript,
            abstract=sections.get("abstract", []),
            decisions=sections.get("decisions", []),
            problems=sections.get("problems", []),
            actions=sections.get("actions", []),
        )

    def load_all(self, limit: int | None = None, require_actions: bool = True) -> list[AmiMeeting]:
        meetings = []
        # Filter on the filesystem before parsing. The unannotated `EN*` series
        # sorts first and is ~100 meetings, so parsing them all to discard them
        # made a two-meeting evaluation take minutes of pure waste.
        for meeting_id in self.meeting_ids(annotated_only=require_actions):
            try:
                meeting = self.load(meeting_id)
            except (FileNotFoundError, ElementTree.ParseError) as exc:
                log.warning("Skipping %s: %s", meeting_id, exc)
                continue
            if require_actions and not meeting.has_ground_truth:
                continue
            meetings.append(meeting)
            if limit and len(meetings) >= limit:
                break
        return meetings

    # -- parsing -----------------------------------------------------------

    def _word_text(self, meeting_id: str, letter: str) -> dict[str, tuple[str, float | None]]:
        """Word id -> (text, start time)."""
        path = self.root / "words" / f"{meeting_id}.{letter}.words.xml"
        words: dict[str, tuple[str, float | None]] = {}
        if not path.exists():
            return words

        for element in ElementTree.parse(path).getroot():
            if _local(element.tag) not in ("w", "word"):
                continue  # skip vocalsound, disfmarker, gap, pause markers
            word_id = _nite_id(element)
            text = (element.text or "").strip()
            if word_id and text:
                words[word_id] = (text, _float_or_none(element.get("starttime")))
        return words

    def _utterances_for(self, meeting_id: str, letter: str) -> list[tuple[float, str, str]]:
        """(start time, speaker letter, text) for one speaker.

        Prefers the annotated segment boundaries. Falls back to grouping words by
        silence when segments are absent, so a partial download still yields
        usable transcripts instead of nothing.
        """
        words = self._word_text(meeting_id, letter)
        if not words:
            return []

        segments_path = self.root / "segments" / f"{meeting_id}.{letter}.segments.xml"
        if segments_path.exists():
            rows = self._from_segments(segments_path, words, letter)
            if rows:
                return rows
            log.debug("No usable segments for %s.%s; grouping by pauses", meeting_id, letter)
        return self._from_pauses(words, letter)

    def _from_segments(
        self, path: Path, words: dict[str, tuple[str, float | None]], letter: str
    ) -> list[tuple[float, str, str]]:
        rows: list[tuple[float, str, str]] = []
        for segment in ElementTree.parse(path).getroot():
            if _local(segment.tag) != "segment":
                continue
            ids = self._referenced_ids(segment)
            tokens = [words[i][0] for i in ids if i in words]
            if not tokens:
                continue
            start = _float_or_none(segment.get("starttime"))
            if start is None:
                start = next((words[i][1] for i in ids if i in words and words[i][1]), 0.0)
            rows.append((start or 0.0, letter, _detokenise(tokens)))
        return rows

    @staticmethod
    def _referenced_ids(segment: ElementTree.Element) -> list[str]:
        """Expand NXT child pointers.

        A pointer is either a single id or an inclusive range:
            ES2002a.A.words.xml#id(ES2002a.A.words1)
            ES2002a.A.words.xml#id(ES2002a.A.words1)..id(ES2002a.A.words9)

        Ranges are the reason this needs real handling: a segment usually names
        only its first and last word, so naive parsing loses everything between.
        """
        ids: list[str] = []
        for child in segment:
            href = child.get("href")
            if not href:
                continue
            found = re.findall(r"id\(([^)]+)\)", href)
            if len(found) == 2:
                first, last = found
                prefix_a, num_a = _split_trailing_number(first)
                prefix_b, num_b = _split_trailing_number(last)
                if prefix_a == prefix_b and num_a is not None and num_b is not None:
                    ids.extend(f"{prefix_a}{n}" for n in range(num_a, num_b + 1))
                    continue
            ids.extend(found)
        return ids

    @staticmethod
    def _from_pauses(
        words: dict[str, tuple[str, float | None]], letter: str, gap_s: float = 1.0
    ) -> list[tuple[float, str, str]]:
        ordered = sorted(
            ((t, s) for t, s in words.values() if s is not None), key=lambda row: row[1]
        )
        if not ordered:
            return [(0.0, letter, _detokenise([t for t, _ in words.values()]))]

        rows, buffer, start, previous = [], [], ordered[0][1], ordered[0][1]
        for text, when in ordered:
            if when - previous > gap_s and buffer:
                rows.append((start, letter, _detokenise(buffer)))
                buffer, start = [], when
            buffer.append(text)
            previous = when
        if buffer:
            rows.append((start, letter, _detokenise(buffer)))
        return rows

    def _abstractive(self, meeting_id: str) -> dict[str, list[str]]:
        """Pull the summary sections, tolerating naming differences."""
        directory = self.root / "abstractive"
        if not directory.is_dir():
            return {}
        candidates = list(directory.glob(f"{meeting_id}.*.xml"))
        if not candidates:
            return {}

        sections: dict[str, list[str]] = {}
        root = ElementTree.parse(candidates[0]).getroot()
        for element in root.iter():
            canonical = self._canonical_section(_local(element.tag))
            if canonical is None:
                continue
            sentences = [
                (child.text or "").strip()
                for child in element
                if _local(child.tag) in ("sentence", "s") and (child.text or "").strip()
            ]
            if not sentences and (element.text or "").strip():
                sentences = [element.text.strip()]
            if sentences:
                sections.setdefault(canonical, []).extend(sentences)
        return sections

    @staticmethod
    def _canonical_section(tag: str) -> str | None:
        for canonical, aliases in SECTION_ALIASES.items():
            if tag in aliases:
                return canonical
        return None


def _split_trailing_number(identifier: str) -> tuple[str, int | None]:
    match = re.match(r"^(.*?)(\d+)$", identifier)
    if not match:
        return identifier, None
    return match.group(1), int(match.group(2))


def _detokenise(tokens: list[str]) -> str:
    """Rejoin word tokens into readable text.

    AMI stores punctuation as separate tokens, so a naive space-join produces
    "so , what do you think ?" - which changes how a model reads the sentence.
    """
    out = ""
    for token in tokens:
        if not token:
            continue
        if token in {",", ".", "?", "!", ";", ":", "'", "n't", "'s", "'re", "'ll", "'ve", "'d", "'m"}:
            out += token
        elif out and not out.endswith(("'", '"', "(")):
            out += " " + token
        else:
            out += token
    return out.strip()
