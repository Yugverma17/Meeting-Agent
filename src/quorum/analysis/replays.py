"""Which parts of a lecture you went back and watched again.

Recording the screen rather than the video file means the capture is a record of
*how you watched*, not just what was said. Skipping leaves a gap, changing speed
bends the timeline — and replaying a section records the same speech twice.

That last one is the only one that carries information. Nobody rewinds the easy
part. A stretch you played three times is, with no further inference, the idea
that did not land, and it is knowable here and almost nowhere else: a tool
reading the transcript file cannot see it, because the file has no memory of
being read twice.

## Telling a replay from a speaker repeating themselves

This is the whole problem, and a naive duplicate search fails it completely.
Lecturers repeat *constantly* — one real recording contains "this is a substring
where C is the last character" four times in a row, and treating that as a
rewatch would fill the section with noise and make it worthless.

The discriminator is **length and contiguity**, not similarity. A replay is the
same audio transcribed twice, so it reproduces a long, *contiguous* stretch of
speech near-verbatim. Rhetorical repetition is a phrase. So:

- the transcript is cut into overlapping word windows,
- windows are matched all-against-all,
- and only an unbroken **diagonal run** of matches counts — window `i` matching
  `j`, `i+1` matching `j+1`, and so on for several windows together.

A repeated sentence produces one isolated match and is discarded. A replayed
minute produces a long diagonal and is kept. The two occurrences must also be
separated in time, so the overlap between adjacent windows cannot masquerade as
a repeat.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from quorum.models import Transcript

log = logging.getLogger(__name__)

WINDOW_WORDS = 24
"""Long enough that a window is a claim rather than a phrase."""

STEP_WORDS = 12
"""Half a window, so a replay starting mid-window still aligns somewhere."""

SIMILARITY = 86.0
"""Not 100: the same audio transcribed twice is *near*-identical, not identical.
Whisper differs run to run on exactly the passages a person replays, which are
the unclear ones."""

MIN_RUN = 3
"""Consecutive aligned windows required, and the whole precision/recall dial.

Three windows is roughly fifty words - about nineteen seconds of speech. Two
(~fourteen seconds) was tried, because it would catch the standard "back ten
seconds" rewind, and it produced a real false positive: a speaker saying a short
line five times in close succession concatenates into enough near-identical text
to align. Checking against one real lecture had suggested 2 was safe; a harsher
case showed it was not, which is why the value is 3.

So a very short rewind is missed. That is the correct direction to fail in: this
section is read as a claim about which ideas did not land, and inventing one is
worse than staying quiet."""

MIN_GAP_S = 25.0
"""How far apart two occurrences must sit before they can be a replay rather
than the window overlap seeing itself."""


@dataclass
class Replay:
    """A stretch of the lecture that was played more than once."""

    text: str
    times: list[float] = field(default_factory=list)
    """Capture-clock start of each occurrence, earliest first."""

    @property
    def count(self) -> int:
        return len(self.times)

    @property
    def first_at(self) -> str:
        return _stamp(self.times[0]) if self.times else "--:--"

    def summary(self, limit: int = 160) -> str:
        text = " ".join(self.text.split())
        return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


@dataclass
class _Window:
    index: int
    start_s: float
    text: str


def _stamp(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes:02d}:{secs:02d}"


def _windows(transcript: Transcript) -> list[_Window]:
    """Overlapping word windows across the whole transcript, with start times.

    Times are interpolated within an utterance rather than taken from its start:
    a merged utterance can span thirty seconds, and attributing every window
    inside it to the same instant would collapse the gaps this relies on.
    """
    words: list[tuple[str, float]] = []
    for utterance in transcript.utterances:
        tokens = utterance.text.split()
        if not tokens:
            continue
        start = utterance.start_s or 0.0
        end = utterance.end_s if utterance.end_s is not None else start
        span = max(end - start, 0.0)
        for position, token in enumerate(tokens):
            offset = span * position / len(tokens) if span else 0.0
            words.append((token, start + offset))

    windows = []
    for index, begin in enumerate(range(0, max(len(words) - WINDOW_WORDS + 1, 0), STEP_WORDS)):
        chunk = words[begin : begin + WINDOW_WORDS]
        windows.append(_Window(
            index=index,
            start_s=chunk[0][1],
            text=" ".join(token for token, _ in chunk),
        ))
    return windows


def _match_matrix(windows: list[_Window]) -> np.ndarray:
    """All-against-all similarity, as unsigned bytes.

    `cdist` is the reason this is affordable: the comparison runs in C++ across
    every core, so a two-hour lecture (~1,500 windows, 2.25M cells) costs a
    couple of megabytes and well under a second.
    """
    from rapidfuzz import fuzz, process

    texts = [w.text for w in windows]
    return process.cdist(
        texts, texts, scorer=fuzz.token_set_ratio, dtype=np.uint8, workers=-1
    )


def find_replays(transcript: Transcript, min_run: int = MIN_RUN) -> list[Replay]:
    """Stretches played more than once, most-replayed first."""
    windows = _windows(transcript)
    if len(windows) < min_run * 2:
        return []

    scores = _match_matrix(windows)
    total = len(windows)

    # Candidate pairs: similar, and far enough apart in time that this is not
    # simply the window overlap recognising itself.
    starts = np.array([w.start_s for w in windows])
    similar = scores >= SIMILARITY
    far_enough = np.abs(starts[:, None] - starts[None, :]) >= MIN_GAP_S
    matched = similar & far_enough

    runs: list[tuple[int, int, int]] = []  # (first_index, second_index, length)
    consumed: set[tuple[int, int]] = set()

    for i in range(total):
        for j in range(i + 1, total):
            if not matched[i, j] or (i, j) in consumed:
                continue
            # Walk the diagonal: i+k aligning with j+k is what makes this a
            # replayed *stretch* rather than a repeated sentence.
            length = 0
            while (
                i + length < total
                and j + length < total
                and matched[i + length, j + length]
            ):
                consumed.add((i + length, j + length))
                length += 1
            if length >= min_run:
                runs.append((i, j, length))

    return _to_replays(windows, runs, scores)


def _to_replays(
    windows: list[_Window], runs: list[tuple[int, int, int]], scores: np.ndarray
) -> list[Replay]:
    """Fold overlapping runs into one entry per replayed stretch.

    Watching something three times produces three pairwise runs over the same
    material; reporting them separately would claim three different struggles
    where there was one.
    """
    if not runs:
        return []

    grouped: list[dict] = []
    for first, second, length in sorted(runs, key=lambda r: r[0]):
        span = (first, first + length)
        for group in grouped:
            if first < group["span"][1] and group["span"][0] < span[1]:
                group["span"] = (min(group["span"][0], span[0]),
                                 max(group["span"][1], span[1]))
                group["at"].update({first, second})
                break
        else:
            grouped.append({"span": span, "at": {first, second}})

    replays = []
    for group in grouped:
        begin, end = group["span"]
        # Rebuild the text from the run's own windows, stepping by the window
        # size so the overlap is not repeated back to the reader.
        pieces = [windows[k].text for k in range(begin, min(end, len(windows)), 2)]
        times = _count_plays(windows, scores, begin)
        replays.append(Replay(text=" ".join(pieces), times=times))

    replays = [r for r in replays if r.count >= 2]
    replays.sort(key=lambda r: (-r.count, r.times[0] if r.times else 0.0))
    replays = _merge_same_material(replays)
    log.info("Found %d replayed stretch(es)", len(replays))
    return replays


def _merge_same_material(replays: list[Replay]) -> list[Replay]:
    """One entry per replayed stretch, however many alignments found it.

    Watching something three times lays down copies at several window offsets,
    and the span-overlap grouping treats non-adjacent copies as separate
    findings. They are the same material and the same struggle, so listing them
    twice would overstate what happened - and this section is read as a claim
    about which ideas did not land.
    """
    from rapidfuzz import fuzz

    kept: list[Replay] = []
    for replay in replays:
        if any(fuzz.token_set_ratio(replay.text, other.text) >= SIMILARITY for other in kept):
            continue
        kept.append(replay)
    return kept


ADJACENT = 2
"""Windows either side of a match that belong to the same playthrough. The step
is half a window, so an anchor matches its own neighbours."""


def _count_plays(windows: list[_Window], scores: np.ndarray, anchor: int) -> list[float]:
    """When each playthrough of this stretch began.

    The count is the entire claim this feature makes, so it is derived directly
    rather than inferred from how many alignments happened to be found. The
    first window of the stretch is used as an anchor and matched against the
    whole lecture; every place it recurs is a playthrough.

    An earlier version clustered the endpoints of overlapping diagonal runs by a
    fixed time gap. It could not tell *windows within one play* from *separate
    plays* - a stretch watched three times was reported as seven - and material
    played twice reported as four times is not a smaller error than missing it
    altogether.
    """
    matches = [k for k in range(len(windows)) if scores[anchor, k] >= SIMILARITY]
    if not matches:
        return [windows[anchor].start_s]

    plays: list[float] = []
    previous = None
    for k in matches:
        if previous is None or k - previous > ADJACENT:
            plays.append(windows[k].start_s)
        previous = k
    return plays
