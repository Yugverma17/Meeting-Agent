"""Topic segmentation.

Segmentation is a budget mechanism before it is a quality mechanism. Groq's free
tier allows 6,000 tokens per minute, so a 40-minute transcript (~30k tokens) can
never be sent in one call. Every downstream stage operates on segments, and
`max_tokens` is a hard ceiling the splitter is not allowed to exceed.

The algorithm is TextTiling with embeddings instead of word overlap: compare the
block of utterances before each gap with the block after, then cut where that
similarity dips into a valley deep enough to look like a real topic change.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from quorum.agents.embedding import Embedder, get_embedder
from quorum.llm.ratelimit import estimate_tokens
from quorum.models import Segment, Transcript

log = logging.getLogger(__name__)


@dataclass
class SegmenterConfig:
    window: int = 3
    """Utterances compared either side of a candidate gap. Larger is smoother
    but blurs short exchanges, and meeting turns are often one line long."""

    min_utterances: int = 4
    """Minimum segment length. Prevents a single interjection ("sure", "yep")
    from being treated as its own topic."""

    max_tokens: int = 1_800
    """Hard ceiling per segment, sized to leave headroom inside a 6,000 TPM
    budget for the prompt scaffolding and the model's reply."""

    depth_factor: float = 0.4
    """Cut threshold in standard deviations above mean valley depth. Lower cuts
    more aggressively."""

    min_absolute_depth: float = 0.02
    """Floor on how shallow a valley may be and still count as a topic change.
    Guards the degenerate case where every gap is equally similar, which makes
    the relative cutoff zero and would otherwise cut everywhere."""


class Segmenter:
    def __init__(
        self, config: SegmenterConfig | None = None, embedder: Embedder | None = None
    ) -> None:
        self.config = config or SegmenterConfig()
        self._embedder = embedder

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = get_embedder()
        return self._embedder

    # -- public ------------------------------------------------------------

    def segment(self, transcript: Transcript) -> list[Segment]:
        utterances = transcript.utterances
        if not utterances:
            return []

        cfg = self.config
        if len(utterances) < cfg.min_utterances * 2:
            return self._finalise(transcript, [(0, len(utterances) - 1)])

        texts = [u.text for u in utterances]
        embeddings = self.embedder.embed(texts)
        similarities = self._gap_similarities(embeddings)
        boundaries = self._pick_boundaries(similarities)

        spans = self._to_spans(boundaries, len(utterances))
        spans = self._enforce_token_ceiling(transcript, spans, similarities)
        return self._finalise(transcript, spans)

    # -- internals ---------------------------------------------------------

    def _gap_similarities(self, embeddings: np.ndarray) -> np.ndarray:
        """Cosine similarity across each gap. Index g compares [g-w:g] with [g:g+w].

        Gaps too close to either end are given similarity 1.0 (no boundary), so
        the algorithm never proposes a cut it lacks the context to justify.
        """
        n = len(embeddings)
        w = self.config.window
        sims = np.ones(n, dtype=np.float32)

        for gap in range(w, n - w + 1):
            left = embeddings[gap - w : gap].mean(axis=0)
            right = embeddings[gap : gap + w].mean(axis=0)
            denom = float(np.linalg.norm(left) * np.linalg.norm(right))
            sims[gap] = float(np.dot(left, right) / denom) if denom else 1.0
        return sims

    def _pick_boundaries(self, sims: np.ndarray) -> list[int]:
        """Local minima whose valleys are deep relative to the rest of the meeting.

        Depth is measured against the nearest peak on each side rather than a
        global threshold, because absolute similarity varies a lot between a
        rambling discussion and a crisp status round.
        """
        cfg = self.config
        n = len(sims)

        # Restrict to gaps that could produce two legal segments. Without this,
        # a boundary at index 1 yields a one-utterance opening segment.
        lo, hi = cfg.min_utterances, n - cfg.min_utterances
        candidates = [
            g
            for g in range(max(1, lo), min(n - 1, hi) + 1)
            if sims[g] <= sims[g - 1] and sims[g] <= sims[g + 1]
        ]
        if not candidates:
            return []

        depths = {g: self._valley_depth(sims, g) for g in candidates}
        values = np.array(list(depths.values()), dtype=np.float32)

        # A relative cutoff alone collapses on degenerate input: if every gap is
        # equally similar the standard deviation is zero, the cutoff becomes
        # zero, and every flat gap qualifies as a boundary. An absolute floor
        # means "no variation" is read as "no topic change", which is correct.
        cutoff = max(
            float(values.mean() + cfg.depth_factor * values.std()), cfg.min_absolute_depth
        )

        # Deepest first, so when spacing forces a choice the strongest wins.
        ranked = sorted(depths.items(), key=lambda kv: kv[1], reverse=True)
        chosen: list[int] = []
        for gap, depth in ranked:
            if depth < cutoff:
                continue
            if all(abs(gap - c) >= cfg.min_utterances for c in chosen):
                chosen.append(gap)
        return sorted(chosen)

    @staticmethod
    def _valley_depth(sims: np.ndarray, gap: int) -> float:
        left = gap
        while left > 0 and sims[left - 1] >= sims[left]:
            left -= 1
        right = gap
        while right < len(sims) - 1 and sims[right + 1] >= sims[right]:
            right += 1
        return float((sims[left] - sims[gap]) + (sims[right] - sims[gap]))

    @staticmethod
    def _to_spans(boundaries: list[int], n: int) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        start = 0
        for boundary in boundaries:
            if boundary > start:
                spans.append((start, boundary - 1))
                start = boundary
        spans.append((start, n - 1))
        return [s for s in spans if s[0] <= s[1]]

    def _enforce_token_ceiling(
        self, transcript: Transcript, spans: list[tuple[int, int]], sims: np.ndarray
    ) -> list[tuple[int, int]]:
        """Split any segment over budget, recursively, at its weakest internal gap.

        This runs after topic detection because a quota ceiling is not a topic
        boundary - we would rather cut a coherent discussion in half than emit a
        segment that cannot physically be sent.
        """
        result: list[tuple[int, int]] = []
        queue = list(spans)

        while queue:
            start, end = queue.pop(0)
            if self._span_tokens(transcript, start, end) <= self.config.max_tokens:
                result.append((start, end))
                continue
            if end - start < 1:
                # A single utterance over budget: nothing left to split.
                log.warning(
                    "Utterance %d alone exceeds max_tokens; emitting oversized segment", start
                )
                result.append((start, end))
                continue

            interior = range(start + 1, end + 1)
            split = min(interior, key=lambda g: sims[g] if g < len(sims) else 1.0)
            queue.insert(0, (split, end))
            queue.insert(0, (start, split - 1))

        return sorted(result)

    @staticmethod
    def _span_tokens(transcript: Transcript, start: int, end: int) -> int:
        return estimate_tokens(
            " ".join(u.text for u in transcript.utterances[start : end + 1])
        )

    @staticmethod
    def _finalise(transcript: Transcript, spans: list[tuple[int, int]]) -> list[Segment]:
        return [
            Segment(meeting_id=transcript.meeting_id, start_index=start, end_index=end)
            for start, end in spans
        ]
