"""Semantic memory over a project's meeting history.

Retrieval here is not decoration - three things in the pipeline were doing a
worse job without it, and each degrades in a specific way as a project grows:

**1. Matching a status update to the commitment it refers to.** "I sent that
over on Tuesday" has to find *"the ingestion API spec"* among forty open items.
Fuzzy string matching handles a restatement, and fails on a paraphrase - "the
spec doc", "that API thing", "the ingestion write-up" all score poorly against
the original wording while meaning exactly it.

**2. Choosing contradiction candidates.** The detector previously sent *every*
prior decision in the prompt. That works for six decisions and breaks at sixty:
cost grows linearly with project age, and the signal gets buried. Retrieving the
handful of semantically related decisions bounds both.

**3. Giving the extractor project context.** A model reading meeting seven in
isolation does not know that "the ledger" is a component here rather than an
accounting term. A short brief drawn from prior decisions fixes a class of
misreadings that no amount of prompt tuning does.

Storage is LanceDB - embedded, file-backed, no server, which suits a machine
that cannot spare RAM for one. Embeddings are local via fastembed. If either is
unavailable the store degrades to an in-process index rather than failing: a
missing model should cost you retrieval quality, not your meeting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from enum import Enum
from pathlib import Path

import numpy as np

from quorum.agents.embedding import Embedder, get_embedder
from quorum.models import MeetingRecord, Transcript

log = logging.getLogger(__name__)

TABLE = "memory"


class MemoryKind(str, Enum):
    COMMITMENT = "commitment"
    DECISION = "decision"
    UTTERANCE = "utterance"
    RISK = "risk"
    NOTE = "note"
    """A point or concept from a lecture. Indexed so questions can be asked
    across everything watched, not just the most recent talk."""


@dataclass
class MemoryHit:
    kind: MemoryKind
    ref_id: str
    """Id of the source object, so a hit can be resolved back to a live
    Commitment or Decision in the ledger rather than just quoted."""

    text: str
    meeting_id: str
    meeting_date: str
    score: float

    def __str__(self) -> str:
        return f"[{self.kind.value} {self.meeting_date}] {self.text}"


class ProjectMemory:
    def __init__(self, path: Path, embedder: Embedder | None = None) -> None:
        self.path = Path(path)
        self._embedder = embedder
        self._table = None
        self._fallback: list[dict] | None = None
        """Used when LanceDB cannot be opened. Same behaviour, no persistence."""

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = get_embedder()
        return self._embedder

    # -- storage -----------------------------------------------------------

    def _connect(self):
        if self._table is not None or self._fallback is not None:
            return
        try:
            import lancedb

            self.path.mkdir(parents=True, exist_ok=True)
            db = lancedb.connect(str(self.path))
            self._db = db
            # Open directly rather than checking a listing first. The listing
            # API has moved between releases, and when the check quietly failed
            # on an existing table the code fell through to create_table and
            # errored with "Table 'memory' already exists" - degrading a working
            # store to the in-process fallback and losing persistence.
            try:
                self._table = db.open_table(TABLE)
            except Exception:  # noqa: BLE001 - absent table is the normal first run
                self._table = None
        except Exception as exc:  # noqa: BLE001 - any failure degrades, never raises
            log.warning("Vector store unavailable (%s); using in-process memory", exc)
            self._fallback = []

    def _append(self, rows: list[dict]) -> None:
        if not rows:
            return
        self._connect()

        if self._fallback is not None:
            self._fallback.extend(rows)
            return

        try:
            if self._table is None:
                import lancedb

                db = getattr(self, "_db", None) or lancedb.connect(str(self.path))
                try:
                    self._table = db.create_table(TABLE, data=rows)
                except Exception:  # noqa: BLE001 - lost a race, or it existed after all
                    self._table = db.open_table(TABLE)
                    self._table.add(rows)
            else:
                self._table.add(rows)
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not write to vector store (%s); falling back", exc)
            # Extend rather than replace: earlier writes in this session are
            # still valid and dropping them would silently shrink the index.
            self._fallback = (self._fallback or []) + list(rows)

    # -- indexing ----------------------------------------------------------

    def index_meeting(
        self, record: MeetingRecord, transcript: Transcript | None = None
    ) -> int:
        """Store one meeting's salient content. Returns rows written.

        Only commitments, decisions and risks are indexed by default. Indexing
        every utterance would bury those signals under small talk and bloat the
        index with lines nobody will ever need to retrieve.
        """
        texts: list[str] = []
        rows: list[dict] = []
        meeting_date = record.meeting_date.isoformat()

        def stage(kind: MemoryKind, ref_id: str, text: str) -> None:
            cleaned = (text or "").strip()
            if not cleaned:
                return
            texts.append(cleaned)
            rows.append({
                "kind": kind.value, "ref_id": ref_id, "text": cleaned,
                "meeting_id": record.meeting_id, "meeting_date": meeting_date,
            })

        for commitment in record.commitments:
            owner = commitment.assignee.display_name or "someone"
            stage(MemoryKind.COMMITMENT, commitment.id, f"{owner}: {commitment.description}")
        for decision in record.decisions:
            stage(MemoryKind.DECISION, decision.id, decision.statement)
        for risk in record.risks:
            stage(MemoryKind.RISK, risk.id, risk.description)

        if not rows:
            return 0

        # Re-processing a meeting must replace its entries, not add a second
        # copy. Without this a re-recorded meeting appears twice in every
        # recall, and duplicates crowd out other results from the top-k.
        self._forget_meeting(record.meeting_id)

        vectors = self.embedder.embed(texts)
        for row, vector in zip(rows, vectors):
            row["vector"] = vector.tolist()

        self._append(rows)
        return len(rows)

    def index_notes(self, source_id: str, title: str, when: date, notes) -> int:
        """Index lecture notes so they can be searched later.

        Concepts are stored as "term: explanation" rather than the term alone -
        a bare term embeds poorly and a question like "how does backprop work"
        needs the explanation's words to match against.
        """
        texts: list[str] = []
        rows: list[dict] = []
        stamp = when.isoformat()

        def stage(ref: str, text: str) -> None:
            cleaned = (text or "").strip()
            if not cleaned:
                return
            texts.append(cleaned)
            rows.append({
                "kind": MemoryKind.NOTE.value, "ref_id": ref, "text": cleaned,
                "meeting_id": source_id, "meeting_date": stamp,
            })

        for index, concept in enumerate(getattr(notes, "concepts", [])):
            stage(f"{source_id}:concept:{index}",
                  f"{concept.term}: {concept.plain_explanation} {concept.why_it_matters}".strip())
        for index, point in enumerate(getattr(notes, "key_points", [])):
            stage(f"{source_id}:point:{index}", f"{point.point} {point.detail}".strip())
        for index, example in enumerate(getattr(notes, "examples", [])):
            stage(f"{source_id}:example:{index}",
                  f"{example.description} {example.what_it_demonstrates}".strip())
        if getattr(notes, "summary", ""):
            stage(f"{source_id}:summary", f"{title}. {notes.summary}")

        if not rows:
            return 0

        self._forget_meeting(source_id)
        vectors = self.embedder.embed(texts)
        for row, vector in zip(rows, vectors):
            row["vector"] = vector.tolist()
        self._append(rows)
        return len(rows)

    def _forget_meeting(self, meeting_id: str) -> None:
        self._connect()
        if self._fallback is not None:
            self._fallback = [r for r in self._fallback if r["meeting_id"] != meeting_id]
            return
        if self._table is None:
            return
        try:
            escaped = meeting_id.replace("'", "''")
            self._table.delete(f"meeting_id = '{escaped}'")
        except Exception as exc:  # noqa: BLE001 - a stale duplicate beats a crash
            log.warning("Could not clear previous entries for %s: %s", meeting_id, exc)

    # -- retrieval ---------------------------------------------------------

    def recall(
        self, query: str, k: int = 5, kind: MemoryKind | None = None, min_score: float = 0.0
    ) -> list[MemoryHit]:
        """Semantically nearest stored items."""
        if not query.strip():
            return []
        self._connect()
        vector = self.embedder.embed([query])[0]

        rows = self._search(vector, k * 4 if kind else k)
        hits = [
            MemoryHit(
                kind=MemoryKind(row["kind"]), ref_id=row["ref_id"], text=row["text"],
                meeting_id=row["meeting_id"], meeting_date=row["meeting_date"],
                score=row["_score"],
            )
            for row in rows
        ]
        if kind is not None:
            hits = [hit for hit in hits if hit.kind is kind]
        return [hit for hit in hits if hit.score >= min_score][:k]

    def _search(self, vector: np.ndarray, k: int) -> list[dict]:
        if self._fallback is not None:
            return self._brute_force(self._fallback, vector, k)

        if self._table is None:
            return []
        try:
            results = self._table.search(vector.tolist()).limit(k).to_list()
        except Exception as exc:  # noqa: BLE001
            log.warning("Vector search failed (%s)", exc)
            return []

        for row in results:
            # LanceDB returns L2 distance; convert so larger is better and the
            # threshold means the same thing in both code paths.
            distance = float(row.get("_distance", 0.0))
            row["_score"] = 1.0 - distance / 2.0
        return results

    @staticmethod
    def _brute_force(rows: list[dict], vector: np.ndarray, k: int) -> list[dict]:
        if not rows:
            return []
        matrix = np.array([row["vector"] for row in rows], dtype=np.float32)
        scores = matrix @ vector  # rows are already L2-normalised
        order = np.argsort(-scores)[:k]
        return [{**rows[i], "_score": float(scores[i])} for i in order]

    # -- the three uses ----------------------------------------------------

    def match_commitment(self, text: str, min_score: float = 0.55) -> str | None:
        """Which open commitment does this status update refer to?

        Returns a commitment id, or None when nothing is close enough. Abstaining
        matters: applying "that's done" to the wrong commitment closes work that
        is still outstanding, and nothing later reopens it.
        """
        hits = self.recall(text, k=3, kind=MemoryKind.COMMITMENT, min_score=min_score)
        return hits[0].ref_id if hits else None

    def contradiction_candidates(self, statement: str, k: int = 5) -> list[MemoryHit]:
        """Prior decisions worth checking against a new one.

        Replaces sending every decision ever made, which grows linearly with the
        project and buries the relevant one.
        """
        return self.recall(statement, k=k, kind=MemoryKind.DECISION)

    def brief(self, limit: int = 8) -> str:
        """A short project context block for the extractor's prompt.

        Gives a model reading meeting seven enough to know that "the ledger" is
        a component in this project, not an accounting term.
        """
        self._connect()
        rows = self._all_rows()
        if not rows:
            return ""

        decisions = [r for r in rows if r["kind"] == MemoryKind.DECISION.value]
        commitments = [r for r in rows if r["kind"] == MemoryKind.COMMITMENT.value]
        chosen = (decisions + commitments)[-limit:]
        if not chosen:
            return ""

        lines = "\n".join(f"- {row['text']}" for row in chosen)
        return (
            "Context from earlier meetings on this project (background only - do "
            f"not extract commitments from it):\n{lines}"
        )

    def _all_rows(self) -> list[dict]:
        """Every stored row.

        LanceDB's full-scan method has moved between releases, so each candidate
        is tried in turn. An earlier version called only `to_pandas()` inside a
        bare `except: return []`, which silently reported an empty index on a
        store that was working perfectly - `recall()` returned results while
        `count()` said zero. Failures are logged now rather than swallowed.
        """
        if self._fallback is not None:
            return list(self._fallback)
        if self._table is None:
            return []

        errors = []
        for name, extract in (
            ("to_arrow", lambda t: t.to_arrow().to_pylist()),
            ("to_pandas", lambda t: t.to_pandas().to_dict("records")),
            ("to_list", lambda t: t.to_list()),
            ("search", lambda t: t.search().limit(100_000).to_list()),
        ):
            try:
                return extract(self._table)
            except Exception as exc:  # noqa: BLE001 - probing for the right API
                errors.append(f"{name}: {type(exc).__name__}")
        log.warning("Could not scan the vector store (%s)", "; ".join(errors))
        return []

    def count(self) -> int:
        self._connect()
        return len(self._all_rows())
