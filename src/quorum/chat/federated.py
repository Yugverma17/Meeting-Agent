"""Searching every project at once.

Memory is stored per project, which is right for indexing - each project's
history lives in its own file and re-recording one meeting cannot corrupt
another. It is wrong for *asking*, because the interesting questions stop being
about one recording as soon as you have several:

    where have I seen sliding windows before
    what did we decide about auth, on any project

A folder structure cannot answer those. Retrieval can, and the indexes already
exist - they simply were never queried together.

**One embedder, shared.** This is the part that would quietly not work
otherwise. Scores from two indexes are only comparable if the same embedding
model produced them, and `get_embedder()` degrades to a hashing fallback when
the ONNX model cannot be loaded. A project indexed with fastembed and one
indexed lexically would score on different scales, and merging them by score
would rank almost arbitrarily - silently, and worse the more projects you have.
Building one embedder here and handing it to every store removes the question,
and loads ~100 MB of model once rather than once per project, which matters on
a machine with 7.6 GB.

**Writes stay single-project.** Only reads federate. "Mark the spec as done"
across five projects has no correct answer when two of them contain something
that matches, and picking one silently closes work that is still outstanding.
"""

from __future__ import annotations

import logging

from quorum.memory.store import MemoryHit, MemoryKind, ProjectMemory
from quorum.workspace import Project, Workspace

log = logging.getLogger(__name__)


class FederatedMemory:
    """Reads across projects. Presents the same surface as `ProjectMemory`.

    Deliberately duck-typed rather than a subclass: the tools call `recall` and
    nothing else, so matching that one method keeps every call site unchanged
    and makes single-project the same code path as many.
    """

    def __init__(self, projects: list[Project], embedder=None) -> None:
        self.projects = projects
        if embedder is None:
            from quorum.agents.embedding import get_embedder

            embedder = get_embedder()
        self.embedder = embedder
        self._stores: dict[str, ProjectMemory] = {}

    def store_for(self, project: Project) -> ProjectMemory:
        if project.meta.id not in self._stores:
            self._stores[project.meta.id] = ProjectMemory(
                project.memory_dir, embedder=self.embedder
            )
        return self._stores[project.meta.id]

    def recall(
        self,
        query: str,
        k: int = 5,
        kind: MemoryKind | None = None,
        min_score: float = 0.0,
        meeting_ids: list[str] | None = None,
    ) -> list[MemoryHit]:
        """Top-k across every project, merged by score.

        Each store is asked for a full k rather than a share of one. A project
        holding all of the best answers should be allowed to supply all of them,
        and an even split would force in weaker hits from elsewhere purely for
        being elsewhere.
        """
        merged: list[MemoryHit] = []
        for project in self.projects:
            try:
                hits = self.store_for(project).recall(
                    query, k=k, kind=kind, min_score=min_score, meeting_ids=meeting_ids
                )
            except Exception as exc:  # noqa: BLE001 - one broken index must not end the search
                log.warning("Could not search %s (%s)", project.meta.id, exc)
                continue
            for hit in hits:
                hit.project_id = project.meta.id
                merged.append(hit)

        merged.sort(key=lambda hit: -hit.score)
        return merged[:k]

    def count(self) -> int:
        return sum(self.store_for(p).count() for p in self.projects)


def all_projects(workspace: Workspace | None = None) -> list[Project]:
    """Every project that has something recorded in it.

    Keyed on *recordings*, not on the index. Filtering by the index looked
    equivalent and was not: a project whose indexing failed - which the code
    treats as non-fatal, on purpose - still has transcripts, and it would have
    become invisible to `@handle` resolution as well as to search. A project
    with nothing recorded is skipped, because opening a store for it costs time
    and returns nothing.
    """
    workspace = workspace or Workspace()
    found = []
    for meta in workspace.list():
        project = workspace.get(meta.id)
        if project is None:
            continue
        has_recordings = (
            project.transcripts_dir.exists() and any(project.transcripts_dir.glob("*.json"))
        )
        if has_recordings or project.memory_dir.exists():
            found.append(project)
    return found


def resolve_across(projects: list[Project], query: str):
    """Find a meeting by handle or title in any project.

    Returns `(project, resolution)`. Ambiguity across projects is reported the
    same way as ambiguity within one - two projects each containing a "Weekly
    sync" is exactly the case where guessing produces a confident answer about
    the wrong team.
    """
    from quorum.chat.naming import Resolution, resolve_meeting

    exact = []
    for project in projects:
        resolution = resolve_meeting(project, query)
        if resolution.ok:
            exact.append((project, resolution))

    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        candidates = [ref for _, res in exact for ref in [res.match]]
        return None, Resolution(candidates=candidates, how="ambiguous")

    # Nothing matched exactly. Surface within-project ambiguity if a single
    # project reported it, so the user gets the useful "which of these two"
    # rather than a bare "not found".
    for project in projects:
        resolution = resolve_meeting(project, query)
        if resolution.ambiguous:
            return project, resolution
    return None, Resolution(how="none")
