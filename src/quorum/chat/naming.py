"""Referring to a meeting the way you would say it out loud.

`mtg_9f2c1a4b7e30` is how the system identifies a meeting and is not how anyone
talks about one. Three ways to name the same thing, tried hardest-first:

1. **A handle you chose** - `@kickoff`. Exact, stable, and the only one that
   survives a retitle. Set with `quorum name`.
2. **The meeting id.** Exact, and what error messages print, so pasting one back
   always works.
3. **Words from the title.** Fuzzy, convenient, and ambiguous by nature - four
   meetings called "Weekly sync" are indistinguishable this way.

The ambiguity rule is the only interesting decision here: when two meetings
score close together, this returns *both* rather than picking. Picking wrongly
is silent - you would get a confident answer about the wrong week and no
indication anything had gone astray. Asking costs one line.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date

from quorum.models import Transcript
from quorum.workspace import Project

log = logging.getLogger(__name__)

_SLUG = re.compile(r"[^a-z0-9]+")
_STOPWORDS = {
    "the", "a", "an", "of", "and", "on", "in", "for", "to", "with",
    "meeting", "lecture", "session", "call", "video", "part",
}

AMBIGUITY_MARGIN = 6.0
"""How close a runner-up has to score before the match is called ambiguous."""

MIN_SCORE = 55.0
"""Below this, a title match is a coincidence rather than a match."""


@dataclass
class MeetingRef:
    """One meeting, addressable."""

    meeting_id: str
    title: str
    meeting_date: date | None = None
    handle: str = ""
    utterances: int = 0
    kind: str = "meeting"
    """"meeting" or "lecture" - a lecture has one speaker and no ledger."""

    @property
    def label(self) -> str:
        name = self.handle or self.title or self.meeting_id
        when = f" ({self.meeting_date.isoformat()})" if self.meeting_date else ""
        return f"{name}{when}"


@dataclass
class Resolution:
    """What `resolve_meeting` found. Deliberately not just a MeetingRef.

    Ambiguity has to be representable, or the caller has no way to tell a
    confident match from a coin flip between two identically-titled standups.
    """

    match: MeetingRef | None = None
    candidates: list[MeetingRef] = field(default_factory=list)
    how: str = ""
    """"handle", "id", "title", "ambiguous" or "none" - surfaced so the chat can
    say *why* it thinks you meant this one."""

    @property
    def ok(self) -> bool:
        return self.match is not None

    @property
    def ambiguous(self) -> bool:
        """Genuinely torn between candidates - not merely unmatched.

        Keyed on `how` rather than on the candidate count, because the
        no-match case fills `candidates` with every meeting so the caller can
        list them. Counting them made "no meeting called that" indistinguishable
        from "which of these two did you mean", and nonsense came back as an
        ambiguity between every recording in the project.
        """
        return self.how == "ambiguous"


def auto_handle(title: str, taken: set[str]) -> str:
    """A handle derived from the title, so meetings are addressable by default.

    Requiring `quorum name` before a lecture can be referred to would mean the
    common case - watch one thing, ask about it - needs a second command first.
    """
    words = [w for w in _SLUG.sub(" ", title.lower()).split() if w and w not in _STOPWORDS]
    stem = "-".join(words[:3]) or "meeting"

    if stem not in taken:
        return stem
    for suffix in range(2, 100):
        candidate = f"{stem}-{suffix}"
        if candidate not in taken:
            return candidate
    return stem  # pragma: no cover - a hundred identically-named meetings


def list_meetings(project: Project) -> list[MeetingRef]:
    """Everything addressable in a project, newest first.

    Built from stored transcripts rather than meeting records: a lecture has a
    transcript and notes but no `MeetingRecord`, and it still has to be
    nameable.
    """
    by_handle = {meeting_id: handle for handle, meeting_id in project.meta.handles.items()}
    refs = []
    for transcript in project.transcripts():
        refs.append(MeetingRef(
            meeting_id=transcript.meeting_id,
            title=transcript.title,
            meeting_date=transcript.meeting_date,
            handle=by_handle.get(transcript.meeting_id, ""),
            utterances=len(transcript.utterances),
            # Who actually spoke, not the declared roster - live capture always
            # adds a placeholder participant, so counting the roster labelled
            # every solo lecture a meeting.
            kind="lecture" if transcript.is_monologue else "meeting",
        ))
    refs.sort(key=lambda r: (r.meeting_date or date.min), reverse=True)
    return refs


def set_handle(project: Project, meeting_id: str, handle: str) -> str:
    """Name a meeting. Returns the normalised handle actually stored."""
    cleaned = _SLUG.sub("-", handle.strip().lower().lstrip("@")).strip("-")
    if not cleaned:
        raise ValueError("A handle needs at least one letter or digit")

    # A handle points at one meeting. Reassigning it moves the name rather than
    # leaving two meetings both answering to it, which would make `@standup`
    # mean whichever one the lookup happened to see first.
    for existing, target in list(project.meta.handles.items()):
        if existing == cleaned and target != meeting_id:
            log.info("Handle %r moved from %s to %s", cleaned, target, meeting_id)
            del project.meta.handles[existing]
        elif target == meeting_id and existing != cleaned:
            del project.meta.handles[existing]

    project.meta.handles[cleaned] = meeting_id
    return cleaned


def register_meeting(project: Project, transcript: Transcript) -> str:
    """Give a freshly captured meeting a handle, unless it already has one."""
    for handle, meeting_id in project.meta.handles.items():
        if meeting_id == transcript.meeting_id:
            return handle
    handle = auto_handle(transcript.title or "meeting", set(project.meta.handles))
    project.meta.handles[handle] = transcript.meeting_id
    return handle


def resolve_meeting(project: Project, query: str) -> Resolution:
    """Find the meeting someone meant. Ambiguity is reported, never guessed."""
    from rapidfuzz import fuzz

    wanted = (query or "").strip().lstrip("@")
    refs = list_meetings(project)
    if not wanted or not refs:
        return Resolution(candidates=refs, how="none")

    lowered = wanted.lower()
    by_id = {ref.meeting_id: ref for ref in refs}

    handle_target = project.meta.handles.get(_SLUG.sub("-", lowered).strip("-"))
    if handle_target and handle_target in by_id:
        return Resolution(match=by_id[handle_target], how="handle")

    if wanted in by_id:
        return Resolution(match=by_id[wanted], how="id")

    scored = sorted(
        ((fuzz.token_set_ratio(lowered, (ref.title or "").lower()), ref) for ref in refs),
        key=lambda pair: -pair[0],
    )
    best_score, best = scored[0]
    if best_score < MIN_SCORE:
        return Resolution(candidates=refs, how="none")

    close = [ref for score, ref in scored if score >= best_score - AMBIGUITY_MARGIN]
    if len(close) > 1:
        # Two meetings called "Weekly sync" are genuinely indistinguishable by
        # title. Answering about the wrong week is silent, so it asks instead.
        return Resolution(candidates=close, how="ambiguous")

    return Resolution(match=best, how="title")


MENTION = re.compile(r"@([a-z0-9][a-z0-9-]*)", re.IGNORECASE)


def extract_mention(text: str) -> str:
    """Pull an explicit `@handle` out of a sentence, if there is one.

    Only the explicit form. Inferring "the kickoff meeting" from prose is the
    model's job in the routing step - doing it with a regex here would silently
    disagree with the model about what the user meant.
    """
    found = MENTION.search(text or "")
    return found.group(1).lower() if found else ""
