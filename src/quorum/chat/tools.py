"""What the chat can actually do.

Two rules shape every tool here, and they are the same two the rest of the
project runs on.

**Reads are free, writes are two-phase.** A read tool runs the moment the model
asks for it. A write tool *cannot* perform anything on its first call - it
returns a `PendingWrite` describing exactly what it would do, and the effect
happens only when a human confirms and the tool is dispatched a second time with
`confirmed=True`. There is no argument the model can set to skip that, because
the model never gets to set it: `confirmed` is passed by the CLI after asking a
person, not carried in the tool request.

**Ambiguity refuses.** `close_commitment("the")` matched everything once, and
closing the wrong commitment is silent - nothing later reopens it. Every tool
that identifies a commitment by description abstains when two candidates score
close together, exactly as `quorum done` does.

Tool selection uses the router's structured output rather than provider-native
tool calling. Gemini and Groq expose different tool-calling shapes, and going
through `structured()` keeps one code path with the cache, quota accounting,
failover and tracing already attached to it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Callable

from pydantic import BaseModel, Field

from quorum.memory.store import MemoryHit
from quorum.workspace import Project, Workspace

log = logging.getLogger(__name__)

AMBIGUITY_MARGIN = 5.0
MIN_MATCH = 55.0
MAX_TRANSCRIPT_CHARS = 3500
"""Roughly 900 tokens - a readable stretch, and small enough that the answering
prompt still has room to produce an answer inside its own token allowance."""
VAGUE = {"the", "a", "an", "it", "that", "this", "and", "of", "to", "my", "our", "thing"}


class ToolRequest(BaseModel):
    """A flat argument bag, deliberately.

    Nested or free-form dict arguments are where structured output breaks
    unevenly across providers - Gemini enforces the schema server-side and Groq
    does not, so a shape one accepts silently becomes a parse retry on the
    other. Flat optional strings survive both.
    """

    tool: str = Field(description="Which tool to run.")
    query: str = Field(default="", description="What to search for.")
    meeting: str = Field(default="", description="Handle, id or title words.")
    what: str = Field(default="", description="Which commitment, by description.")
    when: str = Field(default="", description='A deadline, e.g. "next Friday" or 2026-09-01.')
    who: str = Field(default="", description="A person's name or email.")
    start: str = Field(default="", description='Transcript start time, "12:30".')
    end: str = Field(default="", description="Transcript end time.")
    drop: bool = Field(default=False, description="Abandon rather than complete.")


@dataclass
class PendingWrite:
    """A write that has been described but not performed."""

    tool: str
    request: ToolRequest
    preview: str
    """What the human reads before saying yes. If this under-describes the
    effect, the confirmation is not informed consent."""


@dataclass
class ToolResult:
    ok: bool
    text: str
    """What the model sees next, and usually what the user sees too."""

    hits: list[MemoryHit] = field(default_factory=list)
    pending: PendingWrite | None = None

    scope: str = ""
    """Human-readable label of the meeting this came from, for the banner."""

    scope_id: str = ""
    """The meeting id. Kept separate from `scope` because the label is prose -
    "Weekly sync (2026-08-03)" - and feeding that back in as the next turn's
    focus would re-resolve it by fuzzy title, throwing away the disambiguation
    the user just made between two meetings with the same name."""

    @property
    def needs_confirmation(self) -> bool:
        return self.pending is not None


@dataclass
class ToolContext:
    project: Project
    workspace: Workspace
    today: date = field(default_factory=date.today)
    memory: object | None = None
    scope_meeting: str = ""
    """A meeting the conversation is currently focused on. Tools that accept a
    `meeting` argument fall back to this, so you can say "and what about the
    complexity" without re-naming the lecture every turn."""

    federated: bool = False
    """Whether reads span every project. Writes never do - see `writable`."""

    def get_memory(self):
        if self.memory is None:
            from quorum.memory import ProjectMemory

            self.memory = ProjectMemory(self.project.memory_dir)
        return self.memory

    @property
    def writable(self) -> bool:
        """Whether an action tool may run.

        "Mark the spec as done" across five projects has no correct answer when
        two of them contain something that matches, and picking one silently
        closes work that is still outstanding - which nothing later reopens.
        """
        return not self.federated


# ---------------------------------------------------------------------------
# Identifying things
# ---------------------------------------------------------------------------


def _resolve_scope(ctx: ToolContext, request: ToolRequest) -> tuple[list[str], str, str]:
    """Turn a spoken meeting reference into ids to filter retrieval by.

    Returns (meeting_ids, label, error). An unresolvable reference is an error
    rather than a silent widening to the whole project: answering about a
    different lecture than the one you named is worse than saying you cannot
    find it.
    """
    from quorum.chat.naming import resolve_meeting

    wanted = request.meeting or ctx.scope_meeting
    if not wanted:
        return [], "", ""

    resolution = resolve_meeting(ctx.project, wanted)
    if resolution.ok:
        return [resolution.match.meeting_id], resolution.match.label, ""
    if resolution.ambiguous:
        options = ", ".join(ref.label for ref in resolution.candidates)
        return [], "", f"{wanted!r} matches several: {options}. Which one?"
    return [], "", f"No meeting matching {wanted!r} in this project."


def _find_commitment(ctx: ToolContext, what: str):
    """One open commitment, or an explanation of why not."""
    from rapidfuzz import fuzz

    candidates = ctx.project.ledger.open_commitments()
    if not candidates:
        return None, "Nothing is open on this project."

    words = [w for w in what.lower().split() if w not in VAGUE]
    if len(what.strip()) < 4 or not words:
        return None, f"{what!r} is too vague to identify a commitment."

    scored = sorted(
        ((fuzz.token_set_ratio(what.lower(), c.description.lower()), c) for c in candidates),
        key=lambda pair: -pair[0],
    )
    best_score, best = scored[0]
    if best_score < MIN_MATCH:
        listing = "; ".join(c.description for _, c in scored[:5])
        return None, f"Nothing matches {what!r}. Open: {listing}"

    if len(scored) > 1 and scored[1][0] >= best_score - AMBIGUITY_MARGIN:
        options = "; ".join(c.description for _, c in scored[:3])
        return None, f"{what!r} is ambiguous - it matches: {options}. Be more specific."

    return best, ""


# ---------------------------------------------------------------------------
# Read tools
# ---------------------------------------------------------------------------


def tool_search(ctx: ToolContext, request: ToolRequest, confirmed: bool = False) -> ToolResult:
    """Retrieve from notes, decisions and commitments."""
    meeting_ids, label, error = _resolve_scope(ctx, request)
    if error:
        return ToolResult(False, error)

    hits = ctx.get_memory().recall(
        request.query or request.what, k=6, meeting_ids=meeting_ids or None
    )
    scope_id = meeting_ids[0] if meeting_ids else ""
    if not hits:
        return ToolResult(
            True, "Nothing in the indexed material matches that.",
            scope=label, scope_id=scope_id,
        )
    summary = "\n".join(f"[{i}] {hit.text}" for i, hit in enumerate(hits, start=1))
    return ToolResult(True, summary, hits=hits, scope=label, scope_id=scope_id)


def tool_read_transcript(
    ctx: ToolContext, request: ToolRequest, confirmed: bool = False
) -> ToolResult:
    """The words actually spoken, filtered. For "what exactly did he say"."""
    from quorum.chat.naming import resolve_meeting
    from quorum.export import Style, parse_time, render

    wanted = request.meeting or ctx.scope_meeting
    if not wanted:
        return ToolResult(False, "Which meeting or lecture? Name one first.")

    resolution = resolve_meeting(ctx.project, wanted)
    if not resolution.ok:
        _, _, error = _resolve_scope(ctx, request)
        return ToolResult(False, error or f"No meeting matching {wanted!r}.")

    transcript = next(
        (t for t in ctx.project.transcripts()
         if t.meeting_id == resolution.match.meeting_id),
        None,
    )
    if transcript is None:
        return ToolResult(False, f"No stored transcript for {resolution.match.label}.")

    text = render(
        transcript,
        # Same roster-vs-reality trap: keyed on the declared speaker list, a
        # lecture rendered as "Remote participant (00:15): ..." on every line.
        style=Style.TIMESTAMPED if transcript.is_monologue else Style.SPEAKERS,
        speaker=request.who or None,
        start_s=parse_time(request.start or None),
        end_s=parse_time(request.end or None),
        search=request.query or None,
    )
    if not text.strip():
        return ToolResult(True, "That filter matched no lines.",
                          scope=resolution.match.label,
                          scope_id=resolution.match.meeting_id)

    # A two-hour seminar does not fit in a 6k tokens/minute budget, and the
    # model does not need it to - the filters are the point. 6,000 characters
    # was still too generous: an unfiltered read of a ten-minute lecture filled
    # the answering prompt so completely that the model hit its own output cap
    # mid-JSON, which the provider reports as a validation failure rather than
    # anything that points at length.
    if len(text) > MAX_TRANSCRIPT_CHARS:
        text = (
            text[:MAX_TRANSCRIPT_CHARS]
            + "\n[... truncated - narrow it with start/end times or a search term]"
        )
    return ToolResult(True, text, scope=resolution.match.label,
                      scope_id=resolution.match.meeting_id)


def tool_list_meetings(
    ctx: ToolContext, request: ToolRequest, confirmed: bool = False
) -> ToolResult:
    from quorum.chat.naming import list_meetings

    refs = list_meetings(ctx.project)
    if not refs:
        return ToolResult(True, "Nothing recorded on this project yet.")
    lines = [
        f"@{ref.handle or '(unnamed)'} - {ref.title or 'untitled'} "
        f"({ref.meeting_date}, {ref.kind}, {ref.utterances} lines)"
        for ref in refs
    ]
    return ToolResult(True, "\n".join(lines))


def tool_list_commitments(
    ctx: ToolContext, request: ToolRequest, confirmed: bool = False
) -> ToolResult:
    items = ctx.project.ledger.open_commitments()
    if request.who:
        needle = request.who.lower()
        items = [
            c for c in items
            if needle in (c.assignee.display_name or "").lower()
            or needle in (c.assignee.email or "").lower()
        ]
    if not items:
        return ToolResult(True, "Nothing open matches that.")

    lines = []
    for item in sorted(items, key=lambda c: c.deadline.resolved or date.max):
        due = item.deadline.resolved
        when = due.isoformat() if due else "no date"
        if due and due < ctx.today:
            when += f" ({(ctx.today - due).days}d late)"
        owner = item.assignee.display_name or "unassigned"
        lines.append(f"- {item.description} [{owner}, {when}]")
    return ToolResult(True, "\n".join(lines))


# ---------------------------------------------------------------------------
# Write tools - two-phase, always
# ---------------------------------------------------------------------------


def tool_close_commitment(
    ctx: ToolContext, request: ToolRequest, confirmed: bool = False
) -> ToolResult:
    from quorum.models import CommitmentStatus

    commitment, error = _find_commitment(ctx, request.what or request.query)
    if commitment is None:
        return ToolResult(False, error)

    verb = "Drop" if request.drop else "Close"
    if not confirmed:
        return ToolResult(
            True, f"{verb} {commitment.description!r}?",
            pending=PendingWrite(
                "close_commitment", request,
                f"{verb} this commitment: {commitment.description}",
            ),
        )

    commitment.status = (
        CommitmentStatus.DROPPED if request.drop else CommitmentStatus.VERIFIED_DONE
    )
    commitment.resolution_note = (
        f"marked {'dropped' if request.drop else 'done'} in chat on {ctx.today}"
    )
    ctx.project.save_ledger()
    return ToolResult(
        True,
        f"{'Dropped' if request.drop else 'Closed'}: {commitment.description}. "
        f"{len(ctx.project.ledger.open_commitments())} still open.",
    )


def tool_set_deadline(
    ctx: ToolContext, request: ToolRequest, confirmed: bool = False
) -> ToolResult:
    """Give a commitment a date, or move the one it has.

    This is the missing half of the deadline loop: the planner flags undated
    commitments and cannot chase them, and `calendar` lists them and cannot
    schedule them. Neither could ever ask you for the date.
    """
    from quorum.agents.dates import resolve_deadline
    from quorum.models import DeadlineResolution

    commitment, error = _find_commitment(ctx, request.what or request.query)
    if commitment is None:
        return ToolResult(False, error)

    if not request.when.strip():
        return ToolResult(False, "No date given. When is it due?")

    resolved = resolve_deadline(request.when, ctx.today)
    if resolved.value is None:
        return ToolResult(
            False,
            f"I could not turn {request.when!r} into a date. Try 'next Friday' "
            "or an exact date like 2026-09-01.",
        )

    was = commitment.deadline.resolved
    moving = f" (was {was.isoformat()})" if was else ""
    if not confirmed:
        return ToolResult(
            True,
            f"Set {commitment.description!r} due {resolved.value.isoformat()}{moving}?",
            pending=PendingWrite(
                "set_deadline", request,
                f"Set the deadline for {commitment.description} to "
                f"{resolved.value.isoformat()}{moving}",
            ),
        )

    commitment.record_deadline_change(
        resolved.value, on=ctx.today, source="chat", note=request.when
    )
    commitment.deadline.resolved = resolved.value
    commitment.deadline.raw_text = request.when
    commitment.deadline.method = (
        resolved.method if resolved.method is not DeadlineResolution.NONE
        else DeadlineResolution.EXPLICIT
    )
    commitment.deadline.confidence = resolved.confidence
    ctx.project.save_ledger()
    return ToolResult(
        True,
        f"{commitment.description} is now due {resolved.value.isoformat()}{moving}. "
        "Run the calendar sync to put it in your calendar.",
    )


def tool_sync_calendar(
    ctx: ToolContext, request: ToolRequest, confirmed: bool = False
) -> ToolResult:
    from quorum.config import get_settings
    from quorum.execution import ApprovalGate, CalendarConfig, CalendarSync
    from quorum.execution.calendar import CalendarTransport
    from quorum.integrations import GoogleAuthError, credentials_status, get_calendar_service
    from quorum.tracking import ActionType, PlannedAction

    settings = get_settings()
    config = CalendarConfig(
        calendar_id=settings.calendar_id,
        reminder_days=settings.reminder_days(),
        reminder_hour=settings.reminder_hour,
    )

    google = credentials_status()
    service = None
    if google.ready:
        try:
            service = get_calendar_service()
        except GoogleAuthError as exc:
            return ToolResult(False, str(exc))

    sync = CalendarSync(service, config)
    plan = sync.plan(ctx.project.ledger, ctx.today)

    if plan.is_empty:
        undated = (
            f" {len(plan.undated)} commitment(s) have no deadline and cannot be "
            "scheduled - give them one and I will add them."
            if plan.undated else ""
        )
        return ToolResult(True, f"The calendar already matches the ledger.{undated}")

    if not confirmed:
        return ToolResult(
            True, f"Calendar sync: {plan.summary_line()}.",
            pending=PendingWrite("sync_calendar", request, plan.render()),
        )

    if service is None:
        return ToolResult(False, f"Cannot write to the calendar: {google.message}")

    # Same gate as everywhere else. The chat confirmation is what authorises the
    # approval; it does not replace it.
    gate = ApprovalGate(require_approval=settings.require_approval)
    pending = gate.propose(
        PlannedAction(f"calendar:{ctx.project.meta.id}", ActionType.SCHEDULE,
                      plan.summary_line()),
        f"Calendar sync: {plan.summary_line()}", body=plan.render(),
    )
    transport = CalendarTransport(sync, plan)
    gate.execute(pending.id, gate.approve(pending.id), transport)
    result = transport.result
    return ToolResult(
        True,
        f"Calendar updated: {result.created} added, {result.updated} updated, "
        f"{result.deleted} removed."
        + (f" {len(result.failed)} failed." if result.failed else ""),
    )


def tool_draft_email(
    ctx: ToolContext, request: ToolRequest, confirmed: bool = False
) -> ToolResult:
    """Write an email and save it. Never sends - Gmail OAuth is not wired."""
    from quorum.config import RUNS_DIR, free_path
    from quorum.llm.providers import ModelTier
    from quorum.llm.router import get_router

    recipient = request.who.strip()
    if not recipient:
        return ToolResult(False, "Who is it to?")

    email = ctx.project.meta.members.get(recipient, "")
    if not email:
        email = next(
            (addr for name, addr in ctx.project.meta.members.items()
             if recipient.lower() in name.lower()),
            recipient if "@" in recipient else "",
        )
    if not email:
        known = ", ".join(ctx.project.meta.members) or "nobody on this project"
        return ToolResult(False, f"No email for {recipient!r}. I know: {known}.")

    subject_hint = request.what or request.query or "the outstanding work"

    hits = ctx.get_memory().recall(subject_hint, k=4)
    context = "\n".join(f"- {hit.text}" for hit in hits) or "(no specific context found)"
    prompt = (
        f"Write a short, direct work email to {recipient} about: {subject_hint}\n\n"
        f"Relevant context from the project's meetings:\n{context}\n\n"
        "Two or three sentences. No preamble, no sign-off flourishes. Return the "
        "subject line first, then a blank line, then the body."
    )
    try:
        response = get_router().complete(
            prompt, tier=ModelTier.BALANCED, max_tokens=600, purpose="chat_draft_email"
        )
    except Exception as exc:  # noqa: BLE001
        return ToolResult(False, f"Could not draft that ({type(exc).__name__}).")

    body = response.text.strip()
    if not confirmed:
        return ToolResult(
            True, f"Draft to {email}:\n\n{body}",
            pending=PendingWrite(
                "draft_email", request, f"Save this draft for {email}:\n\n{body}"
            ),
        )

    drafts = RUNS_DIR / "drafts" / ctx.project.meta.id
    drafts.mkdir(parents=True, exist_ok=True)
    # Date plus recipient alone collides: drafting a second mail to the same
    # person on the same day silently replaced the first, which is a thing a
    # person does routinely and would never think to check for.
    stem = f"{ctx.today.isoformat()}_{email.replace('@', '_at_')}"
    path = free_path(drafts, stem, ".txt")
    path.write_text(f"To: {email}\n\n{body}", encoding="utf-8")
    return ToolResult(
        True,
        f"Saved to {path}. Nothing was sent - automatic sending needs Gmail "
        "OAuth, which is not wired up.",
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass
class Tool:
    name: str
    run: Callable[[ToolContext, ToolRequest, bool], ToolResult]
    description: str
    writes: bool = False


TOOLS: dict[str, Tool] = {
    tool.name: tool
    for tool in [
        Tool("search", tool_search,
             "Find what was said or decided. args: query, meeting(optional)"),
        Tool("read_transcript", tool_read_transcript,
             "The exact words spoken. args: meeting, who/start/end/query to filter"),
        Tool("list_meetings", tool_list_meetings,
             "Every recorded meeting and lecture, with its @handle. No args."),
        Tool("list_commitments", tool_list_commitments,
             "Open commitments and their deadlines. args: who(optional)"),
        Tool("close_commitment", tool_close_commitment,
             "Mark work done or abandoned. args: what, drop(optional)", writes=True),
        Tool("set_deadline", tool_set_deadline,
             "Give a commitment a due date. args: what, when", writes=True),
        Tool("sync_calendar", tool_sync_calendar,
             "Put deadlines in the user's calendar. No args.", writes=True),
        Tool("draft_email", tool_draft_email,
             "Write an email and save it as a draft. args: who, what", writes=True),
    ]
}


def run_tool(ctx: ToolContext, request: ToolRequest, confirmed: bool = False) -> ToolResult:
    """Dispatch. An unknown tool is reported back, not raised.

    The model occasionally invents a plausible-sounding tool name; telling it
    what actually exists lets the next turn correct itself, where an exception
    would end the conversation.
    """
    tool = TOOLS.get(request.tool)
    if tool is None:
        return ToolResult(
            False, f"No tool {request.tool!r}. Available: {', '.join(TOOLS)}."
        )
    if tool.writes and not ctx.writable:
        # Refused rather than applied to whichever project happened to match
        # first. Closing the wrong commitment is silent and irreversible.
        return ToolResult(
            False,
            f"{tool.name} changes things, and this chat is searching every "
            "project at once. Re-open it with --project <name> to make changes.",
        )
    if confirmed and not tool.writes:
        log.debug("Confirmation passed to the read-only tool %s; ignored", tool.name)
    try:
        return tool.run(ctx, request, confirmed)
    except Exception as exc:  # noqa: BLE001 - a broken tool must not end the chat
        log.warning("Tool %s failed: %s", request.tool, exc)
        return ToolResult(False, f"{request.tool} failed: {type(exc).__name__}: {exc}")


def describe_tools() -> str:
    """The catalogue the router sees when choosing."""
    return "\n".join(
        f"- {tool.name}: {tool.description}"
        + ("  [asks for confirmation before doing anything]" if tool.writes else "")
        for tool in TOOLS.values()
    )
