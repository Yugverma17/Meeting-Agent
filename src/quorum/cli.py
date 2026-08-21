"""Command line interface.

    python -m quorum.cli doctor          # is my environment sane?
    python -m quorum.cli models          # what can I call, and what are the limits?
    python -m quorum.cli models --probe  # do my API keys actually work? (uses quota)
    python -m quorum.cli quota           # how much of today's budget is left?
    python -m quorum.cli cache --stats
"""

from __future__ import annotations

import datetime as _dt
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from quorum import __version__
from quorum.config import (
    CACHE_DIR,
    DATA_DIR,
    LLM_CACHE_DIR,
    RUNS_DIR,
    ensure_dirs,
    free_path,
    get_settings,
    setup_logging,
)
from quorum.llm.cache import LLMCache
from quorum.llm.providers import WHISPER_MODEL, registry
from quorum.llm.ratelimit import QuotaTracker
from quorum.llm.router import Router

def _make_console_unicode_safe() -> None:
    """Stop a stray character from killing a command on a Windows console.

    A Windows terminal defaults to cp1252, which cannot encode the non-breaking
    hyphens, smart quotes and en dashes that models produce constantly. Printing
    one raised `UnicodeEncodeError` from deep inside rich and lost the whole
    answer - including, on `learn`, notes that had already cost quota to produce.

    UTF-8 first, since modern Windows Terminal handles it; `errors="replace"`
    behind it so a console that does not is left with a substituted character
    rather than a traceback.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError, ValueError):  # pragma: no cover - redirected
            pass


_make_console_unicode_safe()

app = typer.Typer(add_completion=False, help="Quorum - meeting commitment agent.")
console = Console()


def show(text: str, style: str = "") -> None:
    """Print text exactly as it is.

    Rich reads square brackets as style tags and silently deletes anything it
    does not recognise. On a data-structures lecture that is catastrophic and
    invisible: `last[ch] = i` reached the terminal as `last = i`, and
    `dp[i][j] = dp[i-1][j]` as `dp = dp`. The answer was correct; the display
    destroyed it, and nothing anywhere reported an error.

    Use this for anything the project did not write itself - model output,
    transcript lines, notes, commitment descriptions, email bodies.
    `highlight=False` also stops rich recolouring numbers inside code.
    """
    console.print(text, markup=False, highlight=False, style=style or None)


def safe(text: str) -> str:
    """User or model text, escaped so it can sit inside a styled string.

    For the cases `show` cannot serve - a table cell, or content wrapped in
    `[dim]...[/dim]`. Table cells are rendered with markup too, so a commitment
    described as "fix dp[i] handling" loses the subscript in `quorum status`
    exactly as it does anywhere else.
    """
    return escape(text or "")


@app.command()
def doctor(probe: bool = typer.Option(False, help="Make one real call per provider.")) -> None:
    """Check that the environment is ready to run."""
    setup_logging()
    ensure_dirs()
    settings = get_settings()

    table = Table(title="Quorum environment", show_header=True, header_style="bold")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail", overflow="fold")

    ok = "[green]OK[/green]"
    warn = "[yellow]WARN[/yellow]"
    bad = "[red]MISSING[/red]"

    py = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    table.add_row("Python", ok if sys.version_info >= (3, 11) else bad, py)
    table.add_row("Version", ok, f"quorum {__version__}")

    providers = settings.configured_providers()
    if providers:
        table.add_row("LLM providers", ok, ", ".join(providers))
    else:
        table.add_row(
            "LLM providers",
            bad,
            "No keys. Copy .env.example to .env and set GEMINI_API_KEY and/or GROQ_API_KEY.",
        )

    table.add_row(
        "Gemini key", ok if settings.gemini_api_key else warn,
        "set" if settings.gemini_api_key else "https://aistudio.google.com/apikey",
    )
    table.add_row(
        "Groq key", ok if settings.groq_api_key else warn,
        "set" if settings.groq_api_key else "https://console.groq.com/keys",
    )
    from quorum.integrations import credentials_status
    from quorum.llm.tracing import tracing_enabled

    table.add_row(
        "LangSmith", ok if tracing_enabled() else warn,
        f"tracing to project {settings.langsmith_project!r}" if tracing_enabled()
        else "no LANGSMITH_API_KEY - runs are untraced",
    )
    table.add_row(
        "GitHub token", ok if settings.github_token else warn,
        "set" if settings.github_token else "optional until reality-verification phase",
    )

    google = credentials_status()
    table.add_row("Google Calendar", ok if google.ready else warn, google.message)

    entries, total_bytes = LLMCache(LLM_CACHE_DIR).size()
    table.add_row("LLM cache", ok, f"{entries} entries, {total_bytes / 1e6:.1f} MB")
    table.add_row(
        "Approval gate",
        ok if settings.require_approval else warn,
        "enabled" if settings.require_approval else "DISABLED - outbound actions unguarded",
    )
    table.add_row("Data dir", ok, str(DATA_DIR))

    console.print(table)

    if probe:
        _probe_providers()
    elif providers:
        console.print("\n[dim]Run with --probe to verify the keys actually work.[/dim]")


def _probe_providers() -> None:
    """One minimal call per configured provider. Costs a few requests."""
    settings = get_settings()
    router = Router(max_wait_s=0.0)
    console.print("\n[bold]Probing providers[/bold] [dim](consumes a little quota)[/dim]")

    for spec in registry.all():
        if spec.provider not in settings.configured_providers():
            console.print(f"  [dim]skip[/dim]  {spec.key} [dim](no key)[/dim]")
            continue
        try:
            caller = router._call_gemini if spec.provider == "gemini" else router._call_groq
            # Budget must be generous: reasoning models spend output tokens on
            # internal thinking first, and a tight cap returns an empty string
            # that looks exactly like a broken key.
            text, _, out_tok = caller(
                spec, "Reply with the single word: ok", None, 0.0, 256, False, None, False
            )
            reply = text.strip()[:40]
            if reply:
                console.print(f"  [green]ok[/green]    {spec.key} -> {reply!r}")
            else:
                console.print(
                    f"  [yellow]empty[/yellow] {spec.key}: no text but {out_tok} tokens spent "
                    "(reasoning consumed the budget)"
                )
        except Exception as exc:  # noqa: BLE001 - report whatever the SDK raised
            console.print(f"  [red]fail[/red]  {spec.key}: {type(exc).__name__}: {exc}"[:200])


@app.command()
def models(probe: bool = typer.Option(False, help="Verify each model against your account.")) -> None:
    """List the model registry and its free-tier limits."""
    settings = get_settings()
    configured = settings.configured_providers()

    table = Table(title="Model registry (free-tier limits, August 2026)", header_style="bold")
    table.add_column("Model")
    table.add_column("Provider")
    table.add_column("Tier")
    table.add_column("RPM", justify="right")
    table.add_column("RPD", justify="right")
    table.add_column("TPM", justify="right")
    table.add_column("Key?", justify="center")

    for spec in registry.all():
        has_key = "[green]y[/green]" if spec.provider in configured else "[red]n[/red]"
        table.add_row(
            spec.name, spec.provider, spec.tier.value,
            str(spec.rpm or "-"), str(spec.rpd or "-"),
            f"{spec.tpm:,}" if spec.tpm else "-", has_key,
        )
    console.print(table)
    console.print(
        f"\n[dim]Speech-to-text: {WHISPER_MODEL.name} - {WHISPER_MODEL.notes}[/dim]"
    )
    console.print(
        "[yellow]Note:[/yellow] providers change free-tier limits without notice. "
        "If long runs start failing, re-check these numbers against your account."
    )
    if probe:
        _probe_providers()


@app.command()
def quota(reset: bool = typer.Option(False, help="Clear locally recorded usage.")) -> None:
    """Show how much of the local quota budget has been consumed."""
    tracker = QuotaTracker(CACHE_DIR / "quota_state.json")
    if reset:
        tracker.reset()
        console.print("[yellow]Local quota state cleared.[/yellow] "
                      "This does not reset the provider's own counters.")
        return

    snapshot = tracker.snapshot()
    if not snapshot:
        console.print("[dim]No usage recorded yet.[/dim]")
        return

    table = Table(title="Quota usage", header_style="bold")
    table.add_column("Model")
    table.add_column("Req/min", justify="right")
    table.add_column("Tok/min", justify="right")
    table.add_column("Req/day", justify="right")
    table.add_column("Tok/day", justify="right")

    for key, usage in snapshot.items():
        spec = registry.by_name(key.split(":", 1)[-1])
        rpd = f"/{spec.rpd}" if spec and spec.rpd else ""
        table.add_row(
            key,
            str(usage["requests_last_minute"]),
            f"{usage['tokens_last_minute']:,}",
            f"{usage['requests_last_day']}{rpd}",
            f"{usage['tokens_last_day']:,}",
        )
    console.print(table)


@app.command()
def cache(
    stats: bool = typer.Option(True, help="Show cache statistics."),
    clear: bool = typer.Option(False, help="Delete every cached response."),
) -> None:
    """Inspect or clear the LLM response cache."""
    store = LLMCache(LLM_CACHE_DIR)
    if clear:
        removed = store.clear()
        console.print(f"[yellow]Removed {removed} cached responses.[/yellow] "
                      "Re-running evals will now consume quota again.")
        return
    if stats:
        entries, total_bytes = store.size()
        console.print(f"Cached responses: [bold]{entries}[/bold]")
        console.print(f"On disk: [bold]{total_bytes / 1e6:.2f} MB[/bold]")
        console.print(f"Location: [dim]{LLM_CACHE_DIR}[/dim]")


@app.command()
def evaluate(
    seeds: str = typer.Option("0,1,2", help="Comma-separated project seeds."),
    weeks: int = typer.Option(6, help="Meetings per project."),
    out: str = typer.Option("", help="Write the full report to this JSON path."),
    no_evidence: bool = typer.Option(False, help="Disable the reality-verification layer."),
) -> None:
    """Run the synthetic benchmark end to end and print the scores."""
    import json

    from quorum.agents.embedding import LexicalEmbedder
    from quorum.agents.segmenter import Segmenter
    from quorum.eval.harness import EvaluationHarness

    setup_logging("WARNING")
    ensure_dirs()
    seed_list = [int(s) for s in seeds.split(",") if s.strip()]

    harness = EvaluationHarness(
        segmenter=Segmenter(embedder=LexicalEmbedder()), use_evidence=not no_evidence
    )
    with console.status(f"Evaluating {len(seed_list)} project(s)..."):
        results, summary = harness.run(seed_list, weeks=weeks)

    extraction = Table(title="Extraction (single meeting)", header_style="bold")
    extraction.add_column("Metric")
    extraction.add_column("Value", justify="right")
    commitments = summary["extraction"]["commitments"]
    for label, value in [
        ("Precision", commitments["precision"]),
        ("Recall", commitments["recall"]),
        ("F1", commitments["f1"]),
        ("Assignee accuracy", summary["extraction"]["assignee_accuracy"]),
        ("Deadline accuracy", summary["extraction"]["deadline_accuracy"]),
        ("Strength accuracy", summary["extraction"]["strength_accuracy"]),
        ("Musing promotion rate", summary["extraction"]["musing_promotion_rate"]),
        ("Hallucination rate", summary["extraction"]["hallucination_rate"]),
    ]:
        extraction.add_row(label, f"{value:.3f}")
    console.print(extraction)

    tracking = Table(
        title="Tracking (across meetings - no public benchmark covers these)",
        header_style="bold",
    )
    tracking.add_column("Metric")
    tracking.add_column("Value", justify="right")
    tracking.add_column("n", justify="right")
    counts = summary["tracking"]["counts"]
    for label, key, count_key in [
        ("Dropped-commitment recall", "dropped_recall", "dropped_total"),
        ("False-nag rate (lower better)", "false_nag_rate", "nag_targets_total"),
        ("Silent-delivery recall", "silent_delivery_recall", "silent_deliveries_total"),
        ("Contradiction recall", "contradiction_recall", "contradictions_total"),
        ("Contradiction precision", "contradiction_precision", "contradictions_total"),
        ("Blocked propagation", "blocked_propagation_recall", "blocked_total"),
    ]:
        tracking.add_row(label, f"{summary['tracking'][key]:.3f}", str(counts[count_key]))
    console.print(tracking)

    cost = summary["cost"]
    console.print(
        f"\n[bold]{summary['projects']}[/bold] projects, [bold]{summary['meetings']}[/bold] "
        f"meetings, [bold]{cost['llm_calls']}[/bold] LLM calls, "
        f"[bold]{cost['total_tokens']:,}[/bold] tokens "
        f"([bold]{cost['tokens_per_meeting']:,}[/bold]/meeting), "
        f"[bold]{cost['wall_seconds']}s[/bold], cost [bold green]$0.00[/bold green]"
    )

    if out:
        payload = {"summary": summary, "projects": [r.as_dict() for r in results]}
        Path(out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        console.print(f"[dim]Full report written to {out}[/dim]")


@app.command()
def guard(model: bool = typer.Option(False, help="Also use the prompt-guard classifier.")) -> None:
    """Score the speech-injection defence against its attack suite."""
    from quorum.security import SpeechInjectionGuard

    setup_logging("WARNING")
    result = SpeechInjectionGuard(use_model=model).evaluate()

    table = Table(title="Speech-injection guard", header_style="bold")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Attacks", str(result["attacks"]))
    table.add_row("Blocked", str(result["blocked"]))
    table.add_row("Block rate", f"{result['block_rate']:.3f}")
    table.add_row("Benign lines", str(result["benign"]))
    table.add_row("False positives", str(result["false_positives"]))
    table.add_row("False-positive rate", f"{result['false_positive_rate']:.3f}")
    console.print(table)

    if result["missed"]:
        console.print(f"[red]Missed:[/red] {', '.join(result['missed'])}")
    if result["tripped"]:
        console.print(f"[yellow]False positives:[/yellow] {'; '.join(result['tripped'])}")
    console.print(
        "\n[dim]Note: the primary defence is structural - there is no code path from "
        "extracted text to an action. This layer makes attempts visible.[/dim]"
    )


def _load_project(name: str):
    """Resolve a project by name, or exit with the list of what exists."""
    from quorum.workspace import Workspace

    workspace = Workspace()
    if not name:
        projects = workspace.list()
        if len(projects) == 1:
            name = projects[0].id
        else:
            console.print("[red]Specify --project.[/red]")
            if projects:
                console.print("Available: " + ", ".join(p.id for p in projects))
            else:
                console.print('Create one: [bold]quorum project --create "My Project"[/bold]')
            raise typer.Exit(1)

    project = workspace.get(name)
    if project is None:
        # Naming a project that does not exist yet is the overwhelmingly common
        # way to arrive here - usually seconds before a lecture starts. Listing
        # what exists is only half an answer; the other half is the one command
        # that fixes it.
        console.print(f"[red]No project {name!r}.[/red]")
        existing = workspace.list()
        if existing:
            console.print("Existing: " + ", ".join(p.id for p in existing))
        console.print(f'Create it: [bold]quorum project --create "{name}"[/bold]')
        raise typer.Exit(1)
    return workspace, project


@app.command()
def project(
    create: str = typer.Option("", help="Create a project with this name."),
    show: str = typer.Option("", help="Show one project in detail."),
    description: str = typer.Option("", help="What the project is."),
    repo: str = typer.Option("", help='GitHub "owner/name" for delivery verification.'),
    members: str = typer.Option("", help='"Priya:priya@x.com,Sam:sam@x.com"'),
) -> None:
    """Create or list projects. A project gives meetings shared context."""
    from quorum.workspace import Workspace

    setup_logging("WARNING")
    ensure_dirs()
    workspace = Workspace()

    if create:
        people = {}
        for entry in (m for m in members.split(",") if m.strip()):
            name, _, email = entry.partition(":")
            people[name.strip()] = email.strip()
        try:
            made = workspace.create(create, description, repo or None, people)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1)
        console.print(f"[green]Created[/green] [bold]{made.meta.id}[/bold]")
        console.print(f"  Record into it: [bold]quorum record --project {made.meta.id}[/bold]")
        return

    if show:
        _, found = _load_project(show)
        ledger = found.ledger
        console.print(f"[bold]{found.meta.name}[/bold] ({found.meta.id})")
        if found.meta.description:
            console.print(f"  {found.meta.description}")
        console.print(f"  meetings: {found.meta.meeting_count}   "
                      f"last: {found.meta.last_meeting_on or '-'}")
        console.print(f"  open commitments: {len(ledger.open_commitments())}")
        if found.meta.repo:
            console.print(f"  repo: {found.meta.repo}")
        if found.meta.members:
            console.print("  members: " + ", ".join(
                f"{n} <{e}>" for n, e in found.meta.members.items()))
        return

    projects = workspace.list()
    if not projects:
        console.print("[dim]No projects yet.[/dim]")
        console.print('Create one: [bold]quorum project --create "Ingestion revamp"[/bold]')
        return

    table = Table(title="Projects", header_style="bold")
    table.add_column("id")
    table.add_column("name")
    table.add_column("meetings", justify="right")
    table.add_column("last meeting")
    for meta in projects:
        table.add_row(meta.id, meta.name, str(meta.meeting_count), meta.last_meeting_on or "-")
    console.print(table)


@app.command()
def status(project_name: str = typer.Option("", "--project", help="Which project.")) -> None:
    """Everything still open: who owes what, and what is late."""
    from datetime import date as _date

    setup_logging("WARNING")
    _, found = _load_project(project_name)
    ledger = found.ledger
    today_date = _date.today()
    open_items = ledger.open_commitments()

    if not open_items:
        console.print(f"[green]Nothing open on {found.meta.name}.[/green]")
        return

    table = Table(title=f"{found.meta.name} - open commitments", header_style="bold")
    table.add_column("Owner")
    table.add_column("Commitment", overflow="fold")
    table.add_column("Due")
    table.add_column("Status")

    for item in sorted(open_items, key=lambda c: (c.deadline.resolved or _date.max)):
        due = item.deadline.resolved
        if due is None:
            state = "[dim]no date[/dim]"
        elif due < today_date:
            state = f"[red]{(today_date - due).days}d late[/red]"
        elif due == today_date:
            state = "[yellow]due today[/yellow]"
        else:
            state = f"in {(due - today_date).days}d"
        table.add_row(
            safe(item.assignee.display_name) or "[dim]unassigned[/dim]",
            safe(item.description),
            due.isoformat() if due else "-",
            state,
        )
    console.print(table)
    console.print(f"\n[dim]{len(open_items)} open. "
                  f"Run [bold]quorum today[/bold] to see what to chase.[/dim]")


@app.command()
def today(
    project_name: str = typer.Option("", "--project", help="Which project."),
    check_github: bool = typer.Option(True, help="Verify delivery against GitHub."),
) -> None:
    """The daily sweep: what to chase, close, escalate or flag - with drafts."""
    from datetime import date as _date

    from quorum.execution import ApprovalGate, build_digests
    from quorum.tracking import ActionType, Planner
    from quorum.verify import GitHubEvidenceProvider
    from quorum.verify.github import GitHubConfig

    setup_logging("WARNING")
    settings = get_settings()
    workspace, found = _load_project(project_name)
    ledger = found.ledger
    now = _date.today()

    evidence = None
    if check_github and settings.github_token and found.meta.repo:
        evidence = GitHubEvidenceProvider(
            settings.github_token, GitHubConfig(repo=found.meta.repo)
        )
    elif check_github and not settings.github_token:
        console.print("[dim]No GITHUB_TOKEN set - skipping delivery verification.[/dim]")

    with console.status("Planning..."):
        plan = Planner().plan(ledger, now, evidence)
    found.save_ledger()

    if not plan.actions:
        console.print(f"[green]Nothing to do on {found.meta.name}.[/green]")
        return

    colour = {
        ActionType.NUDGE: "yellow", ActionType.ESCALATE: "red",
        ActionType.MARK_DONE: "green", ActionType.MARK_DROPPED: "magenta",
        ActionType.PROPAGATE_SLIP: "cyan", ActionType.FLAG_CONFLICT: "red",
    }
    table = Table(title=f"{found.meta.name} - {now}", header_style="bold")
    table.add_column("Action")
    table.add_column("Commitment", overflow="fold")
    table.add_column("Why", overflow="fold")
    for action in plan.actions:
        if action.action is ActionType.WAIT:
            continue
        item = ledger.by_id(action.commitment_id)
        style = colour.get(action.action, "dim")
        table.add_row(
            f"[{style}]{action.action.value}[/{style}]",
            safe(item.description if item else action.commitment_id),
            safe(action.reason),
        )
    console.print(table)

    digests = build_digests(plan.actions, ledger, now)
    if not digests:
        console.print("\n[dim]No emails needed today.[/dim]")
        return

    gate = ApprovalGate()
    drafts_dir = RUNS_DIR / "drafts" / found.meta.id
    drafts_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"\n[bold]{len(digests)} email(s) drafted[/bold] "
                  "[yellow]- awaiting your approval, nothing sent[/yellow]\n")
    for digest in digests:
        body = digest.render(ledger, now)
        gate.propose(
            plan.actions[0], digest.subject, body=body, recipient=digest.recipient_email
        )
        # Running `today` twice in a morning used to overwrite the first sweep's
        # drafts, including any you had already edited by hand.
        path = free_path(
            drafts_dir, f"{now.isoformat()}_{digest.recipient_email.replace('@', '_at_')}", ".txt"
        )
        path.write_text(f"To: {digest.recipient_email}\nSubject: {digest.subject}\n\n{body}",
                        encoding="utf-8")
        console.print(f"[bold]To:[/bold] {safe(digest.recipient_email)}")
        console.print(f"[bold]Subject:[/bold] {safe(digest.subject)}")
        # The body quotes the transcript verbatim, so it can contain anything.
        show(body)
        console.print(f"[dim]saved to {path}[/dim]\n")

    console.print(
        "[yellow]Review the drafts, then send them yourself.[/yellow] "
        "Automatic sending needs Gmail OAuth, which is not wired up."
    )
    console.print(
        f"[dim]Deadlines can also live in your calendar: "
        f"[bold]quorum calendar --project {found.meta.id}[/bold][/dim]"
    )


@app.command()
def drafts(
    project_name: str = typer.Option("", "--project", help="Which project."),
    apply_changes: bool = typer.Option(
        False, "--apply", help="Put them in your Gmail drafts folder."
    ),
    include_done: bool = typer.Option(False, help="Also draft for closed commitments."),
) -> None:
    """Write the emails the meeting said you would send.

    Most commitments are work. A few are messages - "I'll email Priya the spec by
    Friday" - and those are the only ones where the deliverable is something the
    agent can actually produce for you.

    Runs as a dry run by default. `--apply` puts them in Gmail's Drafts folder,
    behind the approval gate. Nothing is ever sent.
    """
    from quorum.config import RUNS_DIR as _RUNS
    from quorum.execution import ApprovalGate, DraftWriter, GmailDrafts, find_communications
    from quorum.execution.mail import GmailDraftTransport
    from quorum.integrations import GoogleAuthError, credentials_status, get_gmail_service
    from quorum.tracking import ActionType, PlannedAction

    setup_logging("WARNING")
    settings = get_settings()
    _, found = _load_project(project_name)

    pool = found.ledger.commitments if include_done else found.ledger.open_commitments()
    promised = find_communications(pool)
    if not promised:
        console.print(f"[green]No emails were promised on {found.meta.name}.[/green]")
        console.print("[dim]Only firm commitments that name a sending verb count - "
                      '"I\'ll email Priya the spec", not "I\'ll finish the migration".[/dim]')
        return

    console.print(f"[bold]{len(promised)} email(s) promised[/bold] on {found.meta.name}\n")

    writer = DraftWriter()
    written = []
    with console.status("Writing..."):
        for commitment in promised:
            draft = writer.write(commitment, found)
            if draft is not None:
                written.append(draft)

    if not written:
        console.print("[yellow]Could not draft any of them.[/yellow]")
        raise typer.Exit(1)

    unaddressed = [d for d in written if not d.addressed]
    for draft in written:
        console.print(f"[bold]To:[/bold] {safe(draft.to_email) or '[red]no address[/red]'}"
                      + (f" [dim]({safe(draft.to_name)})[/dim]" if draft.to_name else ""))
        console.print(f"[bold]Subject:[/bold] {safe(draft.subject)}")
        show(draft.body)
        if draft.quote:
            console.print(f"[dim]because you said: {safe(draft.quote)}[/dim]")
        console.print()

    # Always keep a copy on disk. Gmail may be unavailable, unauthorised, or
    # simply not what the user wants, and re-drafting costs quota.
    folder = _RUNS / "drafts" / found.meta.id
    folder.mkdir(parents=True, exist_ok=True)
    for draft in written:
        target = free_path(folder, f"{_dt.date.today().isoformat()}_{draft.commitment_id}", ".txt")
        target.write_text(draft.render(), encoding="utf-8")
    console.print(f"[dim]Saved {len(written)} draft(s) to {folder}[/dim]")

    if unaddressed:
        console.print(f"\n[yellow]{len(unaddressed)} has no address[/yellow] - "
                      "the person is not on this project's roster. Add them with "
                      f"[bold]quorum project --create[/bold] members, or fill the "
                      "address in by hand.")

    if not apply_changes:
        console.print("\n[yellow]Dry run - nothing added to Gmail.[/yellow] "
                      "Re-run with [bold]--apply[/bold] to put these in your drafts.")
        return

    google = credentials_status()
    if not google.ready:
        console.print(f"\n[red]Cannot reach Gmail: {google.message}[/red]")
        console.print("Run [bold]quorum auth[/bold] first.")
        raise typer.Exit(1)

    try:
        service = get_gmail_service()
    except GoogleAuthError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)

    sendable = [d for d in written if d.addressed]
    if not sendable:
        console.print("[yellow]Nothing to add - none of them have an address.[/yellow]")
        return

    gate = ApprovalGate(require_approval=settings.require_approval)
    action = PlannedAction(
        commitment_id=f"drafts:{found.meta.id}",
        action=ActionType.SCHEDULE,
        reason=f"{len(sendable)} Gmail draft(s)",
        priority=1,
    )
    pending = gate.propose(
        action, f"Create {len(sendable)} Gmail draft(s)",
        body="\n\n---\n\n".join(d.render() for d in sendable),
    )

    console.print(f"\n[bold]About to create {len(sendable)} draft(s)[/bold] in your "
                  "Gmail. [dim]They are not sent - you send them yourself.[/dim]")
    if not typer.confirm("Create them?", default=False):
        gate.reject(pending.id, "declined at the prompt")
        console.print("[yellow]Nothing created.[/yellow]")
        return

    transport = GmailDraftTransport(GmailDrafts(service), sendable)
    gate.execute(pending.id, gate.approve(pending.id), transport)
    result = transport.result

    console.print(f"[green]{result.created} draft(s) in Gmail.[/green] "
                  "Open Gmail, read them, send the ones you want.")
    if result.skipped:
        console.print(f"[dim]{result.skipped} skipped for having no address.[/dim]")
    for failure in result.failed:
        console.print(f"  [red]failed:[/red] {failure}")


@app.command()
def ui(
    port: int = typer.Option(8501, help="Port to serve on."),
    open_browser: bool = typer.Option(True, help="Open a browser window."),
) -> None:
    """Open the interface in your browser.

    Runs on this laptop rather than on a server: recording your system audio
    needs direct hardware access, which nothing remote can have. The browser is
    only the face - the recording, the models and your data stay here.
    """
    import subprocess

    setup_logging("WARNING")
    ensure_dirs()

    try:
        import streamlit  # noqa: F401
    except ImportError:
        console.print("[red]Streamlit is not installed.[/red]")
        console.print("  [bold]pip install streamlit[/bold]")
        raise typer.Exit(1)

    script = Path(__file__).parent / "ui" / "app.py"
    if not script.exists():  # pragma: no cover - packaging accident
        console.print(f"[red]Interface not found at {script}[/red]")
        raise typer.Exit(1)

    console.print(f"[bold]Quorum[/bold] on [cyan]http://localhost:{port}[/cyan]")
    console.print("[dim]Everything runs on this machine. Ctrl+C here to stop.[/dim]")

    command = [
        sys.executable, "-m", "streamlit", "run", str(script),
        "--server.port", str(port),
        "--server.headless", "true" if not open_browser else "false",
        # Telemetry off: this page displays meeting transcripts, and a tool that
        # phones home about a page showing your colleagues' words is not one to
        # leave on by default.
        "--browser.gatherUsageStats", "false",
        "--server.fileWatcherType", "none",
    ]
    try:
        subprocess.run(command, check=False)
    except KeyboardInterrupt:
        console.print("\n[dim]Stopped.[/dim]")


@app.command()
def week(
    project_name: str = typer.Option("", "--project", help="Which project."),
    days: int = typer.Option(7, help="How far back to look."),
    out: str = typer.Option("", help="Write the report to this markdown file."),
) -> None:
    """What changed between meetings.

    Every meeting tool summarises a meeting. This summarises the gap - and the
    gap is where the interesting things happen: a decision reversed across two
    meetings, a commitment that quietly evaporated, work delivered without
    anyone saying so. None of it is visible in a single transcript.

    Costs nothing: every line is a query over the ledger, not a model call.
    """
    from datetime import date as _date

    from quorum.tracking import build_report

    setup_logging("WARNING")
    _, found = _load_project(project_name)

    report = build_report(found.ledger, found.meta.name, until=_date.today(), days=days)

    console.print()
    show(report.as_markdown())

    if report.is_quiet:
        console.print("[dim]Nothing moved. That is a real answer, not an empty one - "
                      "it means no commitment slipped, reversed or lapsed.[/dim]")

    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(report.as_markdown(), encoding="utf-8")
        console.print(f"[green]Written to {out}[/green]")
    elif not report.is_quiet:
        console.print(f"[dim]Save it: quorum week --project {found.meta.id} "
                      f"--out runs/week.md[/dim]")


@app.command()
def triage(
    project_name: str = typer.Option("", "--project", help="Which project."),
    skip_tentative: bool = typer.Option(True, help="Only ask about firm commitments."),
) -> None:
    """Fill in the deadlines nobody stated out loud.

    The planner flags a commitment with no date and cannot chase it; the
    calendar lists it and cannot schedule it. Neither could ever ask you. This
    is the part that asks - it shows the words that created the obligation, and
    you say when it is due.
    """
    from datetime import date as _date

    from quorum.agents.dates import resolve_deadline
    from quorum.models import CommitmentStatus, CommitmentStrength, DeadlineResolution

    setup_logging("WARNING")
    _, found = _load_project(project_name)
    ledger = found.ledger
    today = _date.today()

    undated = [c for c in ledger.open_commitments() if c.deadline.resolved is None]
    if skip_tentative:
        undated = [c for c in undated if c.strength is CommitmentStrength.FIRM]

    if not undated:
        console.print(
            f"[green]Every open commitment on {found.meta.name} has a date.[/green]"
        )
        return

    console.print(f"[bold]{len(undated)} commitment(s) with no deadline.[/bold]")
    console.print(
        "[dim]Enter a date, or press Enter to leave it. "
        "'skip' to stop, 'drop' to abandon the commitment.[/dim]"
    )
    console.print()

    filled = 0
    dropped = 0
    for index, commitment in enumerate(undated, start=1):
        owner = commitment.assignee.display_name or "unassigned"
        console.print(f"[bold]{index}/{len(undated)}[/bold]  {safe(commitment.description)}")
        console.print(f"        [dim]{safe(owner)}[/dim]")
        for evidence in commitment.evidence[:1]:
            # The words are the point. "When is this due" is a question you can
            # only answer if you are reminded what was actually said about it.
            said = safe(evidence.quote.strip())
            console.print(f"        [dim]said: {said}[/dim]")
        if commitment.deadline.raw_text:
            spoken = safe(commitment.deadline.raw_text)
            console.print(
                f"        [dim]timing as spoken: {spoken} (could not be resolved)[/dim]"
            )

        try:
            answer = console.input("        due > ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            console.print("[dim]Stopped.[/dim]")
            break

        if not answer:
            console.print("        [dim]left without a date[/dim]")
            console.print()
            continue
        if answer.lower() in ("skip", "stop", "q", "quit"):
            break
        if answer.lower() in ("drop", "abandon", "cancel"):
            commitment.status = CommitmentStatus.DROPPED
            commitment.resolution_note = f"dropped during triage on {today}"
            dropped += 1
            console.print("        [magenta]dropped[/magenta]")
            console.print()
            continue

        resolved = resolve_deadline(answer, today)
        if resolved.value is None:
            # Refused rather than guessed: a wrong date silently produces a
            # calendar reminder on the wrong day, which is worse than none.
            console.print(
                f"        [red]Could not read {answer!r} as a date.[/red] "
                "Try 'next Friday' or 2026-09-01. Left without a date."
            )
            console.print()
            continue

        commitment.record_deadline_change(
            resolved.value, on=today, source="triage", note=answer
        )
        commitment.deadline.resolved = resolved.value
        commitment.deadline.raw_text = answer
        commitment.deadline.method = (
            resolved.method
            if resolved.method is not DeadlineResolution.NONE
            else DeadlineResolution.EXPLICIT
        )
        commitment.deadline.confidence = resolved.confidence
        filled += 1
        console.print(f"        [green]{resolved.value.isoformat()}[/green]")
        console.print()

    found.save_ledger()
    remaining = len([c for c in ledger.open_commitments() if c.deadline.resolved is None])
    console.print(
        f"[green]{filled} dated[/green]"
        + (f", {dropped} dropped" if dropped else "")
        + f". {remaining} still without a date."
    )
    if filled:
        console.print(f"  Next: [bold]quorum calendar --project {found.meta.id}[/bold]")


@app.command()
def done(
    what: str = typer.Argument(..., help="Part of the commitment's description."),
    project_name: str = typer.Option("", "--project", help="Which project."),
    drop: bool = typer.Option(False, help="Mark abandoned instead of complete."),
) -> None:
    """Tell it a commitment is finished (or abandoned)."""
    from datetime import date as _date

    from rapidfuzz import fuzz

    from quorum.models import CommitmentStatus

    setup_logging("WARNING")
    _, found = _load_project(project_name)
    ledger = found.ledger

    candidates = ledger.open_commitments()
    if not candidates:
        console.print("[dim]Nothing open.[/dim]")
        return

    # A vague query matches everything. `done "the"` once closed a real
    # commitment, and closing the wrong one is silent - nothing later reopens it.
    filler = {"the", "a", "an", "it", "that", "this", "and", "of", "to", "my", "our"}
    words = [w for w in what.lower().split() if w not in filler]
    if len(what.strip()) < 4 or not words:
        console.print(f"[red]{what!r} is too vague to identify a commitment.[/red] Open:")
        for item in candidates[:8]:
            console.print(f"  - {safe(item.description)}")
        raise typer.Exit(1)

    scored = sorted(
        ((fuzz.token_set_ratio(what.lower(), c.description.lower()), c) for c in candidates),
        key=lambda pair: -pair[0],
    )
    best_score, best = scored[0]
    if best_score < 55:
        console.print(f"[red]Nothing matches {what!r}.[/red] Open commitments:")
        for _, item in scored[:5]:
            console.print(f"  - {safe(item.description)}")
        raise typer.Exit(1)

    # Ambiguity is resolved by asking, not by picking. Closing the wrong
    # commitment is silent and nothing later reopens it.
    if len(scored) > 1 and scored[1][0] >= best_score - 5:
        console.print(f"[yellow]Ambiguous - {what!r} matches several:[/yellow]")
        for score, item in scored[:3]:
            console.print(f"  - {safe(item.description)} [dim]({score:.0f})[/dim]")
        console.print("Be more specific.")
        raise typer.Exit(1)

    best.status = CommitmentStatus.DROPPED if drop else CommitmentStatus.VERIFIED_DONE
    best.resolution_note = f"marked {'dropped' if drop else 'done'} by you on {_date.today()}"
    found.save_ledger()

    verb = "Dropped" if drop else "Closed"
    console.print(f"[green]{verb}:[/green] {best.description}")
    console.print(f"[dim]{len(ledger.open_commitments())} still open.[/dim]")


@app.command()
def name(
    which: str = typer.Argument("", help="Meeting id or words from its title."),
    handle: str = typer.Argument("", help="The short name to give it."),
    project_name: str = typer.Option("", "--project", help="Which project."),
) -> None:
    """Give a meeting or lecture a short name you can use in chat.

      quorum name --project dsa                       list what is named
      quorum name postfix kickoff --project dsa       name it @kickoff
    """
    from quorum.chat.naming import list_meetings, resolve_meeting, set_handle

    setup_logging("WARNING")
    workspace, found = _load_project(project_name)

    if not which:
        refs = list_meetings(found)
        if not refs:
            console.print("[yellow]Nothing recorded on this project yet.[/yellow]")
            return
        table = Table(title=f"{found.meta.name} - recordings", header_style="bold")
        table.add_column("Handle")
        table.add_column("Title", overflow="fold")
        table.add_column("Date")
        table.add_column("Kind")
        table.add_column("Lines", justify="right")
        for ref in refs:
            table.add_row(
                f"@{ref.handle}" if ref.handle else "[dim]-[/dim]",
                safe(ref.title) or "[dim]untitled[/dim]",
                ref.meeting_date.isoformat() if ref.meeting_date else "-",
                ref.kind, str(ref.utterances),
            )
        console.print(table)
        console.print('\n[dim]Name one: [bold]quorum name "postfix" kickoff[/bold][/dim]')
        return

    resolution = resolve_meeting(found, which)
    if resolution.ambiguous:
        console.print(f"[yellow]{which!r} matches several:[/yellow]")
        for ref in resolution.candidates:
            console.print(f"  - {safe(ref.label)} [dim]({ref.meeting_id})[/dim]")
        console.print("Be more specific, or use the id.")
        raise typer.Exit(1)
    if not resolution.ok:
        console.print(f"[red]No meeting matching {which!r}.[/red] "
                      "Run [bold]quorum name[/bold] with no arguments to list them.")
        raise typer.Exit(1)

    if not handle:
        console.print(f"{resolution.match.label} [dim]({resolution.match.meeting_id})[/dim]")
        console.print("Give a second argument to name it.")
        return

    try:
        stored = set_handle(found, resolution.match.meeting_id, handle)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    workspace.save(found)

    console.print(f"[green]@{stored}[/green] -> {resolution.match.title or 'untitled'}")
    console.print(f'  Try: [bold]quorum chat --project {found.meta.id}[/bold] '
                  f'then ask "@{stored} what was the main point?"')


@app.command()
def chat(
    question: str = typer.Argument("", help="Ask once and exit. Omit for a session."),
    project_name: str = typer.Option("", "--project", help="Which project."),
    meeting: str = typer.Option("", help="Start focused on one meeting or lecture."),
    everywhere: bool = typer.Option(
        False, "--all", help="Search every project at once. Read-only."
    ),
) -> None:
    """Talk to a project: ask about its meetings, or tell it to do something.

      quorum chat --project dsa
      quorum chat "what did @kickoff decide about storage" --project team

    Inside a session:
      @handle          focus on one meeting        /meetings   list them
      /open            open commitments            /scope      what is in focus
      /exit            leave
    """
    from quorum.chat import ChatAgent, Conversation, ToolContext
    from quorum.chat.agent import render_answer
    from quorum.chat.naming import list_meetings, resolve_meeting

    setup_logging("WARNING")

    if everywhere:
        from quorum.chat.federated import FederatedMemory, all_projects
        from quorum.workspace import Workspace

        workspace = Workspace()
        projects = all_projects(workspace)
        if not projects:
            console.print("[yellow]Nothing recorded in any project yet.[/yellow]")
            raise typer.Exit(1)
        # One embedder for every store. Scores from two indexes are only
        # comparable if the same model produced them, and it loads ~100 MB once
        # instead of once per project.
        found = projects[0]
        context = ToolContext(
            project=found, workspace=workspace,
            memory=FederatedMemory(projects), federated=True,
        )
    else:
        workspace, found = _load_project(project_name)
        context = ToolContext(project=found, workspace=workspace)

    conversation = Conversation()
    if meeting:
        resolution = resolve_meeting(found, meeting)
        if not resolution.ok:
            console.print(f"[red]No meeting matching {meeting!r}.[/red]")
            raise typer.Exit(1)
        conversation.scope_meeting = resolution.match.handle or resolution.match.meeting_id
        context.scope_meeting = conversation.scope_meeting

    agent = ChatAgent(context)

    def respond(text: str) -> None:
        turn = agent.ask(text, conversation)

        if turn.needs_confirmation:
            console.print()
            show(turn.pending.preview, style="yellow")
            console.print()
            if typer.confirm("Do it?", default=False):
                result = agent.confirm(turn.pending)
                show(result.text, style="green" if result.ok else "red")
                turn.message = result.text
            else:
                console.print("[dim]Left alone.[/dim]")
                turn.message = "declined by the user"
            turn.pending = None
            conversation.add(turn)
            return

        if turn.answer is not None:
            console.print()
            # Never through rich's markup parser: a code answer is mostly
            # square brackets, and every one of them would be eaten.
            show(render_answer(turn.answer))
            for index in turn.answer.cited:
                hit = turn.answer.hits[index - 1]
                console.print(f"  [dim]\\[{index}] {hit.meeting_date} "
                              f"{safe(hit.text[:90])}...[/dim]")
        elif turn.message:
            console.print()
            show(turn.message)
        if turn.tools_used:
            console.print(f"[dim]({', '.join(turn.tools_used)})[/dim]")
        conversation.add(turn)

    if question:
        respond(question)
        return

    if everywhere:
        from quorum.chat.federated import all_projects

        every = all_projects(workspace)
        total = sum(len(list_meetings(p)) for p in every)
        console.print(f"[bold]All projects[/bold] - {len(every)} project(s), "
                      f"{total} recording(s)")
        console.print("[dim]Read-only: use --project <name> to change anything.[/dim]")
    else:
        recordings = list_meetings(found)
        console.print(f"[bold]{found.meta.name}[/bold] - {len(recordings)} recording(s), "
                      f"{len(found.ledger.open_commitments())} open commitment(s)")
    if conversation.scope_meeting:
        console.print(f"[dim]Focused on @{conversation.scope_meeting}[/dim]")
    console.print("[dim]Ask anything. /meetings to list, @handle to focus, "
                  "/exit to leave.[/dim]\n")

    while True:
        try:
            line = console.input("[bold cyan]you >[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Bye.[/dim]")
            return
        if not line:
            continue
        if line in ("/exit", "/quit", "exit", "quit"):
            console.print("[dim]Bye.[/dim]")
            return
        if line == "/meetings":
            for ref in list_meetings(found):
                console.print(f"  [cyan]@{ref.handle or '(unnamed)'}[/cyan] "
                              f"{safe(ref.title) or 'untitled'} "
                              f"[dim]{ref.meeting_date}[/dim]")
            continue
        if line == "/scope":
            console.print(f"  {conversation.scope_meeting or '[dim]whole project[/dim]'}")
            continue
        if line == "/open":
            for item in found.ledger.open_commitments():
                due = item.deadline.resolved
                console.print(f"  {safe(item.description)} [dim]"
                              f"{due.isoformat() if due else 'no date'}[/dim]")
            continue

        respond(line)


@app.command()
def auth(
    revoke_token: bool = typer.Option(False, "--revoke", help="Forget the stored token."),
    status_only: bool = typer.Option(False, "--status", help="Report without logging in."),
) -> None:
    """Authorise Google Calendar. Opens a browser once, then stores a token."""
    from quorum.integrations import (
        CALENDAR_SCOPES,
        GoogleAuthError,
        authorise,
        credentials_status,
    )
    from quorum.integrations import revoke as revoke_stored

    setup_logging("WARNING")
    ensure_dirs()

    if revoke_token:
        if revoke_stored():
            console.print("[green]Stored token deleted.[/green]")
            console.print(
                "[dim]The grant itself still exists. Remove it at "
                "https://myaccount.google.com/permissions[/dim]"
            )
        else:
            console.print("[dim]No stored token to delete.[/dim]")
        return

    current = credentials_status()
    console.print(f"Google: [bold]{current.message}[/bold]")
    console.print(f"[dim]Scope requested: {', '.join(CALENDAR_SCOPES)}[/dim]")
    if status_only or current.ready:
        if current.ready:
            console.print("[dim]Re-run with --revoke to sign out.[/dim]")
        return

    console.print("\nA browser window will open for Google sign-in.")
    try:
        authorise(interactive=True)
    except GoogleAuthError as exc:
        console.print(f"\n[red]{exc}[/red]")
        raise typer.Exit(1)

    console.print(f"[green]Authorised.[/green] {credentials_status().message}")
    console.print("  Next: [bold]quorum calendar --project <name>[/bold]")


@app.command()
def calendar(
    project_name: str = typer.Option("", "--project", help="Which project."),
    apply_changes: bool = typer.Option(False, "--apply", help="Actually write the events."),
    tentative: bool = typer.Option(False, help="Include tentative commitments too."),
    calendar_id: str = typer.Option("", help="Target calendar. Defaults to your primary."),
    keep_resolved: bool = typer.Option(
        False, help="Leave events for commitments that are now closed."
    ),
) -> None:
    """Put deadlines in your calendar, with reminders before each one.

    Runs as a dry run by default: it prints exactly what it would change and
    writes nothing. `--apply` performs it, behind the approval gate.
    """
    from datetime import date as _date

    from quorum.execution import ApprovalGate, CalendarConfig, CalendarSync, ChangeKind
    from quorum.execution.calendar import CalendarTransport
    from quorum.integrations import GoogleAuthError, credentials_status, get_calendar_service
    from quorum.tracking import ActionType, PlannedAction

    setup_logging("WARNING")
    settings = get_settings()
    _, found = _load_project(project_name)

    config = CalendarConfig(
        calendar_id=calendar_id or settings.calendar_id,
        reminder_days=settings.reminder_days(),
        reminder_hour=settings.reminder_hour,
        include_tentative=tentative,
        delete_resolved=not keep_resolved,
    )

    google = credentials_status()
    service = None
    if google.ready:
        try:
            service = get_calendar_service()
        except GoogleAuthError as exc:
            console.print(f"[yellow]{exc}[/yellow]")
    elif apply_changes:
        console.print(f"[red]Cannot write: {google.message}[/red]")
        console.print("Run [bold]quorum auth[/bold] first.")
        raise typer.Exit(1)

    sync = CalendarSync(service, config)
    plan = sync.plan(found.ledger, _date.today())

    leads = ", ".join(f"{d}d" for d in config.reminder_days)
    console.print(f"[bold]{found.meta.name}[/bold] -> calendar {config.calendar_id} "
                  f"[dim](reminders {leads} before, at {config.reminder_hour:02d}:00)[/dim]\n")

    if plan.is_empty and not plan.undated:
        console.print("[green]Calendar already matches the ledger.[/green]")
        return

    if plan.writes:
        table = Table(header_style="bold")
        table.add_column("Change")
        table.add_column("Due")
        table.add_column("Commitment", overflow="fold")
        table.add_column("Why", overflow="fold")
        colour = {
            ChangeKind.CREATE: "green", ChangeKind.UPDATE: "yellow", ChangeKind.DELETE: "magenta"
        }
        for change in plan.writes:
            table.add_row(
                f"[{colour[change.kind]}]{change.kind.value}[/{colour[change.kind]}]",
                change.due.isoformat() if change.due else "-",
                safe(change.title),
                safe(change.reason),
            )
        console.print(table)

    unchanged = len(plan.of_kind(ChangeKind.UNCHANGED))
    if unchanged:
        console.print(f"[dim]{unchanged} event(s) already correct.[/dim]")

    if plan.undated:
        # These are exactly the commitments the planner flags and cannot chase.
        # Naming them here is the difference between "the agent ignored it" and
        # a question the user can answer.
        console.print(f"\n[yellow]{len(plan.undated)} commitment(s) have no deadline[/yellow] "
                      "and cannot be scheduled:")
        for item in plan.undated:
            owner = item.assignee.display_name or "unassigned"
            console.print(f"  - {safe(item.description)} [dim]({safe(owner)})[/dim]")

    if plan.is_empty:
        return

    if not apply_changes:
        console.print(f"\n[yellow]Dry run - nothing written.[/yellow] {plan.summary_line()}.")
        console.print("Re-run with [bold]--apply[/bold] to make these changes.")
        return

    if service is None:
        console.print("\n[red]No calendar service available.[/red] Run [bold]quorum auth[/bold].")
        raise typer.Exit(1)

    # One approval for the whole plan, not one per event. The human reads a
    # complete list of what will change and consents to that list; twenty
    # separate prompts would be approval fatigue, which is how gates end up
    # being clicked through without being read.
    gate = ApprovalGate(require_approval=settings.require_approval)
    action = PlannedAction(
        commitment_id=f"calendar:{found.meta.id}",
        action=ActionType.SCHEDULE,
        reason=plan.summary_line(),
        priority=1,
    )
    pending = gate.propose(action, f"Calendar sync: {plan.summary_line()}", body=plan.render())

    console.print(f"\n[bold]About to change {len(plan.writes)} event(s)[/bold] "
                  f"on {config.calendar_id}.")
    if not typer.confirm("Apply?", default=False):
        gate.reject(pending.id, "declined at the prompt")
        console.print("[yellow]Nothing written.[/yellow]")
        return

    transport = CalendarTransport(sync, plan)
    token = gate.approve(pending.id)
    gate.execute(pending.id, token, transport)
    result = transport.result

    console.print(
        f"[green]Done.[/green] {result.created} added, {result.updated} updated, "
        f"{result.deleted} removed."
    )
    for failure in result.failed:
        console.print(f"  [red]failed:[/red] {failure}")


@app.command()
def resume(
    meeting_id: str = typer.Argument("", help="Which run to continue."),
    list_only: bool = typer.Option(False, "--list", help="Show interrupted runs and stop."),
) -> None:
    """Continue a pipeline run that died partway.

    Free-tier quota walls kill long runs. The pipeline checkpoints after every
    stage, so the transcription and any completed extraction survive - this
    restarts at the stage that failed rather than at the beginning.
    """
    from quorum.pipeline import IngestGraph, RunStatus, interrupted_runs

    setup_logging("WARNING")
    runs = interrupted_runs()

    if list_only or not meeting_id:
        if not runs:
            console.print("[green]No interrupted runs.[/green]")
            return
        table = Table(title="Interrupted runs", header_style="bold")
        table.add_column("Meeting")
        table.add_column("Title", overflow="fold")
        table.add_column("Date")
        table.add_column("Done")
        table.add_column("Next")
        for run in runs:
            table.add_row(
                run.meeting_id, safe(run.title) or "[dim]untitled[/dim]",
                run.meeting_date.isoformat() if run.meeting_date else "-",
                ", ".join(run.completed_stages) or "[dim]none[/dim]",
                run.next_stage,
            )
        console.print(table)
        if not list_only:
            console.print("\n[dim]Continue one: [bold]quorum resume <meeting-id>[/bold][/dim]")
        return

    graph = IngestGraph()
    with console.status(f"Resuming {meeting_id}..."):
        outcome = graph.resume(meeting_id)

    if outcome.status is RunStatus.NOT_FOUND:
        console.print(f"[red]{outcome.error}[/red]")
        console.print("Run [bold]quorum resume --list[/bold] to see what can be continued.")
        raise typer.Exit(1)

    if outcome.status is RunStatus.INTERRUPTED:
        console.print(f"[yellow]Interrupted again:[/yellow] {outcome.error}")
        console.print(f"[dim]Completed so far: "
                      f"{', '.join(outcome.state.get('completed_stages', [])) or 'none'}. "
                      f"Retry later - progress is kept.[/dim]")
        raise typer.Exit(1)

    record = outcome.record
    console.print(
        f"[green]Finished.[/green] Skipped {', '.join(outcome.resumed_from) or 'nothing'}; "
        f"ran {', '.join(outcome.stages_run) or 'nothing'}."
    )
    if record is not None:
        console.print(f"{len(record.commitments)} commitment(s), "
                      f"{len(record.decisions)} decision(s), "
                      f"{record.rejected_items} rejected as ungrounded.")
        _save_resumed(record, outcome)


def _save_resumed(record, outcome) -> None:
    """Fold a resumed run into its project, if it had one."""
    from quorum.workspace import Workspace

    project_id = outcome.state.get("project_id")
    if not project_id:
        console.print("[dim]No project on this run - nothing saved.[/dim]")
        return

    workspace = Workspace()
    project = workspace.get(project_id)
    if project is None:
        console.print(f"[yellow]Project {project_id!r} no longer exists - not saved.[/yellow]")
        return

    from quorum.models import Transcript

    transcript = Transcript.model_validate(outcome.state["transcript"])
    project.add_meeting(record, transcript)
    workspace.save(project)
    console.print(f"[green]Saved to {project.meta.name}.[/green] "
                  f"{len(project.ledger.open_commitments())} open commitment(s).")


@app.command()
def record(
    minutes: float = typer.Option(0.0, help="Stop after N minutes. 0 = until Ctrl+C."),
    roster: str = typer.Option(
        "", help='Participants: "Priya:priya@x.com,Sam:sam@x.com". Enables naming.'
    ),
    me: str = typer.Option("You", help="Your display name."),
    my_email: str = typer.Option("", help="Your email."),
    title: str = typer.Option("Live meeting", help="Meeting title."),
    project_name: str = typer.Option(
        "", "--project", help="Save into this project so it accumulates across meetings."
    ),
    devices: bool = typer.Option(False, help="List audio devices and exit."),
    keep_audio: bool = typer.Option(False, help="Keep the recorded WAV chunks."),
) -> None:
    """Record a live meeting from this machine and run the full pipeline.

    Works with Google Meet, Zoom, Teams or anything else that makes sound: it
    captures the system output (everyone else) and your microphone (you), so
    there is no bot in the call and no platform API involved.

    Start the meeting, then start this. Tell the other participants they are
    being recorded - in many places that is a legal requirement, not a courtesy.
    """
    import time as _time
    from datetime import date as _date

    from quorum.capture.audio import DualRecorder, RecorderConfig
    from quorum.capture.echo import suppress_echo
    from quorum.capture.speakers import RemoteSpeakerAttributor, SpeakerRoster, build_transcript
    from quorum.capture.transcribe import WhisperTranscriber
    from quorum.models import Speaker

    setup_logging("WARNING")
    ensure_dirs()

    if devices:
        try:
            found = DualRecorder.devices()
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Could not enumerate devices: {exc}[/red]")
            raise typer.Exit(1)
        console.print(f"[bold]Microphone:[/bold] {found['mic']['name']}")
        if found["loopback"]:
            console.print(f"[bold]System audio:[/bold] {found['loopback']['name']}")
        else:
            console.print("[red]No WASAPI loopback device found[/red] - "
                          "system audio cannot be captured.")
        return

    active_project = None
    if project_name:
        _, active_project = _load_project(project_name)
        # The project's member list seeds the roster, so you do not retype it
        # before every meeting.
        if not roster and active_project.meta.members:
            roster = active_project.roster_string()
        console.print(f"[dim]Project: {active_project.meta.name} "
                      f"({active_project.meta.meeting_count} meetings so far)[/dim]")

    others = []
    for index, entry in enumerate(p for p in roster.split(",") if p.strip()):
        name, _, email = entry.partition(":")
        others.append(
            Speaker(id=f"spk_r{index}", display_name=name.strip(),
                    email=email.strip() or None, aliases=[name.strip().split()[0]])
        )
    people = SpeakerRoster(
        you=Speaker(id="spk_you", display_name=me, email=my_email or None,
                    aliases=["I", "me"]),
        others=others,
    )

    console.print("[bold yellow]Recording is about to start.[/bold yellow]")
    console.print("Tell the other participants they are being recorded.")
    console.print(
        "[dim]Wear headphones if you can - otherwise your mic hears your own "
        "speakers and the other side's words can be attributed to you.[/dim]\n"
    )

    config = RecorderConfig(
        output_dir=(RUNS_DIR / "audio") if keep_audio else None, keep_wav=keep_audio
    )
    recorder = DualRecorder(config)
    try:
        recorder.start(announced=True)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Could not start recording: {exc}[/red]")
        raise typer.Exit(1)

    started = _time.time()
    deadline = started + minutes * 60 if minutes else None
    console.print("[green]Recording.[/green] Press Ctrl+C to stop.\n")
    try:
        while deadline is None or _time.time() < deadline:
            _time.sleep(1.0)
            # Elapsed wall clock. `captured_seconds` sums the microphone and
            # loopback streams, so it reports roughly double the real duration.
            console.print(
                f"  recording {(_time.time() - started) / 60:.1f} min, "
                f"{recorder.chunks.qsize()} chunk(s) queued, "
                f"{recorder.skipped_silent} silent chunk(s) skipped",
                end="\r",
            )
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping...[/yellow]")
    finally:
        recorder.stop()

    chunks = recorder.drain()
    if not chunks:
        console.print("[red]Nothing recorded.[/red] Was there any audio?")
        raise typer.Exit(1)

    console.print(f"\n{len(chunks)} chunk(s) to transcribe "
                  f"({recorder.skipped_silent} silent chunk(s) never uploaded)")

    transcriber = WhisperTranscriber()
    with console.status("Transcribing..."):
        segments = transcriber.transcribe_all(chunks)
    console.print(f"Transcribed: {transcriber.stats.segments} segment(s), "
                  f"{transcriber.stats.audio_seconds:.0f}s of audio "
                  f"({transcriber.stats.daily_budget_used:.1%} of the daily free budget)")

    if not segments:
        console.print("[red]No speech recognised.[/red]")
        raise typer.Exit(1)

    segments, echo = suppress_echo(segments)
    if echo.removed:
        console.print(
            f"[yellow]Removed {echo.removed} echoed segment(s)[/yellow] - your microphone "
            f"picked up your own speakers ({echo.echo_rate:.0%} of your speech)."
        )
        if echo.likely_no_headphones:
            console.print(
                "[bold yellow]Use headphones.[/bold yellow] Without them the remote audio "
                "loops back through your mic and can be misattributed to you."
            )

    transcript = build_transcript(
        segments, people, meeting_date=_date.today(), title=title,
        project_id=active_project.meta.id if active_project else None,
    )
    if len(people.others) >= 1:
        with console.status("Attributing speakers..."):
            RemoteSpeakerAttributor().attribute(transcript, people)

    console.print(f"\n[bold]Transcript:[/bold] {len(transcript.utterances)} utterances\n")
    show(transcript.as_dialogue(0, min(8, len(transcript.utterances))))
    if len(transcript.utterances) > 8:
        console.print(f"[dim]... and {len(transcript.utterances) - 8} more[/dim]")

    from quorum.pipeline import IngestGraph, RunStatus

    # Through the checkpointed graph rather than four calls in a row. The audio
    # is already spent by this point - a quota wall during extraction used to
    # throw away the transcription along with it.
    with console.status("Extracting commitments..."):
        outcome = IngestGraph().run(
            transcript, project_id=active_project.meta.id if active_project else None
        )

    if outcome.status is RunStatus.INTERRUPTED:
        console.print(f"\n[yellow]Extraction stopped:[/yellow] {outcome.error}")
        console.print(
            f"[dim]Completed: "
            f"{', '.join(outcome.state.get('completed_stages', [])) or 'nothing'}. "
            f"The transcript is checkpointed - nothing recorded is lost.[/dim]"
        )
        console.print(f"  Continue later: [bold]quorum resume {transcript.meeting_id}[/bold]")
        raise typer.Exit(1)

    meeting_record = outcome.record
    commitments = meeting_record.commitments

    table = Table(title="\nCommitments found", header_style="bold")
    table.add_column("Owner")
    table.add_column("Commitment", overflow="fold")
    table.add_column("Due")
    table.add_column("Strength")
    for commitment in commitments:
        table.add_row(
            safe(commitment.assignee.display_name)
            or f"[dim]{safe(commitment.assignee.raw_mention or '')}[/dim]",
            safe(commitment.description),
            commitment.deadline.resolved.isoformat() if commitment.deadline.resolved else "-",
            commitment.strength.value,
        )
    console.print(table)
    console.print(
        f"[dim]{meeting_record.rejected_items} item(s) rejected as ungrounded. "
        f"Nothing was sent - the approval gate is enabled.[/dim]"
    )

    # The summary, which the ledger cannot supply. Costs a call per segment plus
    # one synthesis, and runs after extraction so a failure here still leaves the
    # commitments - the part the whole project is about.
    from quorum.analysis.meeting import MeetingSummariser

    try:
        with console.status("Writing the summary..."):
            digest = MeetingSummariser().summarise(transcript, segments_of(outcome))
        meeting_record.summary = digest.summary
    except Exception as exc:  # noqa: BLE001 - a summary is worth less than the ledger
        console.print(f"[dim]Could not summarise this meeting: {exc}[/dim]")
        digest = None

    if digest is not None and digest.summary:
        console.print()
        show(digest.as_markdown(meeting_record))

    if active_project is None:
        console.print(
            "\n[dim]Not saved. Use [bold]--project <name>[/bold] to build history "
            "across meetings and enable chasing.[/dim]"
        )
        return

    from quorum.chat.naming import register_meeting

    meeting_record.title = title
    before = len(active_project.ledger.open_commitments())
    active_project.add_meeting(meeting_record, transcript)
    handle = register_meeting(active_project, transcript)
    workspace_ref, _ = _load_project(active_project.meta.id)
    workspace_ref.save(active_project)

    if digest is not None and digest.summary:
        notes_path = free_path(
            RUNS_DIR / "notes", f"{transcript.meeting_date}_{transcript.meeting_id}", ".md"
        )
        notes_path.parent.mkdir(parents=True, exist_ok=True)
        notes_path.write_text(digest.as_markdown(meeting_record), encoding="utf-8")
        console.print(f"[green]Minutes saved to {notes_path}[/green]")

    after = len(active_project.ledger.open_commitments())
    console.print(
        f"\n[green]Saved to {active_project.meta.name}.[/green] "
        f"Open commitments: {before} -> {after}. "
        f"{len(meeting_record.status_updates)} status update(s) applied, "
        f"{outcome.state.get('indexed', 0)} item(s) indexed."
    )

    undated = [c for c in commitments if c.deadline.resolved is None and c.is_actionable]
    emails = _communication_count(commitments)

    console.print(f"  Next: [bold]quorum today --project {active_project.meta.id}[/bold]")
    if undated:
        console.print(f"        [bold]quorum triage --project {active_project.meta.id}[/bold]"
                      f" - {len(undated)} commitment(s) need a deadline from you")
    if emails:
        console.print(f"        [bold]quorum drafts --project {active_project.meta.id}[/bold]"
                      f" - {emails} email(s) were promised in this meeting")
    if any(c.deadline.resolved for c in commitments):
        console.print(f"        [bold]quorum calendar --project {active_project.meta.id}"
                      f"[/bold] to put the deadlines in your calendar")
    console.print(f"        [bold]quorum chat --project {active_project.meta.id}[/bold] "
                  f"to ask about it - this one is [cyan]@{handle}[/cyan]")


def _communication_count(commitments) -> int:
    from quorum.execution import find_communications

    return len(find_communications(commitments))


def segments_of(outcome):
    """Segments the pipeline already produced, rebuilt from its state.

    Re-segmenting would be cheap but would not necessarily agree with what the
    extractor saw, and two different segmentations of one meeting is exactly the
    kind of quiet inconsistency that is hard to notice and hard to explain.
    """
    from quorum.models import Segment

    return [Segment.model_validate(s) for s in outcome.state.get("segments", [])]

def _analyse_lecture(transcript, active_project, out: str = "") -> None:
    """Notes, disk, index. Shared by live capture and transcript re-analysis."""
    from datetime import date as _date

    from quorum.agents.embedding import LexicalEmbedder
    from quorum.agents.segmenter import Segmenter, SegmenterConfig
    from quorum.analysis import LectureAnalyser

    # Lectures are monologue: larger segments give the model more context per
    # call, and there are no turn boundaries worth preserving.
    topics = Segmenter(
        config=SegmenterConfig(max_tokens=2400, min_utterances=6),
        embedder=LexicalEmbedder(),
    ).segment(transcript)

    with console.status(f"Taking notes across {len(topics)} topic(s)..."):
        notes = LectureAnalyser().analyse(transcript, topics)

    # Local, free, and not derived from the talk at all - read off how it was
    # watched. Never fatal: a lecture watched straight through simply has none.
    try:
        from quorum.analysis.replays import find_replays

        notes.replays = find_replays(transcript)
    except Exception as exc:  # noqa: BLE001 - a bonus section must not cost the notes
        console.print(f"[dim]Could not check for replayed sections: {exc}[/dim]")

    console.print()
    # Study notes are full of code, subscripts and formulae. The saved .md file
    # was always correct; only this preview was being mangled.
    show(notes.as_markdown())
    console.print(
        f"\n[dim]{notes.llm_calls} calls, {notes.total_tokens:,} tokens, "
        f"{notes.latency_s:.0f}s, cost $0.00"
        + (f", {notes.filler_dropped} content-free item(s) dropped"
           if notes.filler_dropped else "")
        + "[/dim]"
    )

    path = Path(out) if out else RUNS_DIR / "notes" / f"{_date.today()}_{transcript.meeting_id}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(notes.as_markdown(), encoding="utf-8")
    console.print(f"[green]Notes saved to {path}[/green]")

    # Keep the transcript too. Notes are a lossy summary, and the words actually
    # spoken are worth more than the cost of storing them.
    if active_project is not None:
        active_project.transcripts_dir.mkdir(parents=True, exist_ok=True)
        transcript_path = active_project.transcripts_dir / f"{transcript.meeting_id}.json"
    else:
        transcript_path = RUNS_DIR / "transcripts" / f"{transcript.meeting_id}.json"
        transcript_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_path.write_text(transcript.model_dump_json(indent=2), encoding="utf-8")
    console.print(f"[green]Transcript saved to {transcript_path}[/green]")
    # Timestamps are positions in the recording. If the session was not linear -
    # a skip, a replay, a speed change - they do not correspond to the video,
    # and searching the words is the only navigation that still works.
    console.print(
        "[dim]Times are positions in this recording. If you skipped or replayed "
        "parts, find a moment by its words instead:\n"
        '  quorum transcript <name> --search "a phrase you remember"[/dim]'
    )

    if active_project is not None:
        from quorum.chat.naming import register_meeting
        from quorum.memory import ProjectMemory
        from quorum.workspace import Workspace

        indexed = ProjectMemory(active_project.memory_dir).index_notes(
            transcript.meeting_id, notes.title, _date.today(), notes
        )
        # Nameable straight away. Requiring `quorum name` first would mean the
        # common case - watch one lecture, ask about it - needs a second command
        # before the first question can be asked.
        handle = register_meeting(active_project, transcript)
        Workspace().save(active_project)
        console.print(
            f"[green]Indexed {indexed} item(s) into {active_project.meta.name}[/green] "
            f"as [cyan]@{handle}[/cyan]."
        )
        console.print(
            f"  Ask about it: [bold]quorum chat --project {active_project.meta.id}[/bold]"
        )


@app.command()
def learn(
    minutes: float = typer.Option(0.0, help="Stop after N minutes. 0 = until Ctrl+C."),
    title: str = typer.Option("", help="What you are watching."),
    project_name: str = typer.Option(
        "", "--project", help="Save notes here so `quorum ask` can search them."
    ),
    system_only: bool = typer.Option(
        True, help="Capture the video's audio only. Use --no-system-only for a lecture in a room, where the sound reaches the microphone instead."
    ),
    from_transcript: str = typer.Option(
        "", help="Re-analyse a saved transcript instead of recording again."
    ),
    speed: float = typer.Option(
        1.0, help="Constant playback speed, so timestamps map back to the video. "
                  "Only valid if you do not seek or change speed part-way."
    ),
    out: str = typer.Option("", help="Write the notes to this markdown file."),
) -> None:
    """Take study notes from a lecture, talk or seminar.

    Works on anything that makes sound - a YouTube lecture, a recorded seminar,
    a live webinar, a conference talk. Produces a summary, timestamped key
    points, jargon explained in plain English, worked examples and what the
    speaker assumed you already knew.

    Start the video, then start this.
    """
    import time as _time
    from datetime import date as _date

    from quorum.agents.embedding import LexicalEmbedder
    from quorum.agents.segmenter import Segmenter, SegmenterConfig
    from quorum.analysis import LectureAnalyser
    from quorum.capture.audio import SYSTEM, DualRecorder, RecorderConfig
    from quorum.capture.echo import suppress_echo
    from quorum.capture.speakers import SpeakerRoster, build_transcript
    from quorum.capture.transcribe import WhisperTranscriber

    setup_logging("WARNING")
    ensure_dirs()

    active_project = None
    if project_name:
        _, active_project = _load_project(project_name)

    if from_transcript:
        # Re-analysing a stored transcript costs no recording time and no
        # transcription quota. Prompt changes can then be judged against the
        # same lecture rather than a different one.
        from quorum.models import Transcript as TranscriptModel

        source = Path(from_transcript)
        if not source.exists():
            _, owner = _load_project(project_name) if project_name else (None, None)
            candidates = list(owner.transcripts_dir.glob("*.json")) if owner else []
            match = next(
                (c for c in candidates if from_transcript.lower() in c.stem.lower()), None
            )
            if match is None:
                console.print(f"[red]No transcript at {from_transcript}[/red]")
                if candidates:
                    console.print("Available: " + ", ".join(c.stem for c in candidates))
                raise typer.Exit(1)
            source = match

        transcript = TranscriptModel.model_validate_json(source.read_text(encoding="utf-8"))
        if title:
            transcript.title = title
        console.print(f"[dim]Re-analysing {source.name} "
                      f"({len(transcript.utterances)} utterances)[/dim]")
        _analyse_lecture(transcript, active_project, out)
        return

    # The instruction depends on where the sound is coming from. "Start the
    # video" is confusing advice to someone sitting in a lecture theatre.
    if system_only:
        console.print("[bold]Listening to your speakers.[/bold] "
                      "Start the video now if you have not.")
    else:
        console.print("[bold]Listening to your microphone.[/bold] "
                      "Point the laptop towards the speaker.")
    console.print("[dim]Press Ctrl+C when the lecture ends.[/dim]\n")

    recorder = DualRecorder(RecorderConfig())
    try:
        recorder.start(announced=True)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Could not start recording: {exc}[/red]")
        raise typer.Exit(1)

    started = _time.time()
    deadline = started + minutes * 60 if minutes else None
    try:
        while deadline is None or _time.time() < deadline:
            _time.sleep(1.0)
            # Wall clock, not `captured_seconds` - that sums both channels, so
            # it read as double the real elapsed time and looked like half the
            # lecture had gone missing.
            console.print(
                f"  listening {(_time.time() - started) / 60:.1f} min, "
                f"{recorder.skipped_silent} silent chunk(s) skipped",
                end="\r",
            )
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping...[/yellow]")
    finally:
        recorder.stop()

    chunks = recorder.drain()
    if system_only:
        # A lecture is the speaker, not you. Dropping the microphone halves the
        # audio-seconds spent and removes echo as a concern entirely.
        chunks = [c for c in chunks if c.channel == SYSTEM]
    if not chunks:
        if system_only:
            # The default listens to the speakers, which is right for a video
            # and captures nothing at all in a lecture hall. "Was the video
            # playing?" is a baffling thing to be asked while sitting in a room.
            console.print("[red]No audio captured from your speakers.[/red]")
            console.print(
                "If a video was playing, check the output device with "
                "[bold]quorum record --devices[/bold]."
            )
            console.print(
                "If you are in a room and the sound is not coming from this "
                "laptop, record the microphone instead:"
            )
            console.print(
                '  [bold]quorum learn --no-system-only --title "..." '
                "--project <name>[/bold]"
            )
        else:
            console.print("[red]No audio captured.[/red] Is the microphone muted?")
        raise typer.Exit(1)

    transcriber = WhisperTranscriber()
    with console.status(f"Transcribing {len(chunks)} chunk(s)..."):
        segments = transcriber.transcribe_all(chunks)
    if not system_only:
        segments, _ = suppress_echo(segments)

    if speed != 1.0:
        from quorum.capture.transcribe import rescale

        segments = rescale(segments, speed)
        console.print(
            f"[dim]Timestamps scaled by {speed}x to match the video. "
            "Only correct if you watched straight through at that speed - "
            "seeking or changing speed breaks the mapping.[/dim]"
        )

    console.print(f"Transcribed {transcriber.stats.audio_seconds:.0f}s of audio "
                  f"({transcriber.stats.daily_budget_used:.1%} of today's free budget)")

    if not segments:
        console.print("[red]No speech recognised.[/red]")
        raise typer.Exit(1)

    transcript = build_transcript(
        segments, SpeakerRoster.solo("You"), meeting_date=_date.today(),
        title=title or "Lecture",
    )
    _analyse_lecture(transcript, active_project, out)


@app.command()
def transcript(
    which_words: list[str] = typer.Argument(
        None, metavar="[WHICH]...",
        help="Meeting id or words from its title. Omit to list.",
    ),
    project_name: str = typer.Option("", "--project", help="Which project."),
    file: str = typer.Option("", help="Read a transcript JSON directly instead."),
    style: str = typer.Option(
        "", help="speakers | timestamped | plain | markdown | srt. "
                 "Defaults to timestamped for a lecture, speakers otherwise."
    ),
    speaker: str = typer.Option("", help="Only this person's lines."),
    start: str = typer.Option("", help='Skip before this time, e.g. "12:30".'),
    end: str = typer.Option("", help="Stop after this time."),
    search: str = typer.Option("", help="Only lines containing this text."),
    out: str = typer.Option("", help="Write to a file instead of printing."),
    who: bool = typer.Option(False, help="Show who spoke and how much, then exit."),
) -> None:
    """Print the full transcript of a meeting or lecture.

    Examples:

      quorum transcript --project dsa                    list what is available
      quorum transcript postfix --project dsa            print it
      quorum transcript postfix --project dsa --style srt --out lecture.srt
      quorum transcript standup --project team --speaker "Priya"
      quorum transcript seminar --project x --start 40:00 --end 55:00
      quorum transcript standup --project team --search "deadline"
    """
    from rapidfuzz import fuzz

    from quorum.export import Style, parse_time, render, stats
    from quorum.models import Transcript as TranscriptModel

    setup_logging("WARNING")
    # Variadic and rejoined, so a title can be typed as it reads. Requiring
    # quotes meant copying a title straight off the listing above failed with
    # "unexpected extra argument", which is a confusing answer to an obvious
    # thing to try.
    which = " ".join(which_words or []).strip()

    if file:
        path = Path(file)
        if not path.exists():
            console.print(f"[red]No such file: {file}[/red]")
            raise typer.Exit(1)
        found = TranscriptModel.model_validate_json(path.read_text(encoding="utf-8"))
        available = [found]
    else:
        _, active = _load_project(project_name)
        available = active.transcripts()
        loose = RUNS_DIR / "transcripts"
        if loose.exists():
            available += [
                TranscriptModel.model_validate_json(p.read_text(encoding="utf-8"))
                for p in sorted(loose.glob("*.json"))
            ]
        if not available:
            console.print("[yellow]No transcripts stored yet.[/yellow] "
                          "Record with [bold]--project[/bold] to keep them.")
            raise typer.Exit(1)

    if not which and not file:
        table = Table(title="Stored transcripts", header_style="bold")
        table.add_column("id")
        table.add_column("title")
        table.add_column("date")
        table.add_column("lines", justify="right")
        for item in available:
            table.add_row(item.meeting_id, safe(item.title) or "-",
                          item.meeting_date.isoformat(), str(len(item.utterances)))
        console.print(table)
        console.print("\n[dim]Print one: quorum transcript <id or title words>[/dim]")
        return

    if which:
        scored = sorted(
            available,
            key=lambda t: -max(
                fuzz.partial_ratio(which.lower(), t.meeting_id.lower()),
                fuzz.partial_ratio(which.lower(), (t.title or "").lower()),
            ),
        )
        chosen = scored[0]
    else:
        chosen = available[0]

    if who:
        summary = stats(chosen)
        table = Table(title=f"{chosen.title or chosen.meeting_id} - who spoke",
                      header_style="bold")
        table.add_column("Speaker")
        table.add_column("Words", justify="right")
        table.add_column("Share", justify="right")
        for name, row in summary["by_speaker"].items():
            table.add_row(name, str(row["words"]), f"{row['share']:.0%}")
        console.print(table)
        console.print(f"[dim]{summary['utterances']} lines, {summary['words']} words[/dim]")
        return

    # Labelling every line of a one-voice lecture "Remote participant" is noise
    # around the words you actually came for. The placeholder speaker means the
    # roster cannot answer this - only who spoke can.
    chosen_style = style or ("timestamped" if chosen.is_monologue else "speakers")
    try:
        text = render(
            chosen, Style(chosen_style), speaker=speaker or None,
            start_s=parse_time(start), end_s=parse_time(end), search=search or None,
        )
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)

    if not text:
        console.print("[yellow]Nothing matched those filters.[/yellow]")
        raise typer.Exit(1)

    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(text, encoding="utf-8")
        console.print(f"[green]Written to {out}[/green] ({len(text.splitlines())} lines)")
    else:
        # Printed raw, not through rich markup - a transcript containing [square
        # brackets] would otherwise be eaten as formatting tags.
        print(text)


@app.command()
def ask(
    question: str = typer.Argument(..., help="What you want to know."),
    project_name: str = typer.Option("", "--project", help="Which project's notes."),
    k: int = typer.Option(6, help="How many passages to retrieve."),
) -> None:
    """Answer a question from your indexed lectures and meetings."""
    from quorum.llm.providers import ModelTier
    from quorum.llm.router import get_router
    from quorum.memory import ProjectMemory

    setup_logging("WARNING")
    _, found = _load_project(project_name)
    memory = ProjectMemory(found.memory_dir)

    hits = memory.recall(question, k=k)
    if not hits:
        console.print("[yellow]Nothing indexed matches that.[/yellow] "
                      "Record a lecture with [bold]quorum learn --project ...[/bold] first.")
        raise typer.Exit(1)

    passages = "\n\n".join(f"[{i + 1}] ({hit.meeting_date}) {hit.text}"
                           for i, hit in enumerate(hits))
    prompt = (
        f"Retrieved passages from the user's own notes:\n\n{passages}\n\n"
        f"Question: {question}\n\n"
        "Answer using only these passages, citing the [n] you used. If they do "
        "not contain the answer, say so plainly rather than filling the gap from "
        "general knowledge - the user is asking what THEIR material said."
    )
    with console.status("Thinking..."):
        response = get_router().complete(
            prompt, tier=ModelTier.BALANCED, max_tokens=1024, purpose="ask"
        )

    console.print()
    show(response.text)
    console.print("\n[dim]Sources:[/dim]")
    for index, hit in enumerate(hits, start=1):
        console.print(f"  [dim]\\[{index}] {hit.meeting_date} ({hit.score:.2f}) "
                      f"{safe(hit.text[:90])}...[/dim]")


@app.command()
def ami(
    path: str = typer.Option("data/ami", help="Where the corpus was unpacked."),
    limit: int = typer.Option(5, help="Meetings to evaluate. Each costs quota."),
    threshold: float = typer.Option(55.0, help="Fuzzy match threshold."),
    verbose: bool = typer.Option(
        True, help="Show what matched, what was missed and what was extra."
    ),
    out: str = typer.Option("", help="Write the report to this JSON path."),
) -> None:
    """Evaluate extraction against the real AMI corpus.

    The corpus is not bundled - it needs a licence accepted by hand:

      1. Open https://groups.inf.ed.ac.uk/ami/download/
      2. Tick "manual annotations" (you do NOT need the audio or video -
         those are hundreds of gigabytes and nothing here uses them).
      3. Accept the licence and download ami_public_manual_1.6.2.zip
      4. Unzip it into data/ami/ - any nesting is fine, it will be found.

    Then: python -m quorum.cli ami --limit 5
    """
    import json

    from quorum.agents.embedding import LexicalEmbedder
    from quorum.agents.extractor import Extractor
    from quorum.agents.resolver import Resolver
    from quorum.agents.segmenter import Segmenter
    from quorum.agents.verifier import GroundingVerifier
    from quorum.data.ami import AmiCorpus
    from quorum.eval.ami_eval import score_meeting, summarise

    setup_logging("WARNING")
    ensure_dirs()

    try:
        corpus = AmiCorpus(Path(path))
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]\n")
        console.print("[bold]To get the corpus:[/bold]")
        console.print("  1. https://groups.inf.ed.ac.uk/ami/download/")
        console.print("  2. Tick [bold]manual annotations[/bold] "
                      "[dim](not audio/video - those are huge and unused)[/dim]")
        console.print("  3. Accept the licence, download ami_public_manual_1.6.2.zip")
        console.print(f"  4. Unzip into [bold]{path}/[/bold]")
        raise typer.Exit(1)

    console.print(f"Corpus at [dim]{corpus.root}[/dim]")
    meetings = corpus.load_all(limit=limit)
    if not meetings:
        console.print("[red]No meetings with ACTIONS annotations found.[/red] "
                      "Check that the abstractive/ directory was unpacked.")
        raise typer.Exit(1)

    segmenter = Segmenter(embedder=LexicalEmbedder())
    extractor, resolver, verifier = Extractor(), Resolver(), GroundingVerifier()
    results = []

    for meeting in meetings:
        with console.status(f"{meeting.meeting_id}..."):
            segments = segmenter.segment(meeting.transcript)
            extracted = extractor.extract(meeting.transcript, segments)
            commitments, _ = verifier.verify(extracted.commitments, meeting.transcript)
            resolver.resolve(commitments, meeting.transcript)
            result = score_meeting(meeting, commitments, threshold)
        results.append(result)
        console.print(
            f"  {meeting.meeting_id}: {len(meeting.transcript.utterances)} utterances, "
            f"{len(commitments)} extracted, {len(meeting.actions)} annotated, "
            f"P={result.scores.precision:.2f} R={result.scores.recall:.2f}"
        )
        # A bare 0.000 is uninterpretable, and this evaluation produces one
        # often. Seeing the annotator sentence next to what was extracted is
        # what turns "it scored badly" into "these two disagree about what an
        # action item is" - which is the actual finding, and not one a metric
        # can express.
        if verbose:
            for predicted, annotated, score in result.matched:
                console.print(f"      [green]hit {score:.0f}[/green]  "
                              f"{safe(predicted[:60])}  <->  {safe(annotated[:60])}")
            for missed in result.missed:
                console.print(f"      [red]missed[/red]   {safe(missed[:100])}")
            for spurious in result.spurious:
                console.print(f"      [yellow]extra[/yellow]    {safe(spurious[:100])}")

    summary = summarise(results)
    table = Table(title="AMI extraction (real transcripts)", header_style="bold")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for label, key in [("Precision", "precision"), ("Recall", "recall"), ("F1", "f1")]:
        table.add_row(label, f"{summary['commitments'][key]:.3f}")
    table.add_row("Meetings", str(summary["meetings"]))
    console.print(table)
    console.print(f"\n[yellow]Read with care:[/yellow] {summary['note']}")

    if out:
        Path(out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        console.print(f"[dim]Written to {out}[/dim]")


@app.command()
def demo(seed: int = 0, weeks: int = 6) -> None:
    """Run one synthetic project end to end and show what the agent would do."""
    from datetime import timedelta

    from quorum.agents.embedding import LexicalEmbedder
    from quorum.agents.extractor import Extractor
    from quorum.agents.resolver import Resolver
    from quorum.agents.segmenter import Segmenter
    from quorum.agents.verifier import GroundingVerifier
    from quorum.eval.harness import ManifestEvidenceProvider
    from quorum.execution import ApprovalGate, build_digests
    from quorum.models import MeetingRecord
    from quorum.synth import ProjectGenerator
    from quorum.tracking import ActionType, Ledger, Planner

    setup_logging("WARNING")
    transcripts, manifest = ProjectGenerator(seed=seed, weeks=weeks).generate()
    console.print(f"[bold]{manifest.project_id}[/bold] - {len(transcripts)} weekly meetings\n")

    segmenter = Segmenter(embedder=LexicalEmbedder())
    extractor, resolver, verifier = Extractor(), Resolver(), GroundingVerifier()
    ledger = Ledger(manifest.project_id)
    known = {k: __import__("datetime").date.fromisoformat(v)
             for k, v in manifest.known_events.items()}

    for transcript in transcripts:
        with console.status(f"Processing {transcript.title}..."):
            segments = segmenter.segment(transcript)
            extracted = extractor.extract(transcript, segments)
            commitments, report = verifier.verify(extracted.commitments, transcript)
            updates, _ = verifier.verify(extracted.status_updates, transcript)
            resolver.resolve(commitments, transcript, known)
            ledger.ingest(
                MeetingRecord(
                    meeting_id=transcript.meeting_id, project_id=transcript.project_id,
                    meeting_date=transcript.meeting_date, commitments=commitments,
                    status_updates=updates, decisions=[],
                )
            )
        console.print(
            f"  {transcript.meeting_date}  {transcript.title}: "
            f"+{len(commitments)} commitments, {len(updates)} status updates, "
            f"{report.rejected} rejected as ungrounded"
        )

    as_of = transcripts[-1].meeting_date + timedelta(days=7)
    plan = Planner().plan(ledger, as_of, ManifestEvidenceProvider(manifest, as_of))

    table = Table(title=f"\nWhat the agent would do on {as_of}", header_style="bold")
    table.add_column("Action")
    table.add_column("Commitment", overflow="fold")
    table.add_column("Why", overflow="fold")
    colour = {
        ActionType.NUDGE: "yellow", ActionType.ESCALATE: "red",
        ActionType.MARK_DONE: "green", ActionType.MARK_DROPPED: "magenta",
    }
    for item in plan.actions:
        commitment = ledger.by_id(item.commitment_id)
        style = colour.get(item.action, "dim")
        table.add_row(
            f"[{style}]{item.action.value}[/{style}]",
            safe(commitment.description if commitment else item.commitment_id),
            item.reason,
        )
    console.print(table)

    gate = ApprovalGate()
    digests = build_digests(plan.actions, ledger, as_of)
    for digest in digests:
        gate.propose(plan.actions[0], digest.subject, recipient=digest.recipient_email)

    console.print(f"\n[bold]{len(digests)} email digest(s) drafted[/bold] - "
                  f"[yellow]all awaiting approval, nothing sent[/yellow]")
    if digests:
        console.print(f"\n[dim]--- preview: {digests[0].recipient_email} ---[/dim]")
        show(digests[0].render(ledger, as_of))


if __name__ == "__main__":
    app()
