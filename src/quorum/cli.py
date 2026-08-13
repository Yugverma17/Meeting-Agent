"""Command line interface.

    python -m quorum.cli doctor          # is my environment sane?
    python -m quorum.cli models          # what can I call, and what are the limits?
    python -m quorum.cli models --probe  # do my API keys actually work? (uses quota)
    python -m quorum.cli quota           # how much of today's budget is left?
    python -m quorum.cli cache --stats
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from quorum import __version__
from quorum.config import (
    CACHE_DIR,
    DATA_DIR,
    LLM_CACHE_DIR,
    RUNS_DIR,
    ensure_dirs,
    get_settings,
    setup_logging,
)
from quorum.llm.cache import LLMCache
from quorum.llm.providers import WHISPER_MODEL, registry
from quorum.llm.ratelimit import QuotaTracker
from quorum.llm.router import Router

app = typer.Typer(add_completion=False, help="Quorum - meeting commitment agent.")
console = Console()


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
    table.add_row(
        "Langfuse", ok if settings.langfuse_public_key else warn,
        "set" if settings.langfuse_public_key else "optional until tracing phase",
    )
    table.add_row(
        "GitHub token", ok if settings.github_token else warn,
        "set" if settings.github_token else "optional until reality-verification phase",
    )

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
        console.print(f"[red]No project {name!r}.[/red] "
                      "Run [bold]quorum project[/bold] to list them.")
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
            item.assignee.display_name or "[dim]unassigned[/dim]",
            item.description,
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
            item.description if item else action.commitment_id,
            action.reason,
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
        path = drafts_dir / f"{now.isoformat()}_{digest.recipient_email.replace('@', '_at_')}.txt"
        path.write_text(f"To: {digest.recipient_email}\nSubject: {digest.subject}\n\n{body}",
                        encoding="utf-8")
        console.print(f"[bold]To:[/bold] {digest.recipient_email}")
        console.print(f"[bold]Subject:[/bold] {digest.subject}")
        console.print(body)
        console.print(f"[dim]saved to {path}[/dim]\n")

    console.print(
        "[yellow]Review the drafts, then send them yourself.[/yellow] "
        "Automatic sending needs Gmail OAuth, which is not wired up."
    )


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
            console.print(f"  - {item.description}")
        raise typer.Exit(1)

    scored = sorted(
        ((fuzz.token_set_ratio(what.lower(), c.description.lower()), c) for c in candidates),
        key=lambda pair: -pair[0],
    )
    best_score, best = scored[0]
    if best_score < 55:
        console.print(f"[red]Nothing matches {what!r}.[/red] Open commitments:")
        for _, item in scored[:5]:
            console.print(f"  - {item.description}")
        raise typer.Exit(1)

    # Ambiguity is resolved by asking, not by picking. Closing the wrong
    # commitment is silent and nothing later reopens it.
    if len(scored) > 1 and scored[1][0] >= best_score - 5:
        console.print(f"[yellow]Ambiguous - {what!r} matches several:[/yellow]")
        for score, item in scored[:3]:
            console.print(f"  - {item.description} [dim]({score:.0f})[/dim]")
        console.print("Be more specific.")
        raise typer.Exit(1)

    best.status = CommitmentStatus.DROPPED if drop else CommitmentStatus.VERIFIED_DONE
    best.resolution_note = f"marked {'dropped' if drop else 'done'} by you on {_date.today()}"
    found.save_ledger()

    verb = "Dropped" if drop else "Closed"
    console.print(f"[green]{verb}:[/green] {best.description}")
    console.print(f"[dim]{len(ledger.open_commitments())} still open.[/dim]")


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

    from quorum.agents.embedding import LexicalEmbedder
    from quorum.agents.extractor import Extractor
    from quorum.agents.resolver import Resolver
    from quorum.agents.segmenter import Segmenter
    from quorum.agents.verifier import GroundingVerifier
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

    deadline = _time.time() + minutes * 60 if minutes else None
    console.print("[green]Recording.[/green] Press Ctrl+C to stop.\n")
    try:
        while deadline is None or _time.time() < deadline:
            _time.sleep(1.0)
            captured = recorder.captured_seconds
            console.print(
                f"  captured {captured / 60:.1f} min, "
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
    console.print(transcript.as_dialogue(0, min(8, len(transcript.utterances))))
    if len(transcript.utterances) > 8:
        console.print(f"[dim]... and {len(transcript.utterances) - 8} more[/dim]")

    with console.status("Extracting commitments..."):
        segmented = Segmenter(embedder=LexicalEmbedder()).segment(transcript)
        extracted = Extractor().extract(transcript, segmented)
        commitments, report = GroundingVerifier().verify(extracted.commitments, transcript)
        Resolver().resolve(commitments, transcript)

    table = Table(title="\nCommitments found", header_style="bold")
    table.add_column("Owner")
    table.add_column("Commitment", overflow="fold")
    table.add_column("Due")
    table.add_column("Strength")
    for commitment in commitments:
        table.add_row(
            commitment.assignee.display_name or f"[dim]{commitment.assignee.raw_mention}[/dim]",
            commitment.description,
            commitment.deadline.resolved.isoformat() if commitment.deadline.resolved else "-",
            commitment.strength.value,
        )
    console.print(table)
    console.print(
        f"[dim]{report.rejected} item(s) rejected as ungrounded. "
        f"Nothing was sent - the approval gate is enabled.[/dim]"
    )

    if active_project is None:
        console.print(
            "\n[dim]Not saved. Use [bold]--project <name>[/bold] to build history "
            "across meetings and enable chasing.[/dim]"
        )
        return

    from quorum.memory import ProjectMemory
    from quorum.models import MeetingRecord

    updates, _ = GroundingVerifier().verify(extracted.status_updates, transcript)
    decisions, _ = GroundingVerifier().verify(extracted.decisions, transcript)
    meeting_record = MeetingRecord(
        meeting_id=transcript.meeting_id, project_id=active_project.meta.id,
        meeting_date=transcript.meeting_date, title=title,
        commitments=commitments, decisions=decisions, status_updates=updates,
        rejected_items=report.rejected,
    )

    before = len(active_project.ledger.open_commitments())
    active_project.add_meeting(meeting_record, transcript)
    workspace_ref, _ = _load_project(active_project.meta.id)
    workspace_ref.save(active_project)

    indexed = 0
    try:
        indexed = ProjectMemory(active_project.memory_dir).index_meeting(
            meeting_record, transcript
        )
    except Exception as exc:  # noqa: BLE001 - memory is an optimisation, never a blocker
        console.print(f"[dim]Could not index into project memory: {exc}[/dim]")

    after = len(active_project.ledger.open_commitments())
    console.print(
        f"\n[green]Saved to {active_project.meta.name}.[/green] "
        f"Open commitments: {before} -> {after}. "
        f"{len(updates)} status update(s) applied, {indexed} item(s) indexed."
    )
    console.print(f"  Next: [bold]quorum today --project {active_project.meta.id}[/bold]")


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

    console.print()
    console.print(notes.as_markdown())
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

    if active_project is not None:
        from quorum.memory import ProjectMemory

        indexed = ProjectMemory(active_project.memory_dir).index_notes(
            transcript.meeting_id, notes.title, _date.today(), notes
        )
        console.print(
            f"[green]Indexed {indexed} item(s) into {active_project.meta.name}.[/green] "
            f"Ask questions with: [bold]quorum ask \"...\" "
            f"--project {active_project.meta.id}[/bold]"
        )


@app.command()
def learn(
    minutes: float = typer.Option(0.0, help="Stop after N minutes. 0 = until Ctrl+C."),
    title: str = typer.Option("", help="What you are watching."),
    project_name: str = typer.Option(
        "", "--project", help="Save notes here so `quorum ask` can search them."
    ),
    system_only: bool = typer.Option(
        True, help="Capture only the video's audio, not your microphone."
    ),
    from_transcript: str = typer.Option(
        "", help="Re-analyse a saved transcript instead of recording again."
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

    console.print("[bold]Listening.[/bold] Start the video now if you have not.")
    console.print("[dim]Press Ctrl+C when the lecture ends.[/dim]\n")

    recorder = DualRecorder(RecorderConfig())
    try:
        recorder.start(announced=True)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Could not start recording: {exc}[/red]")
        raise typer.Exit(1)

    deadline = _time.time() + minutes * 60 if minutes else None
    try:
        while deadline is None or _time.time() < deadline:
            _time.sleep(1.0)
            console.print(
                f"  listening {recorder.captured_seconds / 60:.1f} min, "
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
        console.print("[red]No audio captured.[/red] Was the video playing?")
        raise typer.Exit(1)

    transcriber = WhisperTranscriber()
    with console.status(f"Transcribing {len(chunks)} chunk(s)..."):
        segments = transcriber.transcribe_all(chunks)
    if not system_only:
        segments, _ = suppress_echo(segments)
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
    which: str = typer.Argument("", help="Meeting id or part of its title. Omit to list."),
    project_name: str = typer.Option("", "--project", help="Which project."),
    file: str = typer.Option("", help="Read a transcript JSON directly instead."),
    style: str = typer.Option(
        "speakers", help="speakers | timestamped | plain | markdown | srt"
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
            table.add_row(item.meeting_id, item.title or "-",
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

    try:
        text = render(
            chosen, Style(style), speaker=speaker or None,
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

    console.print(f"\n{response.text}\n")
    console.print("[dim]Sources:[/dim]")
    for index, hit in enumerate(hits, start=1):
        console.print(f"  [dim][{index}] {hit.meeting_date} ({hit.score:.2f}) "
                      f"{hit.text[:90]}...[/dim]")


@app.command()
def ami(
    path: str = typer.Option("data/ami", help="Where the corpus was unpacked."),
    limit: int = typer.Option(5, help="Meetings to evaluate. Each costs quota."),
    threshold: float = typer.Option(55.0, help="Fuzzy match threshold."),
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
            f"{len(meeting.actions)} annotated actions, "
            f"P={result.scores.precision:.2f} R={result.scores.recall:.2f}"
        )

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
            commitment.description if commitment else item.commitment_id,
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
        console.print(digests[0].render(ledger, as_of))


if __name__ == "__main__":
    app()
