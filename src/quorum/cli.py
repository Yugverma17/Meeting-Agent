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


@app.command()
def record(
    minutes: float = typer.Option(0.0, help="Stop after N minutes. 0 = until Ctrl+C."),
    roster: str = typer.Option(
        "", help='Participants: "Priya:priya@x.com,Sam:sam@x.com". Enables naming.'
    ),
    me: str = typer.Option("You", help="Your display name."),
    my_email: str = typer.Option("", help="Your email."),
    title: str = typer.Option("Live meeting", help="Meeting title."),
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

    transcript = build_transcript(segments, people, meeting_date=_date.today(), title=title)
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
