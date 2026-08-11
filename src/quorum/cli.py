"""Command line interface.

    python -m quorum.cli doctor          # is my environment sane?
    python -m quorum.cli models          # what can I call, and what are the limits?
    python -m quorum.cli models --probe  # do my API keys actually work? (uses quota)
    python -m quorum.cli quota           # how much of today's budget is left?
    python -m quorum.cli cache --stats
"""

from __future__ import annotations

import sys

import typer
from rich.console import Console
from rich.table import Table

from quorum import __version__
from quorum.config import (
    CACHE_DIR,
    DATA_DIR,
    LLM_CACHE_DIR,
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


if __name__ == "__main__":
    app()
