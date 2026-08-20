"""Central configuration.

Settings load from environment / .env. Every API key is optional at import time so
the package can be imported (and most tests run) with no credentials present.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
AMI_DIR = DATA_DIR / "ami"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
CACHE_DIR = PROJECT_ROOT / ".cache"
LLM_CACHE_DIR = CACHE_DIR / "llm"
RUNS_DIR = PROJECT_ROOT / "runs"


class Settings(BaseSettings):
    """Runtime configuration, populated from the environment or a .env file."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Providers -------------------------------------------------------
    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    groq_api_key: str | None = Field(default=None, alias="GROQ_API_KEY")

    # --- Tracing ---------------------------------------------------------
    langsmith_api_key: str | None = Field(default=None, alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field(default="quorum", alias="LANGSMITH_PROJECT")
    langsmith_endpoint: str = Field(
        default="https://api.smith.langchain.com", alias="LANGSMITH_ENDPOINT"
    )
    langsmith_tracing: bool = Field(default=True, alias="LANGSMITH_TRACING")
    """Whether to trace when a key is present. Defaults on, because tracing you
    have to remember to enable is tracing you do not have when it would have
    helped. With no key set this is inert - nothing is uploaded and nothing is
    slowed down."""

    # --- Reality-verification layer --------------------------------------
    github_token: str | None = Field(default=None, alias="GITHUB_TOKEN")
    google_client_secrets_file: str = Field(
        default="credentials.json", alias="GOOGLE_CLIENT_SECRETS_FILE"
    )
    calendar_id: str = Field(default="primary", alias="QUORUM_CALENDAR_ID")
    """Which calendar deadlines are written to. A dedicated secondary calendar
    is worth creating - it can be toggled off in one click without losing the
    events, which a mixed-in personal calendar cannot."""

    reminder_lead_days: str = Field(default="3,1", alias="QUORUM_REMINDER_DAYS")
    """Comma-separated lead times for deadline reminders."""

    reminder_hour: int = Field(default=9, alias="QUORUM_REMINDER_HOUR")

    # --- Behaviour -------------------------------------------------------
    require_approval: bool = Field(default=True, alias="QUORUM_REQUIRE_APPROVAL")
    """Hard gate on outbound side effects. There is no code path that sends
    email or writes calendar events with this set to False in production use;
    it exists only so the eval harness can score proposed actions offline."""

    cache_enabled: bool = Field(default=True, alias="QUORUM_CACHE_ENABLED")
    log_level: str = Field(default="INFO", alias="QUORUM_LOG_LEVEL")

    def reminder_days(self) -> tuple[int, ...]:
        """Parsed lead times, most distant first. Malformed entries are dropped
        rather than crashing a sync - a typo in an env var should cost you a
        reminder, not the run."""
        days = []
        for part in self.reminder_lead_days.split(","):
            try:
                value = int(part.strip())
            except ValueError:
                continue
            if value > 0:
                days.append(value)
        return tuple(sorted(set(days), reverse=True)) or (3, 1)

    def configured_providers(self) -> list[str]:
        """Which providers actually have credentials right now."""
        present = []
        if self.gemini_api_key:
            present.append("gemini")
        if self.groq_api_key:
            present.append("groq")
        return present


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]


def free_path(directory: Path, stem: str, suffix: str) -> Path:
    """A path that does not already exist, suffixing `-2`, `-3` as needed.

    For any filename built from data a person could repeat - a date plus a
    recipient, a date plus a title. Silently overwriting a file the user cannot
    see is the failure mode that has already cost this project a recorded
    lecture, and it never announces itself.
    """
    candidate = directory / f"{stem}{suffix}"
    if not candidate.exists():
        return candidate
    for index in range(2, 1000):
        candidate = directory / f"{stem}-{index}{suffix}"
        if not candidate.exists():
            return candidate
    return directory / f"{stem}-overflow{suffix}"  # pragma: no cover


def ensure_dirs() -> None:
    """Create the directories we write to. Safe to call repeatedly."""
    for path in (DATA_DIR, RAW_DIR, AMI_DIR, SYNTHETIC_DIR, CACHE_DIR, LLM_CACHE_DIR, RUNS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def setup_logging(level: str | None = None) -> None:
    logging.basicConfig(
        level=level or get_settings().log_level,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
