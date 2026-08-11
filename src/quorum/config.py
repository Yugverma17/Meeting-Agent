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
    langfuse_public_key: str | None = Field(default=None, alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str | None = Field(default=None, alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str = Field(default="https://cloud.langfuse.com", alias="LANGFUSE_HOST")

    # --- Reality-verification layer --------------------------------------
    github_token: str | None = Field(default=None, alias="GITHUB_TOKEN")
    google_client_secrets_file: str = Field(
        default="credentials.json", alias="GOOGLE_CLIENT_SECRETS_FILE"
    )

    # --- Behaviour -------------------------------------------------------
    require_approval: bool = Field(default=True, alias="QUORUM_REQUIRE_APPROVAL")
    """Hard gate on outbound side effects. There is no code path that sends
    email or writes calendar events with this set to False in production use;
    it exists only so the eval harness can score proposed actions offline."""

    cache_enabled: bool = Field(default=True, alias="QUORUM_CACHE_ENABLED")
    log_level: str = Field(default="INFO", alias="QUORUM_LOG_LEVEL")

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
