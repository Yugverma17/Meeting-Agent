"""OAuth for the Google APIs, kept as small as it can be.

Three decisions worth stating, because each is a place this could have been done
more conveniently and less safely.

**The narrowest scope that works.** `calendar.events` grants read/write on events
and nothing else - it cannot create or delete calendars, and cannot read the
user's contacts or mail. A tool that asks for `calendar` because it is one word
shorter is asking for permission it never uses. If the scope list ever grows,
the stored token is discarded and consent is asked for again rather than silently
upgraded, which is what `_scopes_match` is for.

**Import cost is paid only when used.** `googleapiclient` and its auth stack pull
in a large dependency tree, so every import here is function-local. The package
imports, and the whole test suite runs, on a machine that has never installed
them.

**Nothing here decides to write.** This module hands back an authenticated
service object; the decision to create or delete an event lives behind the
approval gate in `quorum.execution.calendar`. Keeping authentication separate
from authorisation means the gate cannot be bypassed by reaching for a client.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from quorum.config import DATA_DIR, get_settings

log = logging.getLogger(__name__)

CALENDAR_SCOPES = ["https://www.googleapis.com/auth/calendar.events"]
"""Read/write access to events only. Deliberately not `.../auth/calendar`."""

GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.compose"]
"""Create and manage drafts.

`gmail.compose` is the narrowest scope Google offers that can create a draft -
there is no drafts-only alternative, and it does technically also permit
sending. Nothing in this project calls `messages.send`; drafts are created and
left for the user to read and send themselves, which is the same rule the
approval gate enforces everywhere else. Worth knowing the scope is wider than
the use, because a reviewer should not have to take that on trust:
`grep -rn "messages().send" src/` returns nothing.

Not requested: `gmail.readonly`, `gmail.modify`, or anything touching the
inbox."""

IDENTITY_SCOPES = ["openid", "https://www.googleapis.com/auth/userinfo.email"]
"""Which account signed in - the address, and nothing else.

Not vanity. Drafts are created against `userId="me"`, so the signed-in account
*is* the mailbox they land in, and an app that will not tell you which account
that is asks you to take on trust the one thing worth checking. It also has to
be requested to be knowable: the status line read "authorised as unknown"
because the code asked Google for the address without ever asking permission to
have it.

`userinfo.email` grants the address. Not `userinfo.profile`, which would also
hand over a name and a photograph that nothing here displays."""

ALL_SCOPES = [*IDENTITY_SCOPES, *CALENDAR_SCOPES, *GMAIL_SCOPES]
"""Asked for together, so consent happens once rather than three times."""

TOKEN_PATH = DATA_DIR / "token.json"
"""Where the refresh token lands. Gitignored by filename, at any depth."""


class GoogleAuthError(RuntimeError):
    """Raised with an actionable message - what to do, not just what failed."""


@dataclass
class CredentialsStatus:
    """What `doctor` and the CLI need to know without triggering a login."""

    secrets_present: bool
    token_present: bool
    scopes: list[str]
    account: str = ""
    expired: bool = False
    libraries_installed: bool = True

    @property
    def ready(self) -> bool:
        return self.libraries_installed and self.secrets_present and self.token_present

    @property
    def message(self) -> str:
        if not self.libraries_installed:
            return "google-api-python-client is not installed"
        if not self.secrets_present:
            return "no OAuth client secrets file - see README"
        if not self.token_present:
            return "not connected yet"
        if self.expired:
            return f"authorised as {self.account or 'unknown'} (token will refresh on use)"
        return f"authorised as {self.account or 'unknown'}"


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


def credentials_status(
    secrets_file: str | Path | None = None, token_path: Path | None = None
) -> CredentialsStatus:
    """Inspect local state. Never opens a browser and never hits the network."""
    settings = get_settings()
    secrets = Path(secrets_file or settings.google_client_secrets_file)
    token = Path(token_path or TOKEN_PATH)

    try:
        import google.oauth2.credentials  # noqa: F401
        import googleapiclient  # noqa: F401
    except ImportError:
        return CredentialsStatus(
            secrets_present=secrets.exists(), token_present=token.exists(),
            scopes=[], libraries_installed=False,
        )

    if not token.exists():
        return CredentialsStatus(
            secrets_present=secrets.exists(), token_present=False, scopes=[]
        )

    try:
        stored = json.loads(token.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        # A corrupt token file is reported as "not authorised" rather than
        # crashing a status command - re-running auth fixes it.
        return CredentialsStatus(
            secrets_present=secrets.exists(), token_present=False, scopes=[]
        )

    return CredentialsStatus(
        secrets_present=secrets.exists(),
        token_present=True,
        scopes=stored.get("scopes", []),
        account=stored.get("account", ""),
    )


# ---------------------------------------------------------------------------
# Authorising
# ---------------------------------------------------------------------------


def _scopes_match(stored: list[str], required: list[str]) -> bool:
    """A token is only reusable if it already covers everything we need.

    Google returns whatever the user actually consented to, which can be less
    than what was asked for. Treating a partial grant as success produces a
    403 at the first write instead of a login prompt now.
    """
    return set(required).issubset(set(stored or []))


def _load_credentials(token: Path, scopes: list[str]):
    from google.oauth2.credentials import Credentials

    if not token.exists():
        return None
    try:
        stored = json.loads(token.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Stored Google token unreadable (%s); re-authorising", exc)
        return None

    if not _scopes_match(stored.get("scopes", []), scopes):
        log.info("Stored token is missing a required scope; re-authorising")
        return None

    try:
        return Credentials.from_authorized_user_info(stored, scopes)
    except ValueError as exc:
        log.warning("Stored Google token invalid (%s); re-authorising", exc)
        return None


def _save_credentials(creds, token: Path, account: str = "") -> None:
    token.parent.mkdir(parents=True, exist_ok=True)
    payload = json.loads(creds.to_json())
    if account:
        payload["account"] = account
    token.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    try:
        token.chmod(0o600)
    except OSError:  # pragma: no cover - Windows ACLs, non-fatal
        pass


def authorise(
    scopes: list[str] | None = None,
    secrets_file: str | Path | None = None,
    token_path: Path | None = None,
    interactive: bool = True,
    timeout_seconds: int = 300,
):
    """Return usable credentials, refreshing or prompting for consent as needed.

    `interactive=False` is what every automated path uses: it will refresh an
    expired token silently, but it will never open a browser. A daily sweep that
    blocks on a login window is a daily sweep that stops running.
    """
    try:
        from google.auth.transport.requests import Request
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise GoogleAuthError(
            "Google API libraries are not installed. Run:\n"
            "  pip install google-api-python-client google-auth-oauthlib google-auth-httplib2"
        ) from exc

    settings = get_settings()
    scopes = scopes or ALL_SCOPES
    secrets = Path(secrets_file or settings.google_client_secrets_file)
    token = Path(token_path or TOKEN_PATH)

    creds = _load_credentials(token, scopes)

    if creds and creds.valid:
        return creds

    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            _save_credentials(creds, token)
            return creds
        except Exception as exc:  # noqa: BLE001 - refresh fails in many ways
            log.warning("Token refresh failed (%s); falling back to consent", exc)

    if not interactive:
        raise GoogleAuthError(
            "Google authorisation needed and this path cannot prompt. "
            "Run `quorum auth google` once, then retry."
        )

    if not secrets.exists():
        raise GoogleAuthError(
            f"OAuth client secrets not found at {secrets}.\n"
            "Create a Desktop OAuth client at https://console.cloud.google.com/apis/credentials, "
            "enable the Google Calendar API, download the JSON, and either save it as "
            f"{secrets} or set GOOGLE_CLIENT_SECRETS_FILE."
        )

    # Requesting `openid` makes Google return a slightly different scope set
    # than was asked for, and oauthlib treats any difference as tampering and
    # raises. The grant is a superset, `_scopes_match` checks that properly, and
    # this is the documented way to stop the library refusing it.
    os.environ.setdefault("OAUTHLIB_RELAX_TOKEN_SCOPE", "1")

    flow = InstalledAppFlow.from_client_secrets_file(str(secrets), scopes)
    # port=0 lets the OS pick a free loopback port; a hardcoded one collides
    # with whatever else is running and fails with an unhelpful browser error.
    #
    # The timeout matters when this is called from the interface rather than a
    # terminal: without it, closing the consent tab leaves the page waiting for
    # a redirect that will never arrive, with no way to cancel.
    creds = flow.run_local_server(
        port=0, prompt="consent", timeout_seconds=timeout_seconds,
        success_message=(
            "Quorum is connected. You can close this tab and go back to the app."
        ),
    )
    if creds is None:
        raise GoogleAuthError(
            "Sign-in was not completed - the consent window timed out or was closed."
        )
    _save_credentials(creds, token, account=_account_email(creds))
    return creds


def _account_email(creds) -> str:
    """Best-effort account label for the status line. Never fatal."""
    try:
        from googleapiclient.discovery import build

        service = build("oauth2", "v2", credentials=creds, cache_discovery=False)
        return service.userinfo().get().execute().get("email", "")
    except Exception:  # noqa: BLE001 - purely cosmetic
        return ""


def get_gmail_service(interactive: bool = False, **kwargs):
    """An authenticated Gmail v1 client, for creating drafts and nothing else."""
    from googleapiclient.discovery import build

    creds = authorise(scopes=ALL_SCOPES, interactive=interactive, **kwargs)
    return build("gmail", "v1", credentials=creds, cache_discovery=False)


def get_calendar_service(interactive: bool = False, **kwargs):
    """An authenticated Calendar v3 client.

    `cache_discovery=False` because the default file cache warns noisily under
    every modern Python and buys nothing for a CLI that runs for two seconds.
    """
    from googleapiclient.discovery import build

    creds = authorise(scopes=ALL_SCOPES, interactive=interactive, **kwargs)
    return build("calendar", "v3", credentials=creds, cache_discovery=False)


def revoke(token_path: Path | None = None) -> bool:
    """Forget the stored token. The grant itself is revoked in Google's UI."""
    token = Path(token_path or TOKEN_PATH)
    if not token.exists():
        return False
    token.unlink()
    return True
