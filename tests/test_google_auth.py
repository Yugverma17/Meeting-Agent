"""Connecting a Google account.

The rules here are about what is asked for and what is stored, both of which are
things a user has to be able to check rather than trust.
"""

from __future__ import annotations

import json

import pytest

from quorum.integrations import google_auth
from quorum.integrations.google_auth import (
    ALL_SCOPES,
    CALENDAR_SCOPES,
    GMAIL_SCOPES,
    IDENTITY_SCOPES,
    CredentialsStatus,
    _scopes_match,
    credentials_status,
    revoke,
)


# --- what is asked for --------------------------------------------------------


def test_the_account_address_is_requested_so_it_can_be_shown():
    """Drafts are created against userId="me", so the signed-in account *is* the
    mailbox they land in. The status line used to read "authorised as unknown"
    because the code asked Google for the address without asking permission for
    it."""
    assert "https://www.googleapis.com/auth/userinfo.email" in ALL_SCOPES
    assert "openid" in ALL_SCOPES


def test_no_scope_touches_the_inbox_or_the_profile():
    """gmail.compose is the narrowest scope that can create a draft. Nothing
    here should be able to read mail, or collect a name and a photograph."""
    forbidden = ("gmail.readonly", "gmail.modify", "gmail.metadata",
                 "userinfo.profile", "auth/calendar\"", "contacts")

    joined = " ".join(ALL_SCOPES)
    for scope in forbidden:
        assert scope not in joined


def test_calendar_cannot_create_or_delete_calendars():
    assert CALENDAR_SCOPES == ["https://www.googleapis.com/auth/calendar.events"]


def test_gmail_is_compose_only():
    assert GMAIL_SCOPES == ["https://www.googleapis.com/auth/gmail.compose"]


def test_everything_is_asked_for_at_once():
    """Three separate consent windows for one app is three chances to abandon
    it halfway and end up half-connected."""
    for group in (IDENTITY_SCOPES, CALENDAR_SCOPES, GMAIL_SCOPES):
        assert set(group).issubset(set(ALL_SCOPES))


# --- reusing a stored grant ---------------------------------------------------


def test_a_grant_covering_everything_is_reusable():
    assert _scopes_match([*ALL_SCOPES, "https://www.googleapis.com/auth/extra"], ALL_SCOPES)


def test_a_partial_grant_is_not_treated_as_success():
    """Google returns what the user actually consented to, which can be less
    than was asked for. Accepting that produces a 403 at the first write instead
    of a sign-in prompt now."""
    assert not _scopes_match(CALENDAR_SCOPES, ALL_SCOPES)


def test_no_stored_scopes_is_not_a_match():
    assert not _scopes_match([], ALL_SCOPES)


# --- what the interface reads -------------------------------------------------


def test_a_missing_token_reports_not_connected(tmp_path, monkeypatch):
    monkeypatch.setattr(google_auth, "TOKEN_PATH", tmp_path / "absent.json")
    status = credentials_status(secrets_file=tmp_path / "credentials.json")

    assert not status.ready
    assert "not connected" in status.message or "client secrets" in status.message


def test_a_corrupt_token_reads_as_not_connected_rather_than_crashing(tmp_path):
    """A status line is displayed on every page load. It must never be the
    thing that takes the app down."""
    secrets = tmp_path / "credentials.json"
    secrets.write_text("{}", encoding="utf-8")
    token = tmp_path / "token.json"
    token.write_text("{ this is not json", encoding="utf-8")

    status = credentials_status(secrets_file=secrets, token_path=token)

    assert not status.ready


def test_the_signed_in_address_is_surfaced(tmp_path):
    secrets = tmp_path / "credentials.json"
    secrets.write_text("{}", encoding="utf-8")
    token = tmp_path / "token.json"
    token.write_text(json.dumps({
        "token": "x", "refresh_token": "y", "scopes": ALL_SCOPES,
        "account": "yug@example.com",
    }), encoding="utf-8")

    status = credentials_status(secrets_file=secrets, token_path=token)

    assert status.ready
    assert "yug@example.com" in status.message


def test_disconnecting_removes_the_stored_token(tmp_path):
    token = tmp_path / "token.json"
    token.write_text("{}", encoding="utf-8")

    assert revoke(token_path=token)
    assert not token.exists()
    assert not revoke(token_path=token), "a second disconnect is harmless"


# --- refusing to prompt where it cannot ---------------------------------------


def test_an_automated_path_never_opens_a_browser(tmp_path, monkeypatch):
    """A daily sweep that blocks on a login window is a daily sweep that stops
    running."""
    from quorum.integrations import GoogleAuthError

    monkeypatch.setattr(google_auth, "TOKEN_PATH", tmp_path / "absent.json")

    with pytest.raises(GoogleAuthError, match="cannot prompt"):
        google_auth.authorise(
            interactive=False, token_path=tmp_path / "absent.json",
            secrets_file=tmp_path / "credentials.json",
        )


def test_status_is_a_pure_read(tmp_path, monkeypatch):
    """It runs on every page render, so it must not hit the network."""
    def explode(*args, **kwargs):
        raise AssertionError("credentials_status must not authorise")

    monkeypatch.setattr(google_auth, "authorise", explode)
    credentials_status(secrets_file=tmp_path / "nothing.json",
                       token_path=tmp_path / "nothing.json")


# --- finding the file Google actually gave you --------------------------------


def test_the_configured_path_wins_when_it_exists(tmp_path, monkeypatch):
    from quorum.integrations.google_auth import find_client_secrets

    chosen = tmp_path / "credentials.json"
    chosen.write_text("{}", encoding="utf-8")
    (tmp_path / "client_secret_123.apps.googleusercontent.com.json").write_text(
        "{}", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)

    assert find_client_secrets(chosen) == chosen


def test_googles_own_download_name_is_found_without_renaming(tmp_path, monkeypatch):
    """Every guide says "rename it to credentials.json". That rename has no
    purpose and silently breaks the feature when skipped - the app reports "not
    set up yet" while the file sits in the folder."""
    from quorum.integrations.google_auth import find_client_secrets

    downloaded = tmp_path / "client_secret_44077-6iq1.apps.googleusercontent.com.json"
    downloaded.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(google_auth, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(google_auth, "DATA_DIR", tmp_path)
    monkeypatch.chdir(tmp_path)

    assert find_client_secrets(tmp_path / "credentials.json") == downloaded


def test_the_newest_client_wins(tmp_path, monkeypatch):
    """Someone who regenerated a client wants the one they just downloaded."""
    import os
    import time

    from quorum.integrations.google_auth import find_client_secrets

    old = tmp_path / "client_secret_old.apps.googleusercontent.com.json"
    new = tmp_path / "client_secret_new.apps.googleusercontent.com.json"
    old.write_text("{}", encoding="utf-8")
    new.write_text("{}", encoding="utf-8")
    stale = time.time() - 10_000
    os.utime(old, (stale, stale))
    monkeypatch.setattr(google_auth, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(google_auth, "DATA_DIR", tmp_path)
    monkeypatch.chdir(tmp_path)

    assert find_client_secrets(tmp_path / "credentials.json") == new


def test_nothing_found_returns_the_configured_path_for_the_error_message(tmp_path, monkeypatch):
    """Isolated from the real project folder deliberately. The search also looks
    in PROJECT_ROOT so the file is found whatever directory the app is launched
    from - which meant this test picked up the developer's own client."""
    from quorum.integrations.google_auth import find_client_secrets

    empty = tmp_path / "elsewhere"
    empty.mkdir()
    monkeypatch.setattr(google_auth, "PROJECT_ROOT", empty)
    monkeypatch.setattr(google_auth, "DATA_DIR", empty)
    monkeypatch.chdir(tmp_path)
    wanted = tmp_path / "credentials.json"

    assert find_client_secrets(wanted) == wanted


def test_client_secret_files_are_gitignored():
    """The file holds a client secret. It must never be committable, whatever
    Google named it."""
    import pathlib

    ignored = pathlib.Path(".gitignore").read_text(encoding="utf-8")

    assert "client_secret*.json" in ignored
    assert "credentials.json" in ignored
    assert "token.json" in ignored
