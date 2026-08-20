"""Outside services the agent reads from and writes to.

Everything here touches an account that belongs to a real person, so two rules
hold throughout: ask for the narrowest scope that does the job, and never write
without passing the approval gate first.
"""

from quorum.integrations.google_auth import (
    ALL_SCOPES,
    CALENDAR_SCOPES,
    GMAIL_SCOPES,
    GoogleAuthError,
    authorise,
    credentials_status,
    get_calendar_service,
    get_gmail_service,
    revoke,
)

__all__ = [
    "ALL_SCOPES",
    "CALENDAR_SCOPES",
    "GMAIL_SCOPES",
    "GoogleAuthError",
    "authorise",
    "credentials_status",
    "get_calendar_service",
    "get_gmail_service",
    "revoke",
]
