"""GitHub as a source of truth about whether work happened.

This is the layer that separates Quorum from a meeting summariser. Every other
tool in the category asks the next meeting whether the spec got sent. This one
goes and looks.

It matters most for the case a transcript can never settle: work that was
delivered and then never mentioned again. Conversation says nothing, so an agent
relying on conversation alone nags someone who finished a week ago - the single
most irritating failure a tool like this can have.

Read-only by design. The token needs no write scope, and nothing here mutates
anything on GitHub.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import httpx

from quorum.models import Commitment
from quorum.tracking.planner import DeliveryEvidence

log = logging.getLogger(__name__)

API_ROOT = "https://api.github.com"

# Words that carry no signal about *which* work is meant. Searching for them
# matches everything and ranks nothing.
STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "for", "to", "in", "on", "with", "by",
    "our", "my", "his", "her", "their", "its", "this", "that", "these", "those",
    "work", "task", "item", "thing", "stuff", "update", "change", "changes",
    "fix", "add", "make", "do", "get", "some", "new", "up",
    # Generic engineering nouns: real in the description, useless as a search
    # term because they appear in a large fraction of any repo's pull requests.
    "job", "jobs", "run", "runs", "piece", "bit", "part", "code", "stuff",
}

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")


def keywords_for(description: str, limit: int = 4) -> list[str]:
    """Salient search terms from a commitment description.

    Longest words first: "reconciliation" discriminates between pull requests,
    "job" does not.
    """
    words = [w.lower() for w in _WORD.findall(description)]
    salient = [w for w in words if w not in STOPWORDS]
    return sorted(dict.fromkeys(salient), key=len, reverse=True)[:limit]


@dataclass
class GitHubConfig:
    repo: str | None = None
    """"owner/name". If unset, searches across everything the token can see."""

    lookback_days: int = 45
    min_keyword_hits: int = 2
    """How many of the commitment's keywords a PR title must contain. One word
    in common is a coincidence; two is a signal."""

    timeout_s: float = 10.0


class GitHubEvidenceProvider:
    """Implements the same `EvidenceProvider` protocol the planner already uses.

    That interface is why the evaluation harness can substitute a manifest-backed
    provider and score the *decision* logic without a network, while production
    swaps in this one with no change to the planner at all.
    """

    def __init__(
        self,
        token: str | None,
        config: GitHubConfig | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.token = token
        self.config = config or GitHubConfig()
        self._client = client
        self.lookups = 0
        self.hits = 0

    @property
    def available(self) -> bool:
        return bool(self.token)

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                timeout=self.config.timeout_s,
                headers={
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                    "Authorization": f"Bearer {self.token}",
                },
            )
        return self._client

    # -- the protocol ------------------------------------------------------

    def find_evidence(self, commitment: Commitment) -> DeliveryEvidence | None:
        if not self.available:
            return None

        keywords = keywords_for(commitment.description)
        if len(keywords) < self.config.min_keyword_hits:
            # Too vague to search for. Returning nothing is correct: a loose
            # match here would wrongly close a real, still-open commitment.
            log.debug("Not enough salient keywords in %r", commitment.description)
            return None

        since = self._since(commitment)
        try:
            items = self._search(keywords, commitment, since)
        except (httpx.HTTPError, ValueError) as exc:
            log.warning("GitHub search failed (%s); treating as no evidence", exc)
            return None

        self.lookups += 1
        best = self._best_match(items, keywords)
        if best is None:
            return None

        self.hits += 1
        closed_at = best.get("closed_at") or best.get("updated_at")
        found_on = (
            datetime.fromisoformat(closed_at.replace("Z", "+00:00")).date()
            if closed_at
            else date.today()
        )
        return DeliveryEvidence(
            source="github",
            reference=f"PR #{best['number']}",
            found_on=found_on,
            confidence=0.9,
            detail=best.get("title", ""),
        )

    # -- internals ---------------------------------------------------------

    def _since(self, commitment: Commitment) -> date:
        anchor = commitment.created_on or date.today() - timedelta(
            days=self.config.lookback_days
        )
        return anchor - timedelta(days=1)

    def _search(self, keywords: list[str], commitment: Commitment, since: date) -> list[dict]:
        """Closed PRs mentioning the keywords, optionally by the owner."""
        query = [" ".join(keywords), "is:pr", "is:closed", f"closed:>={since.isoformat()}"]
        if self.config.repo:
            query.append(f"repo:{self.config.repo}")
        if commitment.assignee.github_login:
            query.append(f"author:{commitment.assignee.github_login}")

        response = self.client.get(
            f"{API_ROOT}/search/issues",
            params={"q": " ".join(query), "per_page": 20, "sort": "updated"},
        )
        if response.status_code == 403:
            # Search has its own tight rate limit; back off rather than retry.
            log.warning("GitHub search rate-limited")
            return []
        response.raise_for_status()
        return response.json().get("items", [])

    def _best_match(self, items: list[dict], keywords: list[str]) -> dict | None:
        """Require several keywords in the title, not just one.

        A single shared word ("migration") matches half a repository. Demanding
        agreement on two or more is what keeps this from closing the wrong
        commitment - and closing the wrong one is silent, which makes it worse
        than finding nothing.
        """
        best, best_hits = None, 0
        for item in items:
            title = (item.get("title") or "").lower()
            body = (item.get("body") or "")[:500].lower()
            hits = sum(1 for word in keywords if word in title or word in body)
            if hits > best_hits:
                best, best_hits = item, hits
        return best if best_hits >= self.config.min_keyword_hits else None

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
