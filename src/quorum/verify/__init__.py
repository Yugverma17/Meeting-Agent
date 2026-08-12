"""Checking whether committed work actually happened, outside the meeting."""

from quorum.verify.github import GitHubEvidenceProvider, keywords_for

__all__ = ["GitHubEvidenceProvider", "keywords_for"]
