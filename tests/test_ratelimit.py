from __future__ import annotations

import time

from quorum.llm.ratelimit import QuotaTracker, estimate_tokens


def test_rpm_blocks_after_limit(quota):
    for _ in range(3):
        assert quota.check("m", rpm=3, rpd=100, tpm=None, est_tokens=10).allowed
        quota.record("m", 10)

    verdict = quota.check("m", rpm=3, rpd=100, tpm=None, est_tokens=10)
    assert not verdict.allowed
    assert "RPM" in verdict.reason
    assert 0 < verdict.wait_s <= 61


def test_rpd_blocks_and_reports_long_wait(quota):
    for _ in range(2):
        quota.record("m", 5)
    verdict = quota.check("m", rpm=100, rpd=2, tpm=None, est_tokens=5)
    assert not verdict.allowed
    assert "RPD" in verdict.reason
    assert verdict.wait_s > 60 * 60


def test_tpm_blocks_when_budget_would_be_exceeded(quota):
    """Groq's 6k TPM is the binding constraint - this is the path that fires."""
    quota.record("groq:m", 5_000)
    verdict = quota.check("groq:m", rpm=30, rpd=14_400, tpm=6_000, est_tokens=2_000)
    assert not verdict.allowed
    assert "TPM" in verdict.reason


def test_single_prompt_larger_than_tpm_fails_immediately(quota):
    """A prompt bigger than the whole per-minute budget can never succeed, so
    waiting is pointless - the router must fall back to another model instead."""
    verdict = quota.check("groq:m", rpm=30, rpd=14_400, tpm=6_000, est_tokens=9_000)
    assert not verdict.allowed
    assert verdict.wait_s == 0.0
    assert "exceeds TPM" in verdict.reason


def test_usage_within_budget_is_allowed(quota):
    quota.record("m", 1_000)
    assert quota.check("m", rpm=30, rpd=1_000, tpm=6_000, est_tokens=2_000).allowed


def test_state_persists_across_instances(tmp_path):
    path = tmp_path / "quota.json"
    first = QuotaTracker(path)
    for _ in range(5):
        first.record("m", 100)

    # A crashed-and-resumed eval run must not get a fresh daily budget.
    second = QuotaTracker(path)
    assert second.usage("m")["requests_last_day"] == 5
    assert second.usage("m")["tokens_last_day"] == 500


def test_corrupt_state_file_does_not_crash(tmp_path):
    path = tmp_path / "quota.json"
    path.write_text("{ not json", encoding="utf-8")
    tracker = QuotaTracker(path)
    assert tracker.usage("m")["requests_last_day"] == 0


def test_reset_clears_usage(quota):
    quota.record("m", 100)
    quota.reset("m")
    assert quota.usage("m")["requests_last_day"] == 0


def test_snapshot_lists_all_models(quota):
    quota.record("a", 1)
    quota.record("b", 2)
    snapshot = quota.snapshot()
    assert set(snapshot) == {"a", "b"}


def test_estimate_tokens_scales_and_is_never_zero():
    assert estimate_tokens("") >= 1
    short = estimate_tokens("hello world")
    long = estimate_tokens("hello world " * 100)
    assert long > short * 10


def test_old_events_fall_out_of_the_minute_window(quota, monkeypatch):
    quota.record("m", 10)
    real_time = time.time
    monkeypatch.setattr(time, "time", lambda: real_time() + 61)
    assert quota.usage("m")["requests_last_minute"] == 0
    assert quota.usage("m")["requests_last_day"] == 1
