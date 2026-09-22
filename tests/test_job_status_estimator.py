"""Tests for job_status._estimate_time_remaining.

Wave H replaced the old estimator, which hardcoded per-minute costs for
"rfdiffusion"/"proteinmpnn"/"esmfold" -- the OLD composite pipeline's three
steps, which no longer exist in this server. The new estimator derives a
rate from the job's OWN observed progress (elapsed time / designs completed)
instead of a fixed table. These tests cover the corner cases WAVE-COMMON and
CLAUDE.md's TDD workflow both require: 0, None, a single item, and the
boundaries around "no rate can be computed yet".
"""

from __future__ import annotations

from datetime import datetime, timedelta

from protein_design_mcp.job_status import _estimate_time_remaining


def test_zero_total_returns_none():
    """total_designs = 0: nothing to estimate against."""
    assert _estimate_time_remaining(0, 0, datetime.now()) is None


def test_none_started_at_returns_none():
    """A job with no recorded start time has no elapsed time to derive a
    rate from."""
    assert _estimate_time_remaining(3, 10, None) is None


def test_zero_completed_returns_none_explicitly():
    """completed = 0 must be checked explicitly, not skipped as falsy --
    zero designs completed over SOME elapsed time is not a rate."""
    started = datetime.now() - timedelta(minutes=5)
    assert _estimate_time_remaining(0, 10, started) is None


def test_negative_completed_is_treated_like_zero():
    """Defensive: a completed count that is somehow negative must not be
    divided into, producing a nonsense negative rate."""
    started = datetime.now() - timedelta(minutes=5)
    assert _estimate_time_remaining(-1, 10, started) is None


def test_completed_equals_total_is_boundary_almost_done():
    """remaining = 0 is a boundary: nothing left to project, so the
    estimate is the same "almost done" floor as a tiny remainder."""
    started = datetime.now() - timedelta(minutes=5)
    assert _estimate_time_remaining(10, 10, started) == "less than 1 minute"


def test_single_item_total_one_completed_one():
    """Aggregation with n=1: total=1, completed=1 -- remaining is 0, same
    "almost done" boundary as the general completed==total case."""
    started = datetime.now() - timedelta(minutes=2)
    assert _estimate_time_remaining(1, 1, started) == "less than 1 minute"


def test_future_started_at_returns_none_rather_than_nonsense():
    """A non-positive elapsed time (clock skew, or started_at in the
    future) must refuse to project a rate rather than divide by a
    near-zero or negative elapsed time."""
    started = datetime.now() + timedelta(minutes=5)
    assert _estimate_time_remaining(2, 10, started) is None


def test_normal_case_projects_minutes_from_observed_rate():
    """2 designs completed in 10 minutes -> 5 min/design; 8 remaining ->
    ~40 minutes projected."""
    started = datetime.now() - timedelta(minutes=10)
    result = _estimate_time_remaining(2, 10, started)
    assert result == "40 minutes"


def test_normal_case_projects_hours_when_over_sixty_minutes():
    """1 design completed in 40 minutes -> 40 min/design; 9 remaining ->
    360 minutes = 6.0 hours."""
    started = datetime.now() - timedelta(minutes=40)
    result = _estimate_time_remaining(1, 10, started)
    assert result == "6.0 hours"
