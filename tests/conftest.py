"""Suite-wide pytest configuration.

STRICT_MANIFESTS=1 by default: a broken manifest must fail the build in CI,
never ship silently (see manifest/loader.py's load_manifests_resilient).
Individual tests that exercise the *lenient* runtime behaviour (one bad
manifest excluded, not the whole load aborted) turn this off explicitly via
monkeypatch, which restores the pre-test environment automatically at
teardown — the default here is never weakened for any other test.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _strict_manifests_by_default(monkeypatch):
    monkeypatch.setenv("STRICT_MANIFESTS", "1")
