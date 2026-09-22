"""Task 2: the mount-discovery helper (design §2.3, §3.2).

``mounts`` must be DERIVED, not hand-written, or an engine can ship with a
silently incomplete mount set. ``discover_mounts(prefix, module)`` runs the
TARGET environment's own interpreter (never this process's) so it sees that
environment's real site-packages and ``__editable__*.pth`` machinery.

The primary case is real, not mocked: ``boltz`` is genuinely installed
editable on this host, and section 2.3 of the design doc records exactly
what breaks without the extra mount:

    /home/jk661/.conda/envs/boltz/bin/python -c "import boltz"
    -> ModuleNotFoundError: No module named 'boltz'

although the interpreter runs fine and `import torch` succeeds. This is
"verified, do not re-derive" — the test below exercises the same host paths
that fact was established against.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from protein_design_mcp.mounts import MountDiscoveryError, discover_mounts

BOLTZ_PREFIX = "/home/jk661/.conda/envs/boltz"
BOLTZ_EDITABLE_SRC = "/home/jk661/projects/lightning-boltz-dev/src"

_boltz_env_available = Path(BOLTZ_PREFIX, "bin", "python").exists() and Path(
    BOLTZ_EDITABLE_SRC
).exists()

requires_boltz_env = pytest.mark.skipif(
    not _boltz_env_available,
    reason="real boltz conda env not present on this host",
)


# --- Required case 6: real editable env -------------------------------------


@requires_boltz_env
def test_discovers_the_editable_source_checkout_for_boltz():
    mounts = discover_mounts(BOLTZ_PREFIX, "boltz")
    assert BOLTZ_EDITABLE_SRC in mounts


@requires_boltz_env
def test_discovered_mounts_are_absolute_paths_outside_the_prefix():
    mounts = discover_mounts(BOLTZ_PREFIX, "boltz")
    for mount in mounts:
        assert mount.startswith("/")
        assert not mount.startswith(BOLTZ_PREFIX)


# --- Synthetic fixtures: exercise the mechanism without a real env ---------


def _make_fake_prefix(tmp_path, python_body: str) -> str:
    """A minimal fake 'prefix': just bin/python, a script standing in for
    the real interpreter. Good enough to test discover_mounts' subprocess
    plumbing and JSON parsing without needing a second real conda env.
    """
    prefix = tmp_path / "fake_env"
    bin_dir = prefix / "bin"
    bin_dir.mkdir(parents=True)
    python_path = bin_dir / "python"
    python_path.write_text(python_body)
    python_path.chmod(0o755)
    return str(prefix)


def test_module_that_fails_to_import_raises_mount_discovery_error(tmp_path):
    import sys

    prefix = _make_fake_prefix(
        tmp_path,
        f"#!{sys.executable}\nimport sys\nsys.exit('boom: no such module')\n",
    )
    with pytest.raises(MountDiscoveryError):
        discover_mounts(prefix, "nonexistent_module_xyz")


def test_missing_prefix_python_raises_mount_discovery_error(tmp_path):
    prefix = tmp_path / "no_such_env"
    with pytest.raises(MountDiscoveryError):
        discover_mounts(str(prefix), "whatever")


# --- Consistency: a manifest's declared mounts must match the helper -------
#
# No shipped manifest declares `prefix` yet (this task ships no new tool —
# see task-2-brief.md), so this loop is currently a no-op. It exists so
# that the FIRST manifest that ever declares `prefix` starts being checked
# automatically: an engine reinstalled as non-editable (or newly editable)
# then fails loudly here in CI, rather than at call time inside a
# container. By convention engine.repo names the primary importable module.


@requires_boltz_env
def test_declared_mounts_of_every_prefix_manifest_match_the_helper():
    from protein_design_mcp.manifest.loader import load_manifests

    manifest_dir = Path(__file__).parent.parent / "src" / "protein_design_mcp" / "manifests"
    manifests = load_manifests(manifest_dir)
    prefix_manifests = [m for m in manifests if m.engine.prefix is not None]

    # Today: zero. Once Task 3+ ships a prefix-based manifest, this
    # assertion starts actually exercising the loop below.
    for manifest in prefix_manifests:
        derived = discover_mounts(manifest.engine.prefix, manifest.engine.repo)
        assert sorted(manifest.engine.mounts) == sorted(derived), (
            f"{manifest.name}: declared mounts {manifest.engine.mounts} no "
            f"longer match what discover_mounts derives ({derived}) — the "
            "environment was likely reinstalled non-editable or newly "
            "editable; regenerate with `python -m protein_design_mcp.mounts"
            f" {manifest.engine.prefix} {manifest.engine.repo}`"
        )
