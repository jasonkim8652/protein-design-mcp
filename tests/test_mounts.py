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
    """Every mount ``discover_mounts`` finds for module resolution must be
    declared -- that is the actual regression this test guards against (an
    engine silently reinstalled non-editable/newly-editable, caught here
    instead of at call time inside a container).

    This was written as an exact-equality check when it was still a no-op
    ("Today: zero" below, from when this test was added a task before any
    prefix-based manifest existed). The design doc's own §3.2 says
    ``mounts`` covers more than module resolution -- "editable source
    checkouts, user-site directories, WEIGHT DIRECTORIES" -- and a weight
    cache (e.g. `~/.cache/huggingface`, `~/.foundry/checkpoints`,
    `~/promera_weights`) is never something ``discover_mounts`` can find,
    since nothing `import`s it. The first real prefix-based manifests
    (run_esmfold2, run_promera, run_rf3, run_alphafold2_multimer) all
    legitimately declare such mounts beyond what discover_mounts derives,
    so the check here is a subset, not exact equality: every discovered
    module-resolution mount must still be present (that regression is
    still caught), and a manifest may declare additional mounts
    discover_mounts has no way to know about.

    "Present" means REACHABLE, not string-identical. The invariant being
    guarded is that every path the engine needs to resolve its imports
    exists inside the container; a declared mount of a parent directory
    satisfies it, because `-v /a:/a:ro` makes `/a/b` reachable too. So
    run_rfdiffusion2 declaring `/home/jk661/projects/RFdiffusion2` covers
    the derived `/home/jk661/projects/RFdiffusion2/rf_diffusion` -- the
    package directory inside the checkout the manifest already mounts.
    The containment is one-directional and checked on path components
    (not string prefixes): a declared CHILD does not cover a discovered
    PARENT, and `/a/bc` does not cover `/a/b`, so a genuinely absent
    mount still fails.
    """
    from protein_design_mcp.manifest.loader import load_manifests

    manifest_dir = Path(__file__).parent.parent / "src" / "protein_design_mcp" / "manifests"
    manifests = load_manifests(manifest_dir)
    # run_alphafold3 declares engine.prefix_host (task 16) and is
    # deliberately EXCLUDED here, not just probed at a different path.
    # discover_mounts runs ``<probe_path>/bin/python -c <script>`` on THIS
    # (bare, host-side) process -- but AF3's venv only resolves its own
    # `alphafold3` package (a hand-patched scikit-build-core editable
    # redirect, see run_alphafold3.yaml's own comment) through absolute
    # paths that hardcode the CONTAINER-side prefix
    # (`/alphafold3_venv/app/alphafold/src/...`), which never exists
    # outside the deployed container regardless of whether prefix_host is
    # used as the probe target -- CONFIRMED LIVE, 2026-09-23:
    # `/opt/alphafold3_data/alphafold3_venv/bin/python -c "import
    # alphafold3"` raises `FileNotFoundError:
    # '/alphafold3_venv/app/alphafold/src/alphafold3/__init__.py'` even
    # though that exact python IS the real interpreter. So this static,
    # host-side check is inherently inapplicable to a relocated prefix --
    # the equivalent guarantee for run_alphafold3 comes from actually
    # running inside the container (this task's own in-container
    # ServerApp.call_tool + live_proof.py verification), not from a
    # host-side import probe.
    prefix_manifests = [
        m for m in manifests if m.engine.prefix is not None and m.engine.prefix_host is None
    ]

    # Today: zero non-relocated ones beyond the pre-existing GPU engines.
    for manifest in prefix_manifests:
        # engine.env_vars is passed through: an engine like esmfold2 whose
        # correct import depends on PYTHONNOUSERSITE=1 would otherwise be
        # probed without it, reproducing the very shadowing bug that
        # variable exists to prevent and deriving the WRONG package's
        # mounts (see discover_mounts' own `env` parameter docstring).
        derived = discover_mounts(
            manifest.engine.prefix, manifest.engine.repo, env=manifest.engine.env_vars
        )
        declared = [Path(m) for m in manifest.engine.mounts]
        missing = sorted(
            path
            for path in derived
            if not any(
                Path(path) == mount or mount in Path(path).parents for mount in declared
            )
        )
        assert not missing, (
            f"{manifest.name}: declared mounts {manifest.engine.mounts} are "
            f"missing module-resolution mount(s) {missing} that "
            "discover_mounts derives -- the environment was likely "
            "reinstalled non-editable or newly editable; regenerate with "
            f"`python -m protein_design_mcp.mounts {manifest.engine.prefix} "
            f"{manifest.engine.repo}` and merge the result into "
            "engine.mounts (keeping any additional non-import mounts, e.g. "
            "weight directories, already declared there)."
        )
