"""Guard the derived container-run recipe.

``scripts/container_run.py`` emits the ``docker run`` invocation from the
manifests rather than from a hand-maintained list, because the mount set is a
property of which tools are registered and a hand-written command goes stale
*silently*: a missing mount surfaces as ``ModuleNotFoundError`` deep inside an
engine, not as a startup error.

These tests assert the properties that make the recipe correct, derived at
runtime. They deliberately avoid asserting a snapshot of paths or a count —
tools are still being added, and a snapshot test would break on every addition
while catching nothing (this project replaced two such tests already).
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "container_run.py"


def _load():
    spec = importlib.util.spec_from_file_location("container_run", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def container_run():
    return _load()


@pytest.fixture(scope="module")
def derived(container_run):
    manifest_dir = REPO_ROOT / "src" / "protein_design_mcp" / "manifests"
    prefixes, mounts, _problems = container_run.collect_paths(manifest_dir)
    return prefixes, mounts


def test_every_derived_path_exists_on_this_host(derived):
    """A mount naming a path that is not there fails at engine-import time,
    long after the container has started, with an error that does not mention
    mounting at all."""
    prefixes, mounts = derived
    missing = sorted(p for p in prefixes | mounts if not Path(p).exists())
    assert not missing, f"manifests declare paths that do not exist: {missing}"


def test_no_mount_is_relative_or_traversing(derived):
    """Mounts go in a ``docker run -v`` unquoted by position; a relative or
    ``..`` path would resolve against whatever directory the operator happens to
    be in."""
    prefixes, mounts = derived
    bad = sorted(
        p for p in prefixes | mounts if not p.startswith("/") or ".." in Path(p).parts
    )
    assert not bad, f"mounts must be absolute and free of '..': {bad}"


def test_the_server_never_mounts_its_own_package(derived):
    """Regression for commit 61b0bb2. ``discover_mounts`` reported a *different*
    checkout of this server, because a stale editable install put it on several
    engine environments' ``sys.path``. Mounting it would shadow the container's
    own server code with another working tree — a failure that presents as "the
    container is running old code"."""
    prefixes, mounts = derived
    offenders = sorted(p for p in prefixes | mounts if (Path(p) / "protein_design_mcp").is_dir())
    assert not offenders, f"these mounts would shadow the server's own code: {offenders}"


def test_the_command_pins_exactly_one_gpu(container_run, derived):
    """The GPU is pinned at the container boundary so an engine cannot reach
    another index even by setting CUDA_VISIBLE_DEVICES itself. If this ever
    becomes ``--gpus all`` the constraint silently becomes a convention."""
    prefixes, mounts = derived
    argv = container_run.build_command(prefixes, mounts, "img", "7")
    assert "--device=nvidia.com/gpu=7" in argv
    assert not any(a.startswith("--gpus") for a in argv)


def test_every_mount_is_read_only(container_run, derived):
    """Engine environments and weight caches are inputs. A writable mount lets a
    run mutate the host's shared environments — and several of these paths are
    the user's own project checkouts."""
    prefixes, mounts = derived
    argv = container_run.build_command(prefixes, mounts, "img", "7")
    volumes = [argv[i + 1] for i, a in enumerate(argv) if a == "-v"]
    assert volumes, "expected at least one mount"
    not_ro = [v for v in volumes if not v.endswith(":ro")]
    assert not not_ro, f"mounts must be read-only: {not_ro}"


def test_mounts_use_the_identical_host_path(container_run, derived):
    """Conda environments are not relocatable: their console scripts carry
    absolute shebangs such as ``#!/home/jk661/.conda/envs/boltz/bin/python3.11``.
    Mounting to any other path inside the container breaks every entry point."""
    prefixes, mounts = derived
    argv = container_run.build_command(prefixes, mounts, "img", "7")
    volumes = [argv[i + 1] for i, a in enumerate(argv) if a == "-v"]
    for volume in volumes:
        host, container, mode = volume.rsplit(":", 2)
        assert host == container, f"mount must keep its host path: {volume} ({mode})"


# --- Task 13: container runs as the invoking host user, not the image's
# --- default user (protpardelle's mode-640 model_params/configs fix) -------


def test_the_command_maps_the_container_to_the_given_uid_and_gid(container_run, derived):
    """A mode-640 host file (readable via the host user's own group
    membership) is unreadable to the image's arbitrary default user
    ($MAMBA_USER). Mapping the container onto the invoking user's own
    uid/gid — not chmod'ing the file, not a broader read-only-mount
    exception — makes every already-host-readable mount readable
    identically in-container."""
    prefixes, mounts = derived
    argv = container_run.build_command(prefixes, mounts, "img", "7", uid=1234, gid=5678)
    assert "--user=1234:5678" in argv


def test_the_command_defaults_uid_and_gid_to_the_invoking_process(container_run, derived):
    prefixes, mounts = derived
    argv = container_run.build_command(prefixes, mounts, "img", "7")
    assert f"--user={os.getuid()}:{os.getgid()}" in argv


def test_the_command_sets_a_writable_home_for_the_mapped_user(container_run, derived):
    """The image's own baked-in home directories (e.g. /home/mambauser)
    are not writable by an arbitrary uid, and micromamba needs a writable
    $HOME for its own runtime lockfile -- confirmed live: without this,
    `micromamba run` fails with 'Could not open lockfile
    .../.cache/mamba/proc/proc.lock' the moment the container runs as
    anyone other than the image's own baked-in user."""
    prefixes, mounts = derived
    argv = container_run.build_command(prefixes, mounts, "img", "7")
    idx = argv.index("-e")
    assert argv[idx + 1] == f"HOME={container_run.CONTAINER_HOME}"
