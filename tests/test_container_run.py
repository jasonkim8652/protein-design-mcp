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
MANIFEST_DIR = REPO_ROOT / "src" / "protein_design_mcp" / "manifests"


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
    """``(prefixes, mounts)``, each a set of ``(host_path, container_path)``
    pairs — see ``container_run.collect_paths``'s own docstring for why a
    pair, not a bare path string, is the unit here."""
    prefixes, mounts, _problems = container_run.collect_paths(MANIFEST_DIR)
    return prefixes, mounts


@pytest.fixture(scope="module")
def manifests():
    """The real, loaded manifests — used only by the one test below that
    needs to cross-check a non-identical mount pair against its OWN
    ``engine.prefix_host`` declaration, rather than trusting
    ``collect_paths``'s round-trip of its own logic."""
    from protein_design_mcp.manifest.loader import load_manifests_resilient

    loaded, _failures = load_manifests_resilient(MANIFEST_DIR)
    return loaded


def test_every_derived_path_exists_on_this_host(derived):
    """A mount naming a path that is not there fails at engine-import time,
    long after the container has started, with an error that does not mention
    mounting at all. Checked on the HOST side of each pair — the
    container-side path (e.g. run_alphafold3's ``/alphafold3_venv``) is
    never expected to exist here; nothing on this host is allowed to write
    at that path (see ``EngineSpec.prefix_host``'s docstring)."""
    prefixes, mounts = derived
    missing = sorted(host for host, _container in prefixes | mounts if not Path(host).exists())
    assert not missing, f"manifests declare paths that do not exist: {missing}"


def test_no_mount_is_relative_or_traversing(derived):
    """Mounts go in a ``docker run -v`` unquoted by position; a relative or
    ``..`` path would resolve against whatever directory the operator happens
    to be in. Checked on BOTH sides of every pair — a relocated mount's
    container-side path is just as position-sensitive as its host side."""
    prefixes, mounts = derived
    bad = sorted(
        f"{host}:{container}"
        for host, container in prefixes | mounts
        for p in (host, container)
        if not p.startswith("/") or ".." in Path(p).parts
    )
    assert not bad, f"mounts must be absolute and free of '..': {bad}"


def test_the_server_never_mounts_its_own_package(derived):
    """Regression for commit 61b0bb2. ``discover_mounts`` reported a *different*
    checkout of this server, because a stale editable install put it on several
    engine environments' ``sys.path``. Mounting it would shadow the container's
    own server code with another working tree — a failure that presents as "the
    container is running old code". Checked on the HOST side: that is what is
    actually read off disk regardless of which container path it lands at."""
    prefixes, mounts = derived
    offenders = sorted(
        host for host, _container in prefixes | mounts
        if (Path(host) / "protein_design_mcp").is_dir()
    )
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


def test_mounts_use_the_identical_host_path(container_run, derived, manifests):
    """An environment must be mounted where it believes it lives.

    For an ordinary conda environment, that belief IS its host install path:
    console scripts carry absolute shebangs such as
    ``#!/home/jk661/.conda/envs/boltz/bin/python3.11``, baked in at creation
    time by conda itself. Mounting anywhere else breaks every entry point in
    the environment, so host and container path must be identical — full
    stop — for every ``engine.mounts`` entry (which has no override
    mechanism at all) and for every ``engine.prefix`` that does not declare
    ``prefix_host``.

    AlphaFold 3's ``/alphafold3_venv`` believes it lives at that exact path
    for the SAME reason (its own console-script shebangs, plus a
    hand-patched editable-install redirect table baked at extraction time —
    see run_alphafold3.yaml's own comment) — but that belief has nothing to
    do with where its 8+GB of files are actually stored on THIS host's
    disk, since this venv was never created here; it was extracted from
    ``romerolabduke/alphafast:latest``, and nothing under this host's ``/``
    is writable at ``/alphafold3_venv`` itself (root owns it). So its
    ``prefix`` (the belief, i.e. the container path) and its
    ``prefix_host`` (wherever we actually put the bytes) legitimately
    differ, and ``collect_paths`` mounts ``prefix_host:prefix``.

    This must not become a general loophole: it is checked here against the
    REAL manifest declarations, not merely against ``collect_paths``'s own
    round-trip of its own logic. Every ``mounts`` pair must be identical
    unconditionally (that field has no relocation field to point to at
    all), and every non-identical ``prefixes`` pair must be traceable to an
    ``engine.prefix_host`` that names that exact host path on an engine
    whose ``engine.prefix`` is that exact container path — so a conda
    environment quietly moved without updating its manifest (or a future
    manifest that sets ``prefix_host`` on an engine that does not actually
    need it) still has to explain itself here, and a manifest that never
    touches ``prefix_host`` (every engine but ``run_alphafold3`` today)
    still gets the ORIGINAL, unweakened identical-path requirement.
    """
    prefixes, mounts = derived
    argv = container_run.build_command(prefixes, mounts, "img", "7")
    volumes = [argv[i + 1] for i, a in enumerate(argv) if a == "-v"]

    # (host, container) -> the set of engine.repo values that legitimately
    # declared this exact relocation via prefix_host, straight from the
    # real manifests -- not from collect_paths's own derivation.
    declared_relocations: dict[tuple[str, str], set[str]] = {}
    for manifest in manifests:
        engine = manifest.engine
        host = getattr(engine, "prefix_host", None)
        prefix = getattr(engine, "prefix", None)
        if host and prefix:
            declared_relocations.setdefault((host, prefix), set()).add(engine.repo)

    mount_pairs = {(h, c) for h, c in mounts}
    for volume in volumes:
        host, container, mode = volume.rsplit(":", 2)
        if host == container:
            continue
        assert (host, container) not in mount_pairs, (
            f"engine.mounts entries have no relocation mechanism -- must "
            f"keep their host path: {volume} ({mode})"
        )
        assert (host, container) in declared_relocations, (
            f"mount must keep its host path, unless an engine.prefix_host "
            f"in the manifests explicitly declares this exact relocation: "
            f"{volume} ({mode})"
        )


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
