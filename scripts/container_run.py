#!/usr/bin/env python
"""Emit the ``docker run`` invocation that this server's manifests actually need.

Derived from the manifests, never hand-written. The mount set is a property of
which tools are registered, so a hand-maintained command goes stale the moment a
tool is added — and it goes stale silently, because a missing mount shows up as
``ModuleNotFoundError`` deep inside an engine rather than as a startup error.

Four things this encodes, each established by experiment (see
``docs/superpowers/specs/2026-09-22-gpu-engine-substrate-design.md`` §2):

1. **The GPU is pinned at the container boundary**, not inside it.
   ``--device nvidia.com/gpu=N`` makes the container see exactly one GPU, so an
   engine cannot reach another index even if it sets ``CUDA_VISIBLE_DEVICES``
   itself. Which index is a deployment's own business (``PROTEIN_DESIGN_GPU``);
   that it is exactly one is this script's guarantee.
2. **A mounted environment must land where it believes it lives.** For an
   ordinary conda environment that belief IS its host install path — its
   console scripts carry absolute shebangs
   (``#!/home/jk661/.conda/envs/boltz/bin/python3.11``), baked in at creation
   time — so host and container path have to be identical or every entry
   point breaks. ``EngineSpec.prefix_host`` (see that field's own docstring)
   is the one, narrow, schema-validated exception: it lets a manifest name a
   DIFFERENT host path than ``engine.prefix`` for the one case where an
   environment's belief about where it lives was fixed by hand rather than by
   where conda put it (``run_alphafold3``'s venv, extracted from an image,
   not created on this host) — the container-side path (what the environment
   believes) never moves, only where its bytes physically sit on this host's
   disk does. Every other mount — every ``engine.mounts`` entry, and every
   ``engine.prefix`` that does not set ``prefix_host`` — still mounts at its
   own identical path, exactly as before this field existed.
3. **Editable installs need their source checkout mounted too.** With only the
   environment mounted, ``import boltz`` raises ``ModuleNotFoundError`` while the
   interpreter runs fine and ``import torch`` succeeds. Half the engine
   environments on this host are editable.
4. **The container runs AS the invoking host user, not the image's default
   user.** Established in-container (task-13-report.md): some mounted host
   files (e.g. ``run_protpardelle``'s ``model_params/configs/*.yaml``) are
   mode 640, readable only via the host user's own group membership
   (``ldapusers``). The image's default user (``$MAMBA_USER``, an arbitrary
   fixed uid/gid baked at build time) is in no such group, so a correctly
   *mounted* file is still unreadable. Chmod'ing the user's own files, or
   granting the mount group-world-readable, are both worse than mapping the
   container's own uid/gid onto the invoking user's — every file already
   readable on the host (which is everything this recipe mounts, since the
   invoking user owns or group-reads all of it) is then readable identically
   in-container, with no broader exposure than the host already has. ``-e
   HOME=/tmp`` goes with it: the image's own baked-in directories (e.g.
   ``/home/mambauser``) are NOT world-writable, so an arbitrary uid has
   nowhere to put micromamba's own runtime lockfile (``$HOME/.cache/mamba/proc``)
   unless $HOME is pointed at something that is (``/tmp``, always
   world-writable). Engine-specific $HOME needs (e.g. the promera/rf3/rfd3
   checkpoint-path fix) are handled separately, per-subprocess, by
   ``EnvDispatcher`` — see its own ``_HOST_HOME`` — and are unaffected by
   this container-wide default.

Usage::

    python scripts/container_run.py                     # print the command
    PROTEIN_DESIGN_GPU=7 python scripts/container_run.py
    python scripts/container_run.py --gpu 7 --image protein-design-mcp:envs
    python scripts/container_run.py --check             # verify every path exists
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from protein_design_mcp.manifest.loader import load_manifests_resilient  # noqa: E402

DEFAULT_IMAGE = "protein-design-mcp:envs"

# Docker's default /dev/shm is 64MB, and PyTorch's DataLoader moves tensors
# between its workers through shared memory. Overflowing it is silent: the
# worker waits for space that never comes and the parent waits for the
# worker, so the run shows 0% GPU, a frozen VRAM figure and an empty stderr
# rather than an error. Measured through this image on one 504-residue chain
# with run_boltz -- 64MB: still running at 900s; 16g: 166.7s (21.5s outside
# the container). 252- and 302-token inputs complete under the default, so
# the symptom reads as "large folds are slow" until someone checks /dev/shm.
SHM_SIZE = "16g"

#: Environment variable naming the GPU index to expose to the container.
GPU_ENV_VAR = "PROTEIN_DESIGN_GPU"
#: Index used when that variable is unset.
FALLBACK_GPU = "0"


def default_gpu() -> str:
    """The GPU index to expose, from ``PROTEIN_DESIGN_GPU`` or 0.

    This used to be a hardcoded ``"7"`` because index 7 is the only GPU that
    is ours on the development host. That is a fact about one machine, not
    about this software, and shipping it in a release points every other user
    at an index that need not exist. The single-GPU guarantee is unaffected --
    what moves is *which* index, never *how many*, because the pinning still
    happens at the container boundary (see ``build_command``).
    """
    return os.environ.get(GPU_ENV_VAR) or FALLBACK_GPU
# See point 4 in this module's own docstring: micromamba's own runtime
# lockfile needs A writable $HOME regardless of which uid the container
# runs as, and /tmp is world-writable (sticky bit) in any ordinary Linux
# image, unlike the image's baked-in /home/$MAMBA_USER.
CONTAINER_HOME = "/tmp"


def collect_paths(
    manifest_dir: Path,
) -> tuple[set[tuple[str, str]], set[tuple[str, str]], list[str]]:
    """Return (prefixes, mounts, problems) across every manifest that loads.

    ``prefixes`` and ``mounts`` are both sets of ``(host_path,
    container_path)`` pairs, not bare path strings — see point 2 of this
    module's own docstring. For every ``engine.mounts`` entry the pair is
    always ``(mount, mount)``: that field has no relocation mechanism, full
    stop. For a prefix, the pair is ``(engine.prefix_host or engine.prefix,
    engine.prefix)`` — identical unless the manifest explicitly set
    ``prefix_host`` (see that field's own docstring), which is the ONLY way
    a non-identical pair can ever reach this function. A manifest with a
    plain ``prefix`` and no ``prefix_host`` (every engine but
    ``run_alphafold3`` today) always yields an identical pair here, exactly
    as it did before ``prefix_host`` existed.

    Uses the resilient loader on purpose: one malformed manifest should not stop
    us printing a command for the rest, and its exclusion is reported rather than
    hidden.
    """
    manifests, failures = load_manifests_resilient(manifest_dir)
    prefixes: set[tuple[str, str]] = set()
    mounts: set[tuple[str, str]] = set()
    for manifest in manifests:
        engine = manifest.engine
        if getattr(engine, "prefix", None):
            host = getattr(engine, "prefix_host", None) or engine.prefix
            prefixes.add((host, engine.prefix))
        for mount in getattr(engine, "mounts", ()) or ():
            mounts.add((mount, mount))
    problems = [f"{name}: {reason}" for name, reason in sorted(failures.items())]
    return prefixes, mounts, problems


def build_command(
    prefixes: set[tuple[str, str]],
    mounts: set[tuple[str, str]],
    image: str,
    gpu: str,
    *,
    uid: int | None = None,
    gid: int | None = None,
) -> list[str]:
    # Default to the process actually invoking this script — see point 4 of
    # this module's own docstring. Overridable (uid/gid params) only so
    # tests can assert on a fixed value instead of the test runner's own.
    if uid is None:
        uid = os.getuid()
    if gid is None:
        gid = os.getgid()
    argv = [
        # -i (keep stdin open) but NOT -t: the image's default command is the
        # MCP server speaking JSON-RPC over stdio, and a client pipes into it.
        # With -t docker refuses outright ("cannot attach stdin to a
        # TTY-enabled container because stdin is not a terminal"), so this
        # command could not be used as the `mcpServers` entry it exists to
        # produce. -t only ever suited running a proof script in a terminal.
        "docker", "run", "--rm", "-i",
        # Docker's 64MB default for /dev/shm is smaller than the tensors a
        # DataLoader worker hands to its parent for a large structure. Too
        # small does not raise: the worker blocks on shared memory that never
        # frees, the parent blocks on the worker, and the GPU idles with an
        # empty stderr. See SHM_SIZE.
        f"--shm-size={SHM_SIZE}",
        f"--device=nvidia.com/gpu={gpu}",
        f"--user={uid}:{gid}",
        "-e", f"HOME={CONTAINER_HOME}",
    ]
    # Read-only, host:container (identical unless a prefix's own
    # prefix_host overrides it — see collect_paths). Sorted so the command
    # is stable between runs and a diff of two invocations is meaningful.
    for host, container in sorted(prefixes | mounts):
        argv += ["-v", f"{host}:{container}:ro"]
    argv.append(image)
    return argv


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument(
        "--gpu",
        default=default_gpu(),
        help=(
            f"GPU index to expose to the container. Defaults to ${GPU_ENV_VAR} "
            f"if set, otherwise {FALLBACK_GPU}. Exactly one index is exposed "
            "either way."
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify every derived path exists on this host and exit non-zero if not.",
    )
    args = parser.parse_args()

    manifest_dir = REPO_ROOT / "src" / "protein_design_mcp" / "manifests"
    prefixes, mounts, problems = collect_paths(manifest_dir)

    for problem in problems:
        print(f"# excluded manifest: {problem}", file=sys.stderr)

    if args.check:
        # Existence is a property of the HOST side of each pair — the
        # container-side path (e.g. run_alphafold3's /alphafold3_venv) is
        # never expected to exist on this host at all; see collect_paths.
        missing = [
            host for host, _container in sorted(prefixes | mounts) if not Path(host).exists()
        ]
        for path in missing:
            print(f"MISSING: {path}", file=sys.stderr)
        print(
            f"{len(prefixes)} prefixes, {len(mounts)} extra mounts, "
            f"{len(missing)} missing",
            file=sys.stderr,
        )
        return 1 if missing else 0

    print(" \\\n  ".join(shlex.quote(a) for a in build_command(prefixes, mounts, args.image, args.gpu)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
