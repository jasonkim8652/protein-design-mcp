#!/usr/bin/env python
"""Emit the ``docker run`` invocation that this server's manifests actually need.

Derived from the manifests, never hand-written. The mount set is a property of
which tools are registered, so a hand-maintained command goes stale the moment a
tool is added — and it goes stale silently, because a missing mount shows up as
``ModuleNotFoundError`` deep inside an engine rather than as a startup error.

Three things this encodes, each established by experiment (see
``docs/superpowers/specs/2026-09-22-gpu-engine-substrate-design.md`` §2):

1. **The GPU is pinned at the container boundary**, not inside it.
   ``--device nvidia.com/gpu=7`` makes the container see exactly one GPU, so an
   engine cannot reach another index even if it sets ``CUDA_VISIBLE_DEVICES``
   itself.
2. **Host conda environments mount at their identical host path.** Their console
   scripts carry absolute shebangs (``#!/home/jk661/.conda/envs/boltz/bin/python3.11``),
   so a relocated mount breaks every entry point in the environment.
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
DEFAULT_GPU = "7"
# See point 4 in this module's own docstring: micromamba's own runtime
# lockfile needs A writable $HOME regardless of which uid the container
# runs as, and /tmp is world-writable (sticky bit) in any ordinary Linux
# image, unlike the image's baked-in /home/$MAMBA_USER.
CONTAINER_HOME = "/tmp"


def collect_paths(manifest_dir: Path) -> tuple[set[str], set[str], list[str]]:
    """Return (prefixes, mounts, problems) across every manifest that loads.

    Uses the resilient loader on purpose: one malformed manifest should not stop
    us printing a command for the rest, and its exclusion is reported rather than
    hidden.
    """
    manifests, failures = load_manifests_resilient(manifest_dir)
    prefixes: set[str] = set()
    mounts: set[str] = set()
    for manifest in manifests:
        engine = manifest.engine
        if getattr(engine, "prefix", None):
            prefixes.add(engine.prefix)
        for mount in getattr(engine, "mounts", ()) or ():
            mounts.add(mount)
    problems = [f"{name}: {reason}" for name, reason in sorted(failures.items())]
    return prefixes, mounts, problems


def build_command(
    prefixes: set[str],
    mounts: set[str],
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
        "docker", "run", "--rm", "-it",
        f"--device=nvidia.com/gpu={gpu}",
        f"--user={uid}:{gid}",
        "-e", f"HOME={CONTAINER_HOME}",
    ]
    # Identical-path, read-only. Sorted so the command is stable between runs and
    # a diff of two invocations is meaningful.
    for path in sorted(prefixes | mounts):
        argv += ["-v", f"{path}:{path}:ro"]
    argv.append(image)
    return argv


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument(
        "--gpu",
        default=DEFAULT_GPU,
        help="GPU index to expose. Only index 7 is ours on this host.",
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
        missing = [p for p in sorted(prefixes | mounts) if not Path(p).exists()]
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
