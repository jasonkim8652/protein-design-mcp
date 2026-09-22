"""Derive the host paths a GPU engine's ``EngineSpec.mounts`` needs.

Design §2.3 established, by experiment inside a container, that an editable
install needs its source checkout mounted too — the environment mount alone
is not enough:

    /home/jk661/.conda/envs/boltz/bin/python -c "import boltz"
    -> ModuleNotFoundError: No module named 'boltz'

although the interpreter runs fine and ``import torch`` succeeds. The cause
is ``site-packages/__editable__.boltz-2.2.1.pth`` pointing at
``/home/jk661/projects/lightning-boltz-dev/src``. Two more engines resolve
their module OUTSIDE the env entirely, from the user-site directory.

``mounts`` must therefore be DERIVED, not hand-written, or a manifest can
ship with a silently incomplete mount set. ``discover_mounts`` is that
derivation, and it is also exposed as a small CLI so a contributor adding an
engine can generate the list instead of guessing it::

    python -m protein_design_mcp.mounts /home/jk661/.conda/envs/boltz boltz
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

# Executed under the TARGET prefix's own interpreter (never this process's)
# so module resolution goes through THAT environment's site-packages, .pth
# files and user-site — the whole point being that this can differ from
# whatever interpreter happens to be running discover_mounts itself.
_PROBE_SCRIPT = """
import importlib, json, os, site, sys


def _site_dirs():
    dirs = set()
    try:
        dirs.update(site.getsitepackages())
    except Exception:
        pass
    for p in sys.path:
        if p and (p.endswith("site-packages") or p.endswith("dist-packages")):
            dirs.add(p)
    return dirs


def main(module_name):
    mod = importlib.import_module(module_name)
    origin = getattr(mod, "__file__", None)
    if origin is None:
        paths = list(getattr(mod, "__path__", []) or [])
        origin = paths[0] if paths else None

    editable_targets = []
    for site_dir in _site_dirs():
        if not os.path.isdir(site_dir):
            continue
        for name in sorted(os.listdir(site_dir)):
            if not (name.startswith("__editable__") and name.endswith(".pth")):
                continue
            try:
                with open(os.path.join(site_dir, name)) as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if line.startswith("import "):
                            continue
                        if os.path.isabs(line):
                            editable_targets.append(line)
            except OSError:
                pass

    try:
        user_site = site.getusersitepackages()
    except Exception:
        user_site = None

    print(json.dumps({
        "origin": origin,
        "editable_targets": sorted(set(editable_targets)),
        "user_site": user_site,
    }))


main(sys.argv[1])
"""


class MountDiscoveryError(RuntimeError):
    """A prefix's interpreter could not be used to resolve a module."""


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root)
    except ValueError:
        return False
    return True


def discover_mounts(
    prefix: str, module: str, env: dict[str, str] | None = None
) -> list[str]:
    """Return the read-only host paths ``module`` needs mounted beyond ``prefix``.

    Runs ``<prefix>/bin/python`` — not this process's interpreter — because
    resolution has to happen under the target environment's own
    site-packages and ``.pth`` machinery (design §2.3). The returned set is:

    - every absolute target inside a ``site-packages/__editable__*.pth``
      file in that environment, when it lies outside ``prefix`` (an
      editable install's real source checkout);
    - the module's resolved origin's containing directory, when that origin
      falls outside ``prefix`` and isn't already covered by an editable
      target above (this is the ``esmfold2``-from-``~/.local`` case: no
      ``.pth`` involved at all, the module just resolves from user-site);
    - when that fallback origin is inside the user-site directory, the
      user-site directory itself is returned in its place, matching the
      design's "user-site, not the env" wording — a single package's
      directory would not by itself explain why the mount is there.

    Paths already inside ``prefix`` are never returned: the prefix is
    mounted separately (§2.2), and repeating it would be redundant, not
    additive.

    ``env``, if given, is merged over a COPY of this process's own
    environment before the probe subprocess runs (never replacing it
    outright — same rule ``dispatch.env.EnvDispatcher.run`` follows for
    ``engine.env_vars``). Needed for an engine like ``esmfold2`` whose
    correct import depends on a variable such as ``PYTHONNOUSERSITE=1``
    (see ``run_esmfold2.yaml``): probing WITHOUT that variable set
    reproduces the very shadowing bug it exists to prevent and reports the
    WRONG package's mounts. Omitted (the default), this probes with only
    this process's own environment, exactly as before this parameter
    existed.

    Raises ``MountDiscoveryError`` if ``prefix`` has no usable
    ``bin/python``, or if importing ``module`` under it fails.
    """
    python = str(Path(prefix) / "bin" / "python")
    subprocess_env = {**os.environ, **env} if env else None
    try:
        proc = subprocess.run(
            [python, "-c", _PROBE_SCRIPT, module],
            capture_output=True,
            text=True,
            timeout=120,
            env=subprocess_env,
        )
    except FileNotFoundError as exc:
        raise MountDiscoveryError(
            f"no interpreter found under prefix {prefix!r} ({python!r} "
            f"does not exist or is not executable): {exc}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise MountDiscoveryError(
            f"resolving module {module!r} under prefix {prefix!r} timed out"
        ) from exc

    if proc.returncode != 0:
        raise MountDiscoveryError(
            f"could not import {module!r} under prefix {prefix!r} using "
            f"{python!r}:\n{proc.stderr.strip()[-2000:]}"
        )

    try:
        last_line = proc.stdout.strip().splitlines()[-1]
        info = json.loads(last_line)
    except (IndexError, json.JSONDecodeError) as exc:
        raise MountDiscoveryError(
            f"could not parse mount-discovery output for {module!r} under "
            f"prefix {prefix!r}: {proc.stdout!r}"
        ) from exc

    prefix_path = Path(prefix).resolve()
    mounts: set[str] = set()

    for target in info.get("editable_targets", []):
        target_path = Path(target)
        if not _is_within(target_path, prefix_path):
            mounts.add(str(target_path))

    origin = info.get("origin")
    if origin:
        origin_path = Path(origin)
        already_covered = _is_within(origin_path, prefix_path) or any(
            _is_within(origin_path, Path(m)) for m in mounts
        )
        if not already_covered:
            user_site = info.get("user_site")
            if user_site and _is_within(origin_path, Path(user_site)):
                mounts.add(user_site)
            else:
                mounts.add(str(origin_path.parent))

    return sorted(mounts)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m protein_design_mcp.mounts",
        description=(
            "Derive the mounts= a manifest needs for one engine's module, "
            "by importing it under the environment's OWN interpreter."
        ),
    )
    parser.add_argument("prefix", help="Absolute path to the host conda environment")
    parser.add_argument("module", help="The module the engine's entry point imports")
    args = parser.parse_args(argv)

    try:
        mounts = discover_mounts(args.prefix, args.module)
    except MountDiscoveryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    for mount in mounts:
        print(mount)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
