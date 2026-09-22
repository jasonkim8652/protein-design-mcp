"""Every in-image environment a manifest names must exist in the image.

A manifest with ``engine.env: X`` dispatches as ``micromamba run -n X``, which
resolves only under the image's own root prefix. If ``Dockerfile.envs`` never
creates ``X``, nothing fails at load, at registration, or in any unit test —
the tool appears in ``list_tools()``, accepts arguments, and dies only when a
caller actually invokes it, with ``critical libmamba The given prefix does not
exist``.

That is exactly how ``run_mmseqs_search`` shipped declaring ``env: mmseqs``
while the image created only ``scoring``, ``md``, ``mpnn`` and ``server``.

Host-mounted environments (``engine.prefix``) are a different mechanism and are
covered by ``tests/test_container_run.py``; they are deliberately ignored here.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCKERFILE = REPO_ROOT / "Dockerfile.envs"
MANIFEST_DIR = REPO_ROOT / "src" / "protein_design_mcp" / "manifests"

# `micromamba create -y -n <name>` — the only way an env enters the image.
CREATE = re.compile(r"micromamba\s+create\b[^\n]*?\s-n\s+([A-Za-z0-9_.-]+)")


def image_envs() -> set[str]:
    return set(CREATE.findall(DOCKERFILE.read_text()))


def manifest_envs() -> dict[str, str]:
    """Map each declared in-image env to a tool that needs it."""
    declared: dict[str, str] = {}
    for path in sorted(MANIFEST_DIR.glob("*.yaml")):
        try:
            data = yaml.safe_load(path.read_text()) or {}
        except yaml.YAMLError:
            # A malformed manifest is the manifest-loader tests' problem, not
            # this one; skipping keeps this test's failure message about envs.
            continue
        engine = data.get("engine") or {}
        name = engine.get("env")
        if name:
            declared.setdefault(name, data.get("name") or path.name)
    return declared


def test_the_image_creates_every_env_a_manifest_names():
    declared = manifest_envs()
    missing = {env: tool for env, tool in declared.items() if env not in image_envs()}
    assert not missing, (
        "manifests name in-image environments the Dockerfile never creates — "
        "these tools would register fine and then fail at call time: "
        + ", ".join(f"{env} (needed by {tool})" for env, tool in sorted(missing.items()))
    )


def test_the_dockerfile_declares_at_least_the_known_core_envs():
    """Guards the regex, not the Dockerfile. If `micromamba create` is ever
    reformatted so CREATE stops matching, the test above would pass vacuously
    by finding nothing to compare against — a green test proving nothing."""
    found = image_envs()
    assert {"scoring", "md", "mpnn", "server"} <= found, (
        f"env-parsing regex likely broke; parsed only: {sorted(found)}"
    )
