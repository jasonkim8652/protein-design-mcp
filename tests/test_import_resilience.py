"""Regression tests for FIX 1: a bad manifest directory must not prevent
``import protein_design_mcp`` from succeeding.

``protein_design_mcp/__init__.py`` imports ``server``, which builds the
registry at module scope. Before this fix, a missing manifest directory
(the exact situation an installed wheel/Docker image was in, since
``manifests/`` was not packaged) raised ``ManifestError`` during that import,
so ``pip install protein-design-mcp && protein-design-mcp`` — and even
``describe_tool``, the designed escape hatch — was unreachable.

These tests run in a real subprocess rather than monkeypatching
``os.environ`` in-process, because the failure mode is specifically about
*import time* behaviour: once ``protein_design_mcp`` has been imported once
in this interpreter (as it has, by the time any other test file runs),
re-importing it again would just hit the module cache and prove nothing.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

_SRC = str(Path(__file__).parent.parent / "src")


def _run(code: str, manifest_dir: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PROTEIN_MCP_MANIFEST_DIR"] = manifest_dir
    env["DEVICE"] = "cpu"
    # The subprocess does NOT inherit pytest's sys.path insertion, so a bare
    # ``import protein_design_mcp`` resolves to whatever this interpreter's
    # editable install points at -- which on this machine is a DIFFERENT
    # checkout of this project. That made these tests silently assert against
    # the wrong source tree: they passed while that install pointed here, then
    # failed with ``module 'protein_design_mcp.server' has no attribute
    # '_app'`` once it pointed elsewhere, without a single line of THIS
    # checkout having changed. Pin the path so the subprocess proves something
    # about this tree regardless of what is installed.
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join([_SRC, existing]) if existing else _SRC
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_import_succeeds_when_manifest_dir_is_missing(tmp_path):
    missing = tmp_path / "no-such-manifests"
    result = _run("import protein_design_mcp\nprint('OK')", str(missing))
    assert result.returncode == 0, (
        f"import failed with a missing manifest dir.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "OK" in result.stdout


def test_describe_tool_is_still_listed_when_manifest_dir_is_missing(tmp_path):
    missing = tmp_path / "no-such-manifests"
    code = (
        "import asyncio, json\n"
        "import protein_design_mcp.server as s\n"
        "tools = asyncio.run(s._app.list_tools())\n"
        "print(json.dumps([t.name for t in tools]))\n"
    )
    result = _run(code, str(missing))
    assert result.returncode == 0, (
        f"import/listing failed with a missing manifest dir.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    names = json.loads(result.stdout.strip().splitlines()[-1])
    assert "describe_tool" in names
    # The manifest-backed tools are gone (registry degraded to empty), but
    # the server did not crash and describe_tool remains reachable.
    assert "run_prodigy" not in names
