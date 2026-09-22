"""The live-proof driver must cover every registered tool.

This runs on the host without Docker: it checks the driver's coverage map
against the registry, so an engine added without a live check fails here
rather than being silently unproven.

"registered" here means the FULL live server surface, not just
``build_registry(...).tools()``: that registry-derived listing covers every
manifest-backed engine, but ``describe_tool`` is a meta-tool with its own
manifest (``meta_tools.DESCRIBE_TOOL_MANIFEST``) that ``ServerApp.list_tools``
adds separately (see src/protein_design_mcp/app.py) and is never part of the
registry. It is still a real tool a client can call through the same
handler, and ``scripts/live_proof.py`` exercises it, so it belongs in the
coverage set too.
"""

import sys
from pathlib import Path

from protein_design_mcp.app import build_registry
from protein_design_mcp.meta_tools import DESCRIBE_TOOL_MANIFEST

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from live_proof import CASES  # noqa: E402


def _registered_tool_names() -> set[str]:
    registry_tools = {t.name for t in build_registry(device="cpu").tools()}
    return registry_tools | {DESCRIBE_TOOL_MANIFEST.name}


def test_every_registered_tool_has_a_live_case():
    registered = _registered_tool_names()
    covered = {case["tool"] for case in CASES}
    missing = registered - covered
    assert not missing, f"tools with no live-proof case: {sorted(missing)}"


def test_no_case_references_an_unregistered_tool():
    registered = _registered_tool_names()
    covered = {case["tool"] for case in CASES}
    assert not covered - registered


def test_every_case_declares_expected_result_keys():
    for case in CASES:
        assert case["expect_keys"], f"{case['tool']} declares no expected keys"
