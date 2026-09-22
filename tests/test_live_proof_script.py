"""The live-proof driver must cover every registered tool, on every device.

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

The tool surface is DEVICE-DEPENDENT: ``build_registry(device="cpu")``
excludes every ``requires.gpu: true`` manifest (see
``ToolRegistry._exclusion_reason``), so a coverage check pinned to one
device can never see a GPU-only tool at all -- it would silently miss the
majority of the server as GPU engines land. So "registered" is the UNION of
what ``build_registry`` returns for every device ``scripts/live_proof.py``
itself knows how to dispatch against (``live_proof.DEVICES``), computed
independently of ``live_proof._registered_tool_names_by_device`` (not by
importing and reusing it) so a bug in that helper cannot hide itself from
both the script's own runtime check and this host-side one.

Nothing here hardcodes a tool name: the expected set is built from the
registry at import time, so it tracks whatever manifests currently exist
instead of breaking the moment a wave of new tools lands.
"""

import sys
from pathlib import Path

from protein_design_mcp.app import build_registry
from protein_design_mcp.meta_tools import DESCRIBE_TOOL_MANIFEST

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from live_proof import CASES, DEVICES  # noqa: E402


def _registered_tools_by_device() -> dict[str, set[str]]:
    """tool name -> every device (of DEVICES) that registers it.

    ``describe_tool`` is device-agnostic (``ServerApp.list_tools`` adds it
    unconditionally, regardless of what device its registry was built for),
    so it maps to every entry of DEVICES here.
    """
    registered: dict[str, set[str]] = {}
    for device in DEVICES:
        for tool in build_registry(device=device).tools():
            registered.setdefault(tool.name, set()).add(device)
    registered.setdefault(DESCRIBE_TOOL_MANIFEST.name, set()).update(DEVICES)
    return registered


def test_every_registered_tool_has_a_live_case():
    registered = _registered_tools_by_device()
    covered = {case["tool"] for case in CASES}
    missing = registered.keys() - covered
    detail = sorted(
        f"{name} (device={sorted(registered[name])})" for name in missing
    )
    assert not missing, f"tools with no live-proof case: {detail}"


def test_no_case_references_an_unregistered_tool():
    registered = _registered_tools_by_device()
    covered = {case["tool"] for case in CASES}
    assert not covered - registered.keys()


def test_every_case_declares_expected_result_keys():
    for case in CASES:
        assert case["expect_keys"], f"{case['tool']} declares no expected keys"


def test_every_case_declares_a_known_device():
    for case in CASES:
        assert case.get("device") in ("cpu", "cuda"), (
            f"{case['tool']} does not declare a known device: "
            f"{case.get('device')!r}"
        )


def test_every_case_declares_a_device_where_its_tool_is_registered():
    """A case's declared device must be one the tool is actually available
    on -- otherwise the case would run against a registry that excludes the
    tool for an unrelated (device) reason, and the resulting failure would
    look like a broken engine rather than a mislabelled case.
    """
    registered = _registered_tools_by_device()
    for case in CASES:
        tool, device = case["tool"], case.get("device")
        assert device in registered.get(tool, set()), (
            f"{tool} case declares device={device!r}, but {tool} is only "
            f"registered on {sorted(registered.get(tool, set()))}"
        )
