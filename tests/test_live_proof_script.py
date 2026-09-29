"""The live-proof driver covers the declared tool surface on each device.

Coverage is a property of the shipped manifests, independent of which optional
host environments happen to be installed where this test runs. Runtime path
availability is exercised separately by test_manifest_registry.
"""

import sys
from dataclasses import replace
from pathlib import Path

from protein_design_mcp.app import manifest_dir
from protein_design_mcp.manifest.loader import load_manifests
from protein_design_mcp.manifest.registry import ToolRegistry
from protein_design_mcp.meta_tools import DESCRIBE_TOOL_MANIFEST, GET_JOB_STATUS_MANIFEST

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from live_proof import CASES, DEVICES  # noqa: E402


def _registered_tools_by_device() -> dict[str, set[str]]:
    """tool name -> every device (of DEVICES) that registers it.

    ``describe_tool`` and ``get_job_status`` are both device-agnostic
    (``ServerApp.list_tools`` adds them unconditionally, regardless of what
    device its registry was built for), so they map to every entry of
    DEVICES here. (Task 11: this used to add only ``describe_tool``, which
    made ``get_job_status`` permanently uncoverable -- any CASES entry
    naming it would fail ``test_no_case_references_an_unregistered_tool``
    even though it is a real tool ``ServerApp.call_tool`` dispatches.)
    """
    manifests = [
        replace(m, engine=replace(m.engine, prefix=None, prefix_host=None, env="coverage"))
        for m in load_manifests(manifest_dir())
    ]
    registered: dict[str, set[str]] = {}
    for device in DEVICES:
        for tool in ToolRegistry(manifests, device=device).tools():
            registered.setdefault(tool.name, set()).add(device)
    registered.setdefault(DESCRIBE_TOOL_MANIFEST.name, set()).update(DEVICES)
    registered.setdefault(GET_JOB_STATUS_MANIFEST.name, set()).update(DEVICES)
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
