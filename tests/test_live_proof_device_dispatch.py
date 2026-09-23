"""A live-proof case must run against the registry for its OWN declared
device, not whatever device happens to be built first / cached / default.

Before this fix, ``scripts/live_proof.py`` drove every case through one
``mcp.server.Server`` wired to a single ``ServerApp`` built for
``device="cpu"``. A ``requires.gpu: true`` case dispatched that way would
silently be checked against a registry that EXCLUDES it -- either rejected
for the wrong reason (never actually exercising the GPU dispatch path) or,
worse, some other tool of the same name colliding, and either way reported
as if the device it declared did not matter. This proves it does.

Nothing here hardcodes a tool name: the GPU-only tool used is whichever one
the CURRENT manifests happen to have (derived from ``build_registry`` at
test time), so this stays correct as new engines land.
"""

import asyncio
import sys
from pathlib import Path

import pytest
from mcp import types

from protein_design_mcp.app import build_registry

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import live_proof  # noqa: E402


def _a_gpu_only_tool_name() -> str:
    cpu_names = {t.name for t in build_registry(device="cpu").tools()}
    cuda_names = {t.name for t in build_registry(device="cuda").tools()}
    gpu_only = sorted(cuda_names - cpu_names)
    if not gpu_only:
        pytest.skip("no GPU-only tool currently registered to probe with")
    return gpu_only[0]


def test_gpu_only_tool_is_rejected_for_lacking_a_gpu_when_dispatched_as_cpu():
    tool = _a_gpu_only_tool_name()

    result = asyncio.run(live_proof.call_tool(tool, {}, device="cpu"))

    assert result.isError
    text = result.content[0].text
    assert "GPU" in text, f"expected a GPU-availability rejection, got: {text}"


def test_gpu_only_tool_resolves_past_availability_when_dispatched_as_cuda():
    tool = _a_gpu_only_tool_name()

    # Empty arguments: this still fails (missing required parameters), but
    # the FAILURE REASON must not be "requires a GPU" -- that would mean
    # the case was, despite declaring device="cuda", actually checked
    # against the cpu registry, which is exactly the bug this proves is
    # fixed. Empty arguments keep this fast and deterministic: it never
    # reaches the adapter/dispatcher, so no subprocess or GPU hardware is
    # needed to observe the distinction.
    result = asyncio.run(live_proof.call_tool(tool, {}, device="cuda"))

    assert result.isError
    text = result.content[0].text
    assert "GPU" not in text, (
        f"{tool!r} was rejected for lacking a GPU even though it was "
        f"dispatched as device='cuda': {text}"
    )


def test_device_registries_are_actually_different():
    """Sanity check on the fixture the two tests above rely on: the cpu and
    cuda servers built by live_proof must expose different tool sets, or
    the two tests above would not actually be distinguishing anything.
    """
    tool = _a_gpu_only_tool_name()

    cpu_server = live_proof._server_for_device("cpu")
    cuda_server = live_proof._server_for_device("cuda")

    cpu_result = asyncio.run(cpu_server.request_handlers[types.ListToolsRequest](None))
    cuda_result = asyncio.run(cuda_server.request_handlers[types.ListToolsRequest](None))
    cpu_tools = {t.name for t in cpu_result.root.tools}
    cuda_tools = {t.name for t in cuda_result.root.tools}

    assert tool not in cpu_tools
    assert tool in cuda_tools
