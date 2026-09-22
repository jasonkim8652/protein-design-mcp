"""Regression tests for FIX 3.

mcp 1.25's ``Server.call_tool`` defaults to ``validate_input=True`` and runs
jsonschema against our derived schema BEFORE our handler runs, so a client
would see jsonschema's generic "Input validation error: ..." instead of the
message ``validation.py`` was written to produce (naming the parameter, the
constraint, and a correct example). ``tests/test_server_wiring.py`` calls
``ServerApp`` directly, which bypasses the SDK entirely and so could not have
caught this — these tests drive the REAL ``mcp.server.Server`` request
handler instead, exactly as a live transport would.
"""

from __future__ import annotations

import json

import pytest
from mcp import types

from protein_design_mcp.server import server


async def _call(name: str, arguments: dict) -> types.CallToolResult:
    handler = server.request_handlers[types.CallToolRequest]
    request = types.CallToolRequest(
        params=types.CallToolRequestParams(name=name, arguments=arguments)
    )
    result = await handler(request)
    assert isinstance(result, types.ServerResult)
    root = result.root
    assert isinstance(root, types.CallToolResult)
    return root


@pytest.mark.asyncio
async def test_real_server_surfaces_our_validation_message_with_its_example():
    """Drives the actual wired-up mcp.server.Server, not ServerApp. Before
    FIX 3, the SDK's own jsonschema validation intercepted this call and
    returned its generic message instead of ours."""
    result = await _call(
        "run_prodigy",
        {"complex_pdb": "notes.txt", "chain_a": "A", "chain_b": "B"},
    )
    text = result.content[0].text
    assert "Input validation error" not in text
    payload = json.loads(text)
    assert "complex_pdb" in payload["error"]
    assert "complex.pdb" in payload["error"]  # the manifest's own example


@pytest.mark.asyncio
async def test_real_server_marks_a_validation_failure_as_isError():
    result = await _call(
        "run_prodigy",
        {"complex_pdb": "notes.txt", "chain_a": "A", "chain_b": "B"},
    )
    assert result.isError is True


@pytest.mark.asyncio
async def test_real_server_does_not_mark_success_as_isError():
    result = await _call("describe_tool", {"name": "describe_tool"})
    assert result.isError in (False, None)


@pytest.mark.asyncio
async def test_real_server_routes_describe_tool_through_our_validator():
    """describe_tool is dispatched before validate_and_fill was wired up to
    it, so with SDK schema validation off it had no input validation at
    all. An unexpected parameter must now be rejected with our message."""
    result = await _call("describe_tool", {"bogus_param": "x"})
    assert result.isError is True
    payload = json.loads(result.content[0].text)
    assert "bogus_param" in payload["error"] or "unexpected" in payload["error"]
