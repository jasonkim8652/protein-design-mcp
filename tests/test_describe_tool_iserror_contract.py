"""M1: describe_tool must report failures at the protocol level.

`describe_tool` (meta_tools.py) signals failure by returning a plain dict
containing an "error" key -- it never raises. Before this fix,
`ServerApp.call_tool` always wrapped that dict with `_ok(...)`, which the
MCP SDK reports as `isError=False` regardless of what the payload says.
A model that (correctly, per the MCP contract) branches on the
protocol-level `isError` flag before reading the body could not tell a
`describe_tool` failure from a success, and could carry the error object
forward into a later call as if it were real tool metadata.

These tests drive real dispatch through `ServerApp.call_tool` (not the
internal `describe_tool` helper) and assert on the `isError` flag itself,
per each documented failure mode: an unknown tool name, a category with
no available tools (the empty-collection case), and a bad parameter
(neither `name` nor `category` supplied -- explicit absence, not an
explicit `None` value, since the MCP arguments dict simply omits unset
keys). A known-good call is included in the same suite so a broken probe
that always reports isError=True cannot masquerade as a pass (the trap
recorded in the plan-3 carry-forward spec).
"""

from __future__ import annotations

import json

import pytest

from protein_design_mcp.app import ServerApp
from protein_design_mcp.manifest.registry import ToolRegistry


def _text(result):
    if isinstance(result, list):
        return result[0].text
    return result.content[0].text


@pytest.mark.asyncio
async def test_unknown_tool_name_is_a_protocol_error():
    app = ServerApp(ToolRegistry([]))
    result = await app.call_tool("describe_tool", {"name": "run_nonexistent"})
    assert result.isError is True
    payload = json.loads(_text(result))
    assert "error" in payload
    assert "run_nonexistent" in payload["error"]


@pytest.mark.asyncio
async def test_category_with_no_available_tools_is_a_protocol_error():
    """Empty-collection corner case: an entirely empty registry has zero
    tools in any category, including one that is a legitimate value of
    the category enum."""
    app = ServerApp(ToolRegistry([]))
    result = await app.call_tool("describe_tool", {"category": "scoring"})
    assert result.isError is True
    payload = json.loads(_text(result))
    assert "error" in payload
    assert "scoring" in payload["error"]


@pytest.mark.asyncio
async def test_meta_category_self_exclusion_is_also_a_protocol_error():
    """The permanent case called out in the carry-forward spec:
    describe_tool is the only meta tool and excludes itself from its own
    category listing, so category='meta' against an empty registry is
    also an empty-collection failure, not a success."""
    app = ServerApp(ToolRegistry([]))
    result = await app.call_tool("describe_tool", {"category": "meta"})
    assert result.isError is True
    payload = json.loads(_text(result))
    assert "error" in payload


@pytest.mark.asyncio
async def test_bad_parameter_neither_name_nor_category_is_a_protocol_error():
    """Neither key present at all (arguments == {}) -- the MCP arguments
    dict omits unset parameters rather than sending them as an explicit
    null, so this is the 'missing key' corner case, not 'value is None'."""
    app = ServerApp(ToolRegistry([]))
    result = await app.call_tool("describe_tool", {})
    assert result.isError is True
    payload = json.loads(_text(result))
    assert "error" in payload


@pytest.mark.asyncio
async def test_bad_parameter_both_name_and_category_is_a_protocol_error():
    app = ServerApp(ToolRegistry([]))
    result = await app.call_tool(
        "describe_tool", {"name": "describe_tool", "category": "scoring"}
    )
    assert result.isError is True
    payload = json.loads(_text(result))
    assert "error" in payload


@pytest.mark.asyncio
async def test_known_good_call_is_still_not_an_error():
    """Sanity anchor: proves this probe discriminates rather than always
    reporting isError=True. describe_tool documenting itself is real,
    manifest-free metadata (see app.py's DESCRIBE_TOOL_MANIFEST branch),
    so it succeeds even against a totally empty registry."""
    app = ServerApp(ToolRegistry([]))
    result = await app.call_tool("describe_tool", {"name": "describe_tool"})
    assert not (hasattr(result, "isError") and result.isError)
    payload = json.loads(_text(result))
    assert payload["name"] == "describe_tool"
    assert "error" not in payload
