"""Guard against modal_proxy.py importing a name server.py no longer exports.

modal_proxy.py drives the `protein-design-mcp-modal` console script
(pyproject.toml) and is not imported by any other test, so a dangling
`from protein_design_mcp.server import TOOLS`-style reference would only
surface when someone actually runs that script. This test makes the import
and the tool listing part of the ordinary suite instead.
"""

import pytest


@pytest.mark.asyncio
async def test_list_tools_returns_a_non_empty_list():
    from protein_design_mcp.modal_proxy import list_tools

    tools = await list_tools()
    assert len(tools) > 0
    assert {t.name for t in tools} >= {"describe_tool"}
