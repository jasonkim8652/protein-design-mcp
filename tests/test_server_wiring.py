import json

import pytest

from protein_design_mcp.app import ServerApp, manifest_dir
from protein_design_mcp.manifest.registry import ToolRegistry
from protein_design_mcp.manifest.schema import parse_manifest

MANIFEST_DIR = manifest_dir()


def _composite():
    return parse_manifest(
        {
            "name": "run_boltzgen_run",
            "category": "generation",
            "composite": True,
            "engine": {"repo": "boltzgen", "env": "boltzgen", "entry": ["boltzgen", "run"]},
            "summary": "Full pipeline.",
            "doc": "## What this is\nFull pipeline.\n",
            "schema": {"spec": {"type": "string", "required": True, "example": "s.yaml"}},
        }
    )


@pytest.mark.asyncio
async def test_list_tools_includes_describe_tool():
    app = ServerApp(ToolRegistry([]))
    assert "describe_tool" in {t.name for t in await app.list_tools()}


@pytest.mark.asyncio
async def test_composite_tool_is_absent_from_the_listing():
    app = ServerApp(ToolRegistry([_composite()]))
    assert "run_boltzgen_run" not in {t.name for t in await app.list_tools()}


@pytest.mark.asyncio
async def test_calling_a_composite_tool_by_name_is_refused_with_a_reason():
    app = ServerApp(ToolRegistry([_composite()]))
    payload = json.loads((await app.call_tool("run_boltzgen_run", {}))[0].text)
    assert "composite" in payload["error"]


@pytest.mark.asyncio
async def test_calling_an_unknown_tool_reports_it():
    app = ServerApp(ToolRegistry([]))
    payload = json.loads((await app.call_tool("run_nope", {}))[0].text)
    assert "unknown" in payload["error"]


@pytest.mark.asyncio
async def test_describe_tool_is_dispatched_without_a_subprocess():
    registry = ToolRegistry([])
    app = ServerApp(registry)
    payload = json.loads(
        (await app.call_tool("describe_tool", {"name": "describe_tool"}))[0].text
    )
    assert payload["name"] == "describe_tool"


@pytest.mark.asyncio
async def test_invalid_input_returns_a_correctable_error_not_a_crash():
    registry = ToolRegistry(
        [m for m in _load_real() if m.name == "run_prodigy"]
    )
    app = ServerApp(registry)
    payload = json.loads(
        (await app.call_tool("run_prodigy", {"complex_pdb": "notes.txt",
                                             "chain_a": "A", "chain_b": "B"}))[0].text
    )
    assert "complex_pdb" in payload["error"]
    assert "complex.pdb" in payload["error"]


def _load_real():
    from protein_design_mcp.manifest.loader import load_manifests

    return load_manifests(MANIFEST_DIR)


def test_real_manifests_all_load():
    assert {m.name for m in _load_real()} >= {"run_prodigy"}
