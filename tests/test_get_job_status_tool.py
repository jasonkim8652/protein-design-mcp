"""Wave H: get_job_status wiring.

get_job_status has no engine (see job_status.py's docstring and the wave
report for why the manifest+adapter+scripts/engines three-file pattern does
not fit it). It is wired the same way describe_tool already is: a manifest
object built in code (never loaded from manifests/*.yaml), special-cased in
ServerApp.list_tools()/call_tool() before the registry or ADAPTERS are ever
consulted. These tests mirror tests/test_describe_tool.py and the
describe_tool-specific cases in tests/test_server_wiring.py.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from protein_design_mcp.app import ServerApp
from protein_design_mcp.manifest.registry import ToolRegistry
from protein_design_mcp.meta_tools import GET_JOB_STATUS_MANIFEST
from protein_design_mcp.utils.job_queue import JobProgress, JobQueue, JobStatus


def _text(result):
    if isinstance(result, list):
        return result[0].text
    return result.content[0].text


@pytest.mark.asyncio
async def test_list_tools_includes_get_job_status():
    app = ServerApp(ToolRegistry([]))
    assert "get_job_status" in {t.name for t in await app.list_tools()}


@pytest.mark.asyncio
async def test_get_job_status_is_dispatched_without_a_subprocess(tmp_path):
    """No manifest names 'get_job_status' in the registry, and no adapter
    module exists for it -- if this were routed through the ordinary
    registry.resolve()/ADAPTERS path it would fail with "unknown tool"."""
    queue = JobQueue(storage_dir=tmp_path)
    job_id = queue.create_job()

    app = ServerApp(ToolRegistry([]))
    with patch("protein_design_mcp.job_status.get_job_queue", return_value=queue):
        result = await app.call_tool("get_job_status", {"job_id": job_id})

    payload = json.loads(_text(result))
    assert payload["status"] == "queued"
    assert payload["job_id"] == job_id


@pytest.mark.asyncio
async def test_get_job_status_running_job_includes_progress(tmp_path):
    queue = JobQueue(storage_dir=tmp_path)
    job_id = queue.create_job()
    queue.update_status(job_id, JobStatus.RUNNING)
    queue.update_progress(
        job_id,
        JobProgress(current_step="run_rfdiffusion_binder", designs_completed=0, total_designs=5),
    )

    app = ServerApp(ToolRegistry([]))
    with patch("protein_design_mcp.job_status.get_job_queue", return_value=queue):
        result = await app.call_tool("get_job_status", {"job_id": job_id})

    payload = json.loads(_text(result))
    assert payload["status"] == "running"
    assert payload["progress"]["designs_completed"] == 0
    # Zero designs completed so far -- no rate can be computed yet.
    assert payload.get("estimated_time_remaining") is None


@pytest.mark.asyncio
async def test_get_job_status_unknown_job_is_a_protocol_error(tmp_path):
    queue = JobQueue(storage_dir=tmp_path)

    app = ServerApp(ToolRegistry([]))
    with patch("protein_design_mcp.job_status.get_job_queue", return_value=queue):
        result = await app.call_tool("get_job_status", {"job_id": "nonexistent"})

    assert result.isError is True
    payload = json.loads(_text(result))
    assert "not found" in payload["error"].lower()


@pytest.mark.asyncio
async def test_get_job_status_is_routed_through_validate_and_fill():
    """Missing job_id (required, no default) must be a correctable error,
    not a KeyError."""
    app = ServerApp(ToolRegistry([]))
    result = await app.call_tool("get_job_status", {})

    assert result.isError is True
    payload = json.loads(_text(result))
    assert "job_id" in payload["error"]


@pytest.mark.asyncio
async def test_get_job_status_rejects_unexpected_parameters():
    app = ServerApp(ToolRegistry([]))
    result = await app.call_tool("get_job_status", {"job_id": "x", "bogus": "y"})

    assert result.isError is True
    payload = json.loads(_text(result))
    assert "bogus" in payload["error"] or "unexpected" in payload["error"]


@pytest.mark.asyncio
async def test_describe_tool_can_describe_get_job_status():
    """Regression: describe_tool(name="get_job_status") used to fail with
    "unknown tool" because get_job_status, like describe_tool itself, is
    never loaded into the registry's own manifest set -- caught by driving
    this live through the real describe_tool path, not just unit-testing
    get_job_status in isolation."""
    app = ServerApp(ToolRegistry([]))
    result = await app.call_tool("describe_tool", {"name": "get_job_status"})

    payload = json.loads(_text(result))
    assert "error" not in payload
    assert payload["name"] == "get_job_status"
    assert "job_id" in payload["parameters"]


def test_the_meta_tool_manifest_is_itself_valid():
    assert GET_JOB_STATUS_MANIFEST.name == "get_job_status"
    assert GET_JOB_STATUS_MANIFEST.category == "meta"
    assert GET_JOB_STATUS_MANIFEST.composite is False


@pytest.mark.asyncio
async def test_get_job_status_schema_matches_the_registry_derivation():
    """The meta-tool's schema must come from the same code path as every
    other tool's -- same regression test shape as describe_tool's own."""
    from protein_design_mcp.manifest.registry import json_schema_for

    derived = json_schema_for(GET_JOB_STATUS_MANIFEST)
    assert derived["additionalProperties"] is False
    assert derived["required"] == ["job_id"]

    app = ServerApp(ToolRegistry([]))
    tools = await app.list_tools()
    tool = next(t for t in tools if t.name == "get_job_status")
    assert tool.inputSchema == derived
