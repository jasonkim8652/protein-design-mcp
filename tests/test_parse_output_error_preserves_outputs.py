"""G2: an adapter's parse_output raising must not discard run.outputs.

`ServerApp.call_tool` already collects `run.outputs` from the dispatcher
before handing the run to the adapter's `parse_output`. If that parser
raises, the exception used to propagate straight to the generic exception
handler, which built an error payload from nothing but the exception
message -- the collected output paths were dropped. Only 2 of the 4
shipped adapters happen to name those paths in their own exception
message, so for the other two (and for all future adapters) a completed,
collected run became unreachable purely because a parser choked on it.

These tests drive real dispatch through `ServerApp.call_tool` with a fake
dispatcher standing in for the subprocess layer (matching the pattern
already used in tests/test_server_wiring.py), and a `parse_output` that
always raises. They assert the returned payload still names the collected
output paths AND still reports that parsing failed -- for both an engine
that collected several outputs and one that collected none at all (the
empty-collection corner case, which must not crash).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import protein_design_mcp.app as app_module
from protein_design_mcp.app import ServerApp
from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.registry import ToolRegistry
from protein_design_mcp.manifest.schema import parse_manifest


def _text(result):
    if isinstance(result, list):
        return result[0].text
    return result.content[0].text


def _manifest(name="run_flaky_parser"):
    return parse_manifest(
        {
            "name": name,
            "category": "scoring",
            "engine": {"repo": "flaky", "env": "e", "entry": ["x"]},
            "summary": "Summary.",
            "doc": "## What this is\nDoc.\n",
            "outputs": [{"name": "structure", "pattern": "*.pdb"}],
            "schema": {},
        }
    )


class _DispatcherWithOutputs:
    """Stands in for the subprocess layer: always 'completes' and reports
    the outputs the dispatcher claims to have collected, regardless of
    what the (never-invoked) subprocess would really have produced."""

    def __init__(self, outputs):
        self._outputs = outputs

    async def run(self, engine, args, *, timeout, outputs=(), workdir=None):
        return CompletedRun(
            returncode=0,
            stdout="",
            stderr="",
            workdir=Path("/tmp"),
            outputs=self._outputs,
        )


def _parse_output_raises(manifest, run):
    raise ValueError("regex did not match engine stdout")


@pytest.mark.asyncio
async def test_parse_failure_preserves_several_collected_outputs(monkeypatch):
    manifest = _manifest()
    monkeypatch.setattr(
        app_module,
        "ADAPTERS",
        {manifest.name: (lambda m, p: [], _parse_output_raises)},
    )
    collected = {
        "structure": "/scratch/pdmcp-abc/out1.pdb",
        "log": ["/scratch/pdmcp-abc/run.log", "/scratch/pdmcp-abc/run2.log"],
    }
    dispatcher = _DispatcherWithOutputs(collected)
    app = ServerApp(ToolRegistry([manifest]), dispatcher=dispatcher)

    result = await app.call_tool(manifest.name, {})

    # Error still clearly reported as a parse failure -- not silently
    # swallowed or turned into a success.
    assert getattr(result, "isError", False) is True
    payload = json.loads(_text(result))
    assert "error" in payload
    assert "regex did not match engine stdout" in payload["error"]

    # But the paths the dispatcher already collected must survive.
    assert payload.get("outputs") == collected


@pytest.mark.asyncio
async def test_parse_failure_with_no_collected_outputs_does_not_crash(monkeypatch):
    """Nothing to preserve (run.outputs == {}) -- must not raise and must
    still report the parse failure clearly, with no stray empty 'outputs'
    noise implying something was collected when nothing was."""
    manifest = _manifest("run_flaky_parser_empty")
    monkeypatch.setattr(
        app_module,
        "ADAPTERS",
        {manifest.name: (lambda m, p: [], _parse_output_raises)},
    )
    dispatcher = _DispatcherWithOutputs({})
    app = ServerApp(ToolRegistry([manifest]), dispatcher=dispatcher)

    result = await app.call_tool(manifest.name, {})

    assert getattr(result, "isError", False) is True
    payload = json.loads(_text(result))
    assert "error" in payload
    assert "regex did not match engine stdout" in payload["error"]
    assert not payload.get("outputs")
