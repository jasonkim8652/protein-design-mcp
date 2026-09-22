"""Regression tests for FIX 5: excluded tools must be visible at startup.

build_registry() never passed available_weights or licensed to
ToolRegistry, so both defaulted to empty and any manifest declaring
requires.weights or requires.license_gated was unconditionally excluded.
ToolRegistry already stored the reason per tool (registry.excluded()), but
nothing ever logged it, so a tool vanishing from the listing looked
mysterious. build_registry must now log the full exclusion table.
"""

from __future__ import annotations

import logging
import textwrap

import pytest

from protein_design_mcp.app import _log_exclusions, build_registry
from protein_design_mcp.manifest.registry import ToolRegistry
from protein_design_mcp.manifest.schema import parse_manifest


def _gpu_manifest(name="run_needs_gpu"):
    return parse_manifest(
        {
            "name": name,
            "category": "generation",
            "engine": {"repo": "r", "env": "e", "entry": ["x"]},
            "summary": "Summary.",
            "doc": "## What this is\nDoc.\n",
            "schema": {},
            "requires": {"gpu": True},
        }
    )


def _weights_manifest(name="run_needs_weights"):
    return parse_manifest(
        {
            "name": name,
            "category": "generation",
            "engine": {"repo": "r", "env": "e", "entry": ["x"]},
            "summary": "Summary.",
            "doc": "## What this is\nDoc.\n",
            "schema": {},
            "requires": {"weights": "ckpts/model.pt"},
        }
    )


def test_log_exclusions_is_silent_when_nothing_is_excluded(caplog):
    manifest = _gpu_manifest()
    registry = ToolRegistry([manifest], device="cuda")  # GPU available, not excluded
    with caplog.at_level(logging.WARNING, logger="protein_design_mcp.app"):
        _log_exclusions(registry, [manifest])
    assert caplog.text == ""


def test_log_exclusions_names_the_tool_and_the_reason(caplog):
    manifest = _gpu_manifest("run_needs_gpu")
    registry = ToolRegistry([manifest], device="cpu")  # excluded: no GPU
    with caplog.at_level(logging.WARNING, logger="protein_design_mcp.app"):
        _log_exclusions(registry, [manifest])
    assert "run_needs_gpu" in caplog.text
    assert "GPU" in caplog.text


def test_log_exclusions_covers_the_weights_requirement_too(caplog):
    """The exact case the review calls out: a manifest declaring
    requires.weights is unconditionally excluded because build_registry
    never supplies available_weights."""
    manifest = _weights_manifest("run_needs_weights")
    registry = ToolRegistry([manifest])  # available_weights defaults to empty
    with caplog.at_level(logging.WARNING, logger="protein_design_mcp.app"):
        _log_exclusions(registry, [manifest])
    assert "run_needs_weights" in caplog.text
    assert "weights" in caplog.text


def test_log_exclusions_lists_every_excluded_tool_not_just_one(caplog):
    gpu = _gpu_manifest("run_a")
    weights = _weights_manifest("run_b")
    registry = ToolRegistry([gpu, weights], device="cpu")
    with caplog.at_level(logging.WARNING, logger="protein_design_mcp.app"):
        _log_exclusions(registry, [gpu, weights])
    assert "run_a" in caplog.text
    assert "run_b" in caplog.text


@pytest.mark.asyncio
async def test_build_registry_logs_exclusions_at_startup(tmp_path, monkeypatch, caplog):
    manifest_yaml = textwrap.dedent(
        """\
        name: run_needs_gpu
        category: generation
        engine: {repo: r, env: e, entry: [x]}
        summary: Summary.
        doc: |
          ## What this is
          Doc.
        schema: {}
        requires:
          gpu: true
        """
    )
    (tmp_path / "run_needs_gpu.yaml").write_text(manifest_yaml)
    monkeypatch.setenv("PROTEIN_MCP_MANIFEST_DIR", str(tmp_path))

    with caplog.at_level(logging.WARNING, logger="protein_design_mcp.app"):
        registry = build_registry(device="cpu")

    assert registry.excluded("run_needs_gpu") is not None
    assert "run_needs_gpu" in caplog.text
