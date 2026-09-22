"""Transport-independent MCP application logic.

Keeping this out of ``server.py`` means the listing and dispatch behaviour can
be tested without standing up a transport.
"""

from __future__ import annotations

import importlib.resources
import json
import logging
import os
from pathlib import Path
from typing import Any

from mcp.types import TextContent, Tool

from protein_design_mcp.adapters import prodigy
from protein_design_mcp.dispatch.env import EngineError, EnvDispatcher
from protein_design_mcp.dispatch.serialize import to_jsonable
from protein_design_mcp.manifest.loader import load_manifests
from protein_design_mcp.manifest.registry import ToolNotAvailable, ToolRegistry
from protein_design_mcp.manifest.schema import ManifestError
from protein_design_mcp.meta_tools import DESCRIBE_TOOL_MANIFEST, describe_tool
from protein_design_mcp.validation import ToolInputError, validate_and_fill

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = float(os.environ.get("PROTEIN_MCP_TIMEOUT", "3600"))

# Engine name -> (build_args, parse_output). Populated per adapter.
ADAPTERS = {
    "prodigy": (prodigy.build_args, prodigy.parse_output),
}


def manifest_dir() -> Path:
    """Directory holding ``*.yaml`` tool manifests.

    Ships as package data at ``protein_design_mcp/manifests/`` so it is
    present in both the wheel and the Docker image (whatever copies
    ``src/`` picks it up automatically). Resolved via ``importlib.resources``
    so it also works from a zipped/namespace install, not just a plain
    directory checkout. ``PROTEIN_MCP_MANIFEST_DIR`` always overrides it.
    """
    override = os.environ.get("PROTEIN_MCP_MANIFEST_DIR")
    if override:
        return Path(override)
    return Path(importlib.resources.files("protein_design_mcp") / "manifests")


def build_registry(device: str = "cuda") -> ToolRegistry:
    """Load manifests from disk into a registry.

    A manifest problem must degrade the server, not prevent import: if the
    directory is missing or a manifest fails to parse, log a loud
    diagnostic naming the directory and serve an empty registry (plus
    ``describe_tool``, which is not manifest-backed) instead of raising.
    """
    directory = manifest_dir()
    try:
        manifests = load_manifests(directory)
    except ManifestError as exc:
        logger.error(
            "Could not load tool manifests from %s: %s. Serving an EMPTY "
            "tool registry (describe_tool is still available). Set "
            "PROTEIN_MCP_MANIFEST_DIR to point at a valid manifest "
            "directory and restart.",
            directory,
            exc,
        )
        manifests = []
    return ToolRegistry(manifests, device=device)


def _error(message: str) -> list[TextContent]:
    return [TextContent(type="text", text=json.dumps({"error": message}, indent=2))]


def _ok(payload: Any) -> list[TextContent]:
    return [
        TextContent(type="text", text=json.dumps(to_jsonable(payload), indent=2))
    ]


class ServerApp:
    """Owns the registry and turns MCP calls into engine runs."""

    def __init__(
        self,
        registry: ToolRegistry,
        dispatcher: EnvDispatcher | None = None,
    ) -> None:
        self._registry = registry
        self._dispatcher = dispatcher or EnvDispatcher()

    async def list_tools(self) -> list[Tool]:
        tools = self._registry.tools()
        tools.append(
            Tool(
                name=DESCRIBE_TOOL_MANIFEST.name,
                description=DESCRIBE_TOOL_MANIFEST.summary,
                inputSchema={
                    "type": "object",
                    "properties": {
                        key: {
                            k: v
                            for k, v in spec.items()
                            if k not in ("required", "example")
                        }
                        for key, spec in DESCRIBE_TOOL_MANIFEST.schema.items()
                    },
                    "required": [],
                    "additionalProperties": False,
                },
            )
        )
        return tools

    async def call_tool(
        self, name: str, arguments: dict[str, Any] | None
    ) -> list[TextContent]:
        arguments = arguments or {}
        logger.info("tool call: %s %s", name, arguments)

        if name == DESCRIBE_TOOL_MANIFEST.name:
            return _ok(
                describe_tool(
                    self._registry,
                    name=arguments.get("name"),
                    category=arguments.get("category"),
                )
            )

        try:
            manifest = self._registry.resolve(name)
        except ToolNotAvailable as exc:
            return _error(str(exc))

        try:
            params = validate_and_fill(manifest, arguments)
        except ToolInputError as exc:
            return _error(str(exc))

        adapter = ADAPTERS.get(manifest.engine.repo)
        if adapter is None:
            return _error(
                f"{name} has no adapter registered for engine "
                f"{manifest.engine.repo!r}"
            )

        build_args, parse_output = adapter
        try:
            run = await self._dispatcher.run(
                manifest.engine, build_args(params), timeout=DEFAULT_TIMEOUT_S
            )
            return _ok(parse_output(run))
        except EngineError as exc:
            return _error(str(exc))
        except ValueError as exc:
            return _error(f"{name}: could not parse engine output: {exc}")
