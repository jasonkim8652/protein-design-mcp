"""
Protein Binder Design MCP Server

Main entry point for the MCP server that exposes protein design tools.
"""

import asyncio
import logging
import os
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    Resource,
    ResourceTemplate,
)

from protein_design_mcp.app import ServerApp, build_registry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create server instance
server = Server("protein-design-mcp")

# Device detection: "auto" checks for CUDA availability, "cpu" forces CPU mode
_DEVICE_ENV = os.environ.get("DEVICE", "auto").lower()
if _DEVICE_ENV == "auto":
    try:
        import torch
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        DEVICE = "cpu"
else:
    DEVICE = _DEVICE_ENV

logger.info(f"Device mode: {DEVICE} (GPU-only tools {'enabled' if DEVICE != 'cpu' else 'disabled'})")


# =============================================================================
# Manifest-driven tool registry
# =============================================================================

_app = ServerApp(build_registry(device=DEVICE))


@server.list_tools()
async def list_tools() -> list[Tool]:
    return await _app.list_tools()


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    return await _app.call_tool(name, arguments)


# =============================================================================
# Resources
# =============================================================================


@server.list_resource_templates()
async def list_resource_templates() -> list[ResourceTemplate]:
    """Return list of available resource templates."""
    return [
        ResourceTemplate(
            uriTemplate="protein://structures/{pdb_id}",
            name="PDB Structure",
            description="Access PDB structures by ID",
        ),
        ResourceTemplate(
            uriTemplate="protein://designs/{job_id}/{design_id}",
            name="Design Result",
            description="Access generated design files",
        ),
    ]


# =============================================================================
# Main Entry Point
# =============================================================================


async def run_server():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def main():
    """Main entry point."""
    asyncio.run(run_server())


if __name__ == "__main__":
    main()
