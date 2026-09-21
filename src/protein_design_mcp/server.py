"""
Protein Binder Design MCP Server

Main entry point for the MCP server that exposes protein design tools.
"""

import argparse
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments. Pure — safe to assert on in tests."""
    parser = argparse.ArgumentParser(prog="protein-design-mcp")
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help=(
            "stdio for a local client; http to serve over streamable HTTP so "
            "clients on other machines can reach the GPU host."
        ),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args(argv)


async def run_server(transport: str = "stdio", host: str = "127.0.0.1", port: int = 8765):
    """Run the MCP server over the chosen transport."""
    if transport == "http":
        from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
        import uvicorn

        manager = StreamableHTTPSessionManager(app=server)
        config = uvicorn.Config(
            manager.handle_request, host=host, port=port, log_level="info"
        )
        await uvicorn.Server(config).serve()
        return

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream, server.create_initialization_options()
        )


def main():
    args = parse_args()
    asyncio.run(run_server(args.transport, args.host, args.port))


if __name__ == "__main__":
    main()
