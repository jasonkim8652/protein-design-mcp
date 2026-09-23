"""
Protein Binder Design MCP Server

Main entry point for the MCP server that exposes protein design tools.
"""

import argparse
import asyncio
import logging
import os
import re
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    ResourceTemplate,
    Tool,
)

from protein_design_mcp.app import ServerApp, build_registry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create server instance.
#
# The version is read from the installed distribution rather than written
# here, so exactly one place (pyproject.toml) states it. Passing it matters:
# without it the SDK reports ITS OWN version in the initialize handshake -- a
# live probe of the built image answered `version: '1.30.0'`, the `mcp`
# library's version, which tells a client nothing about whether it is talking
# to v1's composite tools or v2's atomistic ones.
try:
    _VERSION = _dist_version("protein-design-mcp")
except PackageNotFoundError:  # a source tree that was never installed
    _VERSION = "0+unknown"

server = Server("protein-design-mcp", version=_VERSION)

#: Where the NVIDIA character devices appear. Overridable for testing only.
DEV_DIR = Path("/dev")

#: A GPU assigned to this container/host, e.g. ``nvidia0``, ``nvidia7``.
#: Deliberately NOT ``nvidiactl``/``nvidia-uvm``/``nvidia-modeset``, which are
#: control nodes that can exist with no GPU attached.
_GPU_NODE = re.compile(r"^nvidia\d+$")


def detect_device(env: Mapping[str, str] | None = None, dev_dir: Path | None = None) -> str:
    """Resolve ``DEVICE``: an explicit value, else detect from device nodes.

    Detection asks "is a GPU attached to this container", not "can I run CUDA
    from this interpreter". Those came apart in v2: this server never runs
    CUDA itself -- every engine runs in its own environment with its own torch
    -- and the image's ``server`` environment has no torch at all, by design.
    The old ``import torch; torch.cuda.is_available()`` probe therefore hit
    ``ImportError`` and reported ``cpu`` on a correctly GPU-pinned container,
    silently excluding 27 of 39 tools.

    A numbered node under ``/dev`` is the fact we can actually observe, needs
    no dependency, and matches the pinning: ``--device=nvidia.com/gpu=7``
    yields exactly ``/dev/nvidia7``. ``/proc/driver/nvidia/gpus`` is not
    usable -- it is the host driver's procfs and lists every GPU on the
    machine even inside a container pinned to one.
    """
    env = os.environ if env is None else env
    requested = (env.get("DEVICE") or "auto").strip().lower()
    if requested != "auto":
        return requested

    dev_dir = DEV_DIR if dev_dir is None else dev_dir
    try:
        return "cuda" if any(_GPU_NODE.match(p.name) for p in dev_dir.iterdir()) else "cpu"
    except OSError:
        # No /dev to read (an unusual sandbox, or the path does not exist).
        return "cpu"


DEVICE = detect_device()

logger.info(f"Device mode: {DEVICE} (GPU-only tools {'enabled' if DEVICE != 'cpu' else 'disabled'})")


# =============================================================================
# Manifest-driven tool registry
# =============================================================================

_app = ServerApp(build_registry(device=DEVICE))


@server.list_tools()
async def list_tools() -> list[Tool]:
    return await _app.list_tools()


@server.call_tool(validate_input=False)
async def call_tool(name: str, arguments: dict[str, Any]):
    # validate_input=False: the SDK's default jsonschema validation runs
    # BEFORE our handler and, on failure, replaces our message with its own
    # generic "Input validation error: ...". validation.py exists precisely
    # to name the offending parameter, state the constraint, and show a
    # correct example for a model that will read the error and retry — so
    # our validator, not jsonschema, must be the actual boundary.
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
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help=(
            "Bind address. 0.0.0.0 exposes this server to other machines, "
            "but it has no authentication and every tool takes a "
            "caller-supplied filesystem path — put it behind a reverse "
            "proxy or an SSH tunnel, never expose it directly."
        ),
    )
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args(argv)


async def run_server(transport: str = "stdio", host: str = "127.0.0.1", port: int = 8765):
    """Run the MCP server over the chosen transport."""
    if transport == "http":
        import uvicorn
        from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

        manager = StreamableHTTPSessionManager(app=server)
        async with manager.run():
            async def asgi_app(scope, receive, send):
                await manager.handle_request(scope, receive, send)

            config = uvicorn.Config(
                asgi_app, host=host, port=port, log_level="info"
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
