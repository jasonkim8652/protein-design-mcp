"""
Protein Binder Design MCP Server

Main entry point for the MCP server that exposes protein design tools.
"""

import argparse
import asyncio
import logging
import ipaddress
import os
import re
import secrets
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


#: Environment variable holding the bearer token for the HTTP transport.
TOKEN_ENV_VAR = "PROTEIN_DESIGN_MCP_TOKEN"


def is_loopback(host: str) -> bool:
    """True when ``host`` can only be reached from this machine.

    The wildcards ``0.0.0.0``, ``::`` and ``""`` are explicitly NOT loopback:
    uvicorn binds every interface for each of them, which is the exposure the
    caller is being asked about.
    """
    if not host:
        return False
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def resolve_http_token(host: str, env: Mapping[str, str] | None = None) -> tuple[str | None, bool]:
    """The bearer token to require, and whether this call minted it.

    Returns ``(None, False)`` for a loopback bind with no token configured --
    the local case stays exactly as frictionless as it was.

    For any other bind a token is mandatory. If the operator did not supply
    one, mint it rather than refusing to start: refusing is safe but hostile,
    since the operator would have to go away and invent a secret before they
    could try anything, and the predictable result is that they reach for a
    worse workaround. Minting keeps enabling remote access a single command.

    A blank value is treated as unset. ``PROTEIN_DESIGN_MCP_TOKEN=`` is how an
    environment variable arrives when something upstream failed to set it, and
    honouring it as an empty password would authenticate everyone.
    """
    env = os.environ if env is None else env
    configured = (env.get(TOKEN_ENV_VAR) or "").strip()
    if configured:
        # Honoured even on loopback: an operator who sets a token means it, and
        # a shared machine has other users on the same loopback interface.
        return configured, False
    if is_loopback(host):
        return None, False
    return secrets.token_urlsafe(32), True


def require_bearer_token(app: Any, token: str | None) -> Any:
    """Wrap an ASGI app so every request must carry ``Authorization: Bearer``.

    With no token this returns ``app`` unchanged, so a loopback server does not
    pay for a check it does not need and cannot accidentally start refusing
    local clients.
    """
    if not token:
        return app

    async def guarded(scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await app(scope, receive, send)
            return

        presented = ""
        for key, value in scope.get("headers", []):
            if key == b"authorization":
                presented = value.decode("latin-1", "replace")
                break

        scheme, _, candidate = presented.partition(" ")
        # compare_digest, never ==: a byte-by-byte comparison on a secret leaks
        # it to a caller patient enough to time the responses.
        if scheme.lower() != "bearer" or not secrets.compare_digest(candidate.strip(), token):
            body = (
                b'{"error":"unauthorized: this server requires '
                b'Authorization: Bearer <token>"}'
            )
            await send({
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"www-authenticate", b'Bearer realm="protein-design-mcp"'),
                    (b"content-length", str(len(body)).encode()),
                ],
            })
            await send({"type": "http.response.body", "body": body})
            return

        await app(scope, receive, send)

    return guarded


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
            "Bind address. 0.0.0.0 exposes this server to other machines; a "
            "bearer token is then required (see --auth-token), because every "
            "tool takes a caller-supplied filesystem path. A token is not a "
            "substitute for a reverse proxy with TLS on an untrusted network: "
            "plain HTTP sends it in the clear."
        ),
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--auth-token",
        default=None,
        help=(
            f"Bearer token clients must present. Defaults to ${TOKEN_ENV_VAR}. "
            "Required for any non-loopback bind; one is generated and printed "
            "if you do not supply it. Prefer the environment variable — a "
            "token on the command line is visible in `ps`."
        ),
    )
    return parser.parse_args(argv)


async def run_server(
    transport: str = "stdio",
    host: str = "127.0.0.1",
    port: int = 8765,
    auth_token: str | None = None,
):
    """Run the MCP server over the chosen transport."""
    if transport == "http":
        import uvicorn
        from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

        if auth_token:
            token, minted = auth_token, False
        else:
            token, minted = resolve_http_token(host)

        if minted:
            # Printed once, prominently. The operator asked to be reachable
            # from the network and needs the credential to do anything; hiding
            # it in a log line they might not see would just push them back to
            # running unauthenticated.
            logger.warning(
                "%s is not a loopback address, so this server requires a bearer "
                "token. No %s was set, so one was generated for this run:\n\n"
                "    %s\n\n"
                "Clients must send `Authorization: Bearer <token>`. Set %s to "
                "keep the same token across restarts. Plain HTTP sends it in "
                "the clear -- put a TLS reverse proxy in front on any network "
                "you do not control.",
                host, TOKEN_ENV_VAR, token, TOKEN_ENV_VAR,
            )
        elif token:
            logger.info("HTTP transport requires a bearer token (from %s).", TOKEN_ENV_VAR)
        else:
            logger.info("HTTP transport bound to %s (loopback); no token required.", host)

        manager = StreamableHTTPSessionManager(app=server)
        async with manager.run():
            async def asgi_app(scope, receive, send):
                await manager.handle_request(scope, receive, send)

            config = uvicorn.Config(
                require_bearer_token(asgi_app, token), host=host, port=port, log_level="info"
            )
            await uvicorn.Server(config).serve()
        return

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream, server.create_initialization_options()
        )


def main():
    args = parse_args()
    asyncio.run(run_server(args.transport, args.host, args.port, args.auth_token))


if __name__ == "__main__":
    main()
