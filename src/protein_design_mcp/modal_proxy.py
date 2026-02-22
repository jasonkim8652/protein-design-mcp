"""
Local MCP proxy that forwards tool calls to a Modal GPU deployment.

This runs locally as a standard MCP server (stdio transport) and proxies
all tool calls to the Modal web endpoint via HTTP POST. File arguments
(PDB paths) are read locally and sent inline.

Usage:
    # Set your Modal endpoint URL (printed after `modal deploy`)
    export MODAL_URL=https://<your-workspace>--protein-design-tools.modal.run

    # Start the proxy as an MCP server
    python -m protein_design_mcp.modal_proxy

Configure in Claude Desktop (claude_desktop_config.json):
    {
      "mcpServers": {
        "protein-design": {
          "command": "python",
          "args": ["-m", "protein_design_mcp.modal_proxy"],
          "env": {
            "MODAL_URL": "https://<your-workspace>--protein-design-tools.modal.run"
          }
        }
      }
    }
"""

import asyncio
import json
import logging
import os
import urllib.request
import urllib.error
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

server = Server("protein-design-mcp-modal")

MODAL_URL = os.environ.get("MODAL_URL", "")

# Arguments that reference local PDB/structure files
_FILE_ARGS = {"target_pdb", "complex_pdb", "pdb_path", "expected_structure"}


# Import tool definitions from main server (all 11 tools — GPU available on Modal)
from protein_design_mcp.server import TOOLS  # noqa: E402


@server.list_tools()
async def list_tools() -> list[Tool]:
    """Return all 11 tools (GPU available on Modal)."""
    return TOOLS


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Forward tool call to Modal endpoint."""
    if not MODAL_URL:
        return [
            TextContent(
                type="text",
                text=json.dumps(
                    {
                        "error": (
                            "MODAL_URL not set. Deploy first:\n"
                            "  modal deploy deploy/modal_app.py\n"
                            "Then set MODAL_URL to the printed endpoint URL."
                        )
                    },
                    indent=2,
                ),
            )
        ]

    # Inline local file contents so they travel over HTTP
    prepared = dict(arguments)
    for key in _FILE_ARGS:
        value = prepared.get(key)
        if value and os.path.isfile(value):
            logger.info(f"Inlining local file: {key}={value}")
            with open(value) as f:
                prepared[f"_file_{key}"] = f.read()

    # POST to Modal web endpoint
    payload = json.dumps({"name": name, "arguments": prepared}).encode()
    req = urllib.request.Request(
        MODAL_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: urllib.request.urlopen(req, timeout=1800),
        )
        result = json.loads(response.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else str(e)
        result = {"error": f"Modal endpoint returned {e.code}: {body}", "tool": name}
    except urllib.error.URLError as e:
        result = {"error": f"Cannot reach Modal endpoint: {e.reason}", "tool": name}
    except Exception as e:
        result = {"error": f"Proxy error: {e}", "tool": name}

    text = json.dumps(result, indent=2)
    if len(text) > 1_000_000:
        text = json.dumps(result, separators=(",", ":"))

    return [TextContent(type="text", text=text)]


async def run_server():
    """Run the MCP proxy server."""
    logger.info(f"Starting Modal proxy (endpoint: {MODAL_URL or 'NOT SET'})")
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
