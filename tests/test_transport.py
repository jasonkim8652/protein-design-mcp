import asyncio
import socket

import pytest

from protein_design_mcp.server import parse_args, run_server


def test_stdio_is_the_default():
    assert parse_args([]).transport == "stdio"


def test_http_transport_can_be_selected():
    assert parse_args(["--transport", "http"]).transport == "http"


def test_http_has_a_default_host_and_port():
    args = parse_args(["--transport", "http"])
    assert args.host == "127.0.0.1"
    assert args.port == 8765


def test_host_and_port_are_overridable():
    args = parse_args(["--transport", "http", "--host", "0.0.0.0", "--port", "9000"])
    assert args.host == "0.0.0.0"
    assert args.port == 9000


def test_unknown_transport_is_rejected():
    with pytest.raises(SystemExit):
        parse_args(["--transport", "carrier-pigeon"])


def test_mcp_is_pinned_below_v2():
    from pathlib import Path

    text = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    assert "mcp>=1.25,<2" in text
    assert "mcp>=0.1.0" not in text


@pytest.mark.asyncio
async def test_http_transport_serves_requests():
    """
    Test that HTTP transport actually serves requests without RuntimeError.

    This test catches the critical bug where manager.run() is not entered,
    leaving _task_group=None, which causes handle_request() to raise RuntimeError
    on every request.
    """
    import httpx

    # Find an available port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    # Start run_server in background
    server_task = asyncio.create_task(run_server(transport="http", host="127.0.0.1", port=port))

    try:
        # Wait for server to be ready (with timeout)
        await asyncio.sleep(0.5)

        # Make an HTTP request to the server
        async with httpx.AsyncClient() as client:
            # POST to /rpc endpoint (standard MCP path)
            response = await asyncio.wait_for(
                client.post(
                    f"http://127.0.0.1:{port}/rpc",
                    json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
                    timeout=2.0,
                ),
                timeout=5.0,
            )

            # Should NOT be 500 (which would happen if manager._task_group is None)
            # A 400/401/other client error is ok; what matters is no RuntimeError crash
            assert response.status_code != 500, (
                f"Server returned 500 error. Check logs for: "
                f"'Task group is not initialized. Make sure to use run().' "
                f"Response: {response.text}"
            )
    finally:
        # Always cancel the server task
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass
