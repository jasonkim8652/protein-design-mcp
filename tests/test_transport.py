import asyncio
import socket
from unittest.mock import patch

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

    This test calls the real production run_server() and verifies it works
    end-to-end. It catches the critical bug where manager.run() is not entered,
    leaving _task_group=None, which causes handle_request() to raise RuntimeError
    on every request.

    Uses a real HTTP request over a real socket (not in-process ASGI call) to
    verify the entire request/response path works in production code.
    """
    import httpx
    import uvicorn

    # Find an available port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    # Capture uvicorn.Server instance from production run_server()
    server_instance = {}
    original_init = uvicorn.Server.__init__

    def init_with_capture(self, *args, **kwargs):
        """Wrapper that captures server instance then calls original __init__."""
        server_instance["server"] = self
        original_init(self, *args, **kwargs)

    # Keep patch active throughout the test
    patcher = patch.object(uvicorn.Server, "__init__", init_with_capture)
    patcher.start()

    try:
        # Start real run_server() in background with patch still active
        server_task = asyncio.create_task(
            run_server(transport="http", host="127.0.0.1", port=port)
        )
    except BaseException:
        patcher.stop()
        raise

    try:
        # Initial small delay to let server start
        await asyncio.sleep(0.1)

        # Retry loop: wait for server to accept connections
        max_retries = 20
        retry_interval = 0.2
        last_error = None

        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient() as client:
                    response = await asyncio.wait_for(
                        client.post(
                            f"http://127.0.0.1:{port}/rpc",
                            json={
                                "jsonrpc": "2.0",
                                "id": 1,
                                "method": "initialize",
                                "params": {},
                            },
                            timeout=1.0,
                        ),
                        timeout=2.0,
                    )
                # Request succeeded
                break
            except (httpx.ConnectError, asyncio.TimeoutError, OSError) as e:
                last_error = e
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_interval)

        if last_error:
            raise AssertionError(
                f"Server did not become ready after "
                f"{max_retries * retry_interval:.1f}s: {last_error}"
            ) from last_error

        # Verify response is not 500 (which indicates RuntimeError in handler)
        assert response.status_code != 500, (
            f"Server returned 500 error. This indicates the critical bug "
            f"'Task group is not initialized. Make sure to use run().' "
            f"Response: {response.text}"
        )
    finally:
        # Graceful shutdown: signal server to exit cleanly
        uv_server = server_instance.get("server")
        if uv_server:
            uv_server.should_exit = True

        try:
            # Wait for server to shut down cleanly with timeout
            await asyncio.wait_for(server_task, timeout=3.0)
        except asyncio.TimeoutError:
            # Fallback: forceful cancellation if graceful shutdown times out
            server_task.cancel()
            try:
                await server_task
            except asyncio.CancelledError:
                pass
        finally:
            patcher.stop()
