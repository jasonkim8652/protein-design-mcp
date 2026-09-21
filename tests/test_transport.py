import pytest

from protein_design_mcp.server import parse_args


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
