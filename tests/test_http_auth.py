"""The HTTP transport must not be an unauthenticated remote filesystem.

``--transport http`` served every tool to anyone who could reach the port. That
is worse than it sounds, because **every tool takes a caller-supplied
filesystem path**: an unauthenticated request can name any file the server user
can read and have an engine open it, and can spend the GPU indefinitely. The
only thing standing between that and the network was the default bind address
and a warning in ``--help`` -- while the same help text told the reader that
``0.0.0.0`` is how you let other machines reach the GPU host.

The rule implemented here keeps the common case frictionless and the dangerous
case impossible:

* bound to loopback -> no token, nothing changes for a local client;
* bound anywhere else -> a bearer token is required, and if the operator did
  not supply one the server mints one and prints it, so turning on remote
  access is still a single command with no config file to write.
"""

from __future__ import annotations

import pytest

from protein_design_mcp.server import (
    TOKEN_ENV_VAR,
    is_loopback,
    parse_args,
    resolve_http_token,
)


# --- which binds are safe ----------------------------------------------------


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1", "127.5.6.7"])
def test_loopback_addresses_are_recognised(host):
    assert is_loopback(host) is True


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.10", "10.0.0.2", ""])
def test_everything_else_is_not_loopback(host):
    """``""`` and ``::`` are wildcards: uvicorn binds every interface, which is
    exactly the exposure this guards."""
    assert is_loopback(host) is False


# --- when a token is required ------------------------------------------------


def test_a_local_bind_needs_no_token(monkeypatch):
    """The common case stays exactly as convenient as it was."""
    monkeypatch.delenv(TOKEN_ENV_VAR, raising=False)
    token, generated = resolve_http_token("127.0.0.1", env={})
    assert token is None
    assert generated is False


def test_a_public_bind_without_a_configured_token_mints_one():
    """Refusing to start would be the safe answer and a hostile one -- the
    operator would have to go and invent a secret before they could try
    anything. Minting one keeps it a single command."""
    token, generated = resolve_http_token("0.0.0.0", env={})
    assert token
    assert generated is True


def test_a_configured_token_is_used_as_given():
    token, generated = resolve_http_token("0.0.0.0", env={TOKEN_ENV_VAR: "s3cret-from-the-operator"})
    assert token == "s3cret-from-the-operator"
    assert generated is False


def test_a_configured_token_also_applies_to_a_local_bind():
    """An operator who sets a token means it, even on loopback -- a shared
    machine has other users on the same loopback interface."""
    token, _ = resolve_http_token("127.0.0.1", env={TOKEN_ENV_VAR: "abc"})
    assert token == "abc"


def test_a_blank_token_is_treated_as_unset_not_as_an_empty_password():
    """``PROTEIN_DESIGN_MCP_TOKEN=`` is how an env var arrives when something
    upstream failed to set it. Accepting it would authenticate everyone."""
    token, generated = resolve_http_token("0.0.0.0", env={TOKEN_ENV_VAR: "   "})
    assert generated is True
    assert token and token.strip()


def test_a_minted_token_is_long_enough_to_be_worth_minting():
    token, _ = resolve_http_token("0.0.0.0", env={})
    assert len(token) >= 32


def test_two_minted_tokens_differ():
    a, _ = resolve_http_token("0.0.0.0", env={})
    b, _ = resolve_http_token("0.0.0.0", env={})
    assert a != b


# --- the middleware ----------------------------------------------------------


def _run(app, headers, path="/mcp"):
    """Drive one ASGI request and collect the response status and body."""
    import asyncio

    sent: list[dict] = []
    scope = {
        "type": "http",
        "method": "POST",
        "path": path,
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
    }

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    asyncio.run(app(scope, receive, send))
    start = next(m for m in sent if m["type"] == "http.response.start")
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return start["status"], body


@pytest.fixture
def guarded():
    from protein_design_mcp.server import require_bearer_token

    async def inner(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"reached the tools"})

    return require_bearer_token(inner, "the-token")


def test_a_request_with_no_authorization_header_is_refused(guarded):
    status, body = _run(guarded, {})
    assert status == 401
    assert b"reached the tools" not in body


def test_a_request_with_the_wrong_token_is_refused(guarded):
    status, _ = _run(guarded, {"authorization": "Bearer not-the-token"})
    assert status == 401


def test_a_request_with_the_right_token_reaches_the_tools(guarded):
    status, body = _run(guarded, {"authorization": "Bearer the-token"})
    assert status == 200
    assert b"reached the tools" in body


def test_the_scheme_is_matched_case_insensitively(guarded):
    """``bearer`` and ``Bearer`` are both valid per RFC 7235; rejecting one
    would be a confusing failure for a correctly configured client."""
    status, _ = _run(guarded, {"authorization": "bearer the-token"})
    assert status == 200


def test_a_token_without_the_bearer_scheme_is_refused(guarded):
    status, _ = _run(guarded, {"authorization": "the-token"})
    assert status == 401


def test_the_refusal_says_how_to_authenticate_without_leaking_the_token(guarded):
    status, body = _run(guarded, {})
    assert status == 401
    assert b"the-token" not in body, "the 401 body must never echo the expected token"
    assert b"Bearer" in body


def test_no_token_means_no_wrapper_at_all():
    """A loopback server must not pay for a check it does not need, and must
    not accidentally start refusing local clients."""
    from protein_design_mcp.server import require_bearer_token

    async def inner(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    assert require_bearer_token(inner, None) is inner


def test_the_comparison_is_constant_time():
    """A byte-by-byte ``==`` on a secret leaks it to a patient caller."""
    import inspect

    from protein_design_mcp.server import require_bearer_token

    source = inspect.getsource(require_bearer_token)
    assert "compare_digest" in source, "use secrets.compare_digest, not =="


# --- the CLI still says what it does ----------------------------------------


def test_the_default_bind_is_still_loopback():
    assert parse_args([]).host == "127.0.0.1"


def test_a_token_can_be_passed_on_the_command_line():
    assert parse_args(["--auth-token", "xyz"]).auth_token == "xyz"
