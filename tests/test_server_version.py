"""The version an MCP client sees must be THIS project's version.

``Server(name)`` with no ``version`` makes the SDK report its own package
version in ``serverInfo`` -- a live stdio handshake against the built image
returned ``{'name': 'protein-design-mcp', 'version': '1.30.0'}``, which is
the `mcp` library's version, not this server's. A client (or a human reading
a bug report) cannot tell v1's composite tool surface from v2's atomistic one
from that string, and 1.30.0 is not a version this project has ever had.
"""

from __future__ import annotations

from importlib.metadata import version as _dist_version

import protein_design_mcp.server as server_module


def test_server_reports_this_projects_version_not_the_sdks():
    declared = _dist_version("protein-design-mcp")
    assert server_module.server.version == declared, (
        f"serverInfo would report {server_module.server.version!r} but this "
        f"distribution is {declared!r}; pass version= to Server(...) so the "
        "handshake identifies the server rather than the MCP SDK."
    )


def test_the_declared_version_is_the_v2_surface():
    """v2 removed every composite tool -- a breaking change, so the major
    version must have moved. Guards against tagging a v2 release while
    pyproject still says 1.0.0 (which it did)."""
    major = int(_dist_version("protein-design-mcp").split(".")[0])
    assert major >= 2, (
        "the atomistic rewrite removed design_binder/predict_complex/"
        "score_stability, so this is not a 1.x release"
    )
