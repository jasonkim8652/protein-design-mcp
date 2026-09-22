"""Live end-to-end proof that the run_prodigy dispatch contract works.

Drives the REAL ``mcp.server.Server`` request handler registered by
``protein_design_mcp.server`` (not ``ServerApp.call_tool`` directly), so the
call traverses the exact path a real MCP client's ``tools/call`` request
would:

    mcp.server.Server's CallToolRequest handler
      -> ServerApp.call_tool
        -> validate_and_fill
        -> _resolve_path_params
        -> ADAPTERS["run_prodigy"] = (prodigy.build_args, prodigy.parse_output)
        -> EnvDispatcher.run
          -> subprocess: micromamba run -n scoring prodigy <args>
        -> prodigy.parse_output

Intended to run inside the Dockerfile.envs image, in the "server"
micromamba environment, e.g.:

    micromamba run -n server python scripts/live_proof_prodigy.py

Exercises one success path (a real two-chain complex, PDB 1BRS) and three
failure paths (nonexistent input file, a malformed chain ID rejected by our
own validation, and a well-formed but nonexistent chain ID rejected by the
engine itself).
"""

from __future__ import annotations

import asyncio
import json

from mcp import types

from protein_design_mcp import server as pdmcp

FIXTURE_1BRS = "tests/fixtures/test_pdbs/1BRS.pdb"


async def call_tool(name: str, arguments: dict) -> types.CallToolResult:
    """Invoke the tool the same way a real MCP client's request would.

    ``pdmcp.server`` is the ``mcp.server.Server`` instance built in
    server.py; ``@server.call_tool(validate_input=False)`` registered our
    handler under ``types.CallToolRequest`` in ``request_handlers``. This
    fetches that handler and calls it with a real ``CallToolRequest``,
    rather than calling ``ServerApp.call_tool`` (or ``_app.call_tool``)
    directly.
    """
    handler = pdmcp.server.request_handlers[types.CallToolRequest]
    request = types.CallToolRequest(
        method="tools/call",
        params=types.CallToolRequestParams(name=name, arguments=arguments),
    )
    result = await handler(request)
    return result.root


def _print_result(label: str, result: types.CallToolResult) -> None:
    print(f"\n=== {label} ===")
    print(f"isError: {result.isError}")
    for block in result.content:
        if isinstance(block, types.TextContent):
            print(block.text)


async def main() -> None:
    # --- 1. SUCCESS: a real two-chain complex, PRODIGY actually executes ---
    success = await call_tool(
        "run_prodigy",
        {"complex_pdb": FIXTURE_1BRS, "chain_a": "A", "chain_b": "D"},
    )
    _print_result("SUCCESS: run_prodigy on 1BRS chains A/D", success)
    if success.isError:
        raise SystemExit("FATAL: expected the 1BRS run to succeed, it did not")
    payload = json.loads(success.content[0].text)
    if not isinstance(payload.get("binding_affinity_kcal_per_mol"), float):
        raise SystemExit(
            "FATAL: binding_affinity_kcal_per_mol missing or not a float: "
            f"{payload}"
        )

    # --- 2. FAILURE: nonexistent input path ---------------------------------
    missing = await call_tool(
        "run_prodigy",
        {"complex_pdb": "/no/such/file.pdb", "chain_a": "A", "chain_b": "B"},
    )
    _print_result("FAILURE: nonexistent complex_pdb", missing)
    if not missing.isError:
        raise SystemExit("FATAL: expected an error for a nonexistent input path")

    # --- 3. FAILURE: malformed chain ID (caught by OUR validation) ---------
    malformed_chain = await call_tool(
        "run_prodigy",
        {"complex_pdb": FIXTURE_1BRS, "chain_a": "AB", "chain_b": "D"},
    )
    _print_result("FAILURE: malformed (multi-character) chain_a", malformed_chain)
    if not malformed_chain.isError:
        raise SystemExit("FATAL: expected ToolInputError for a multi-character chain id")

    # --- 4. FAILURE: well-formed but nonexistent chain ID (engine-level) ---
    bad_chain = await call_tool(
        "run_prodigy",
        {"complex_pdb": FIXTURE_1BRS, "chain_a": "A", "chain_b": "Z"},
    )
    _print_result("FAILURE: well-formed but nonexistent chain_b='Z'", bad_chain)
    if not bad_chain.isError:
        raise SystemExit("FATAL: expected an error for a chain id absent from the structure")

    print("\n=== ALL CHECKS PASSED ===")


if __name__ == "__main__":
    asyncio.run(main())
