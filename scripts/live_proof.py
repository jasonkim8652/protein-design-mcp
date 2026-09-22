"""Live end-to-end proof that EVERY registered tool's dispatch contract works.

Generalises ``live_proof_prodigy.py`` (which proved exactly one engine,
PRODIGY) into a table-driven driver covering every engine the manifest
registry exposes, plus the ``describe_tool`` meta-tool. Drives the REAL
``mcp.server.Server`` request handler registered by ``protein_design_mcp.server``
(not ``ServerApp.call_tool`` directly), so every call traverses the exact path
a real MCP client's ``tools/call`` request would:

    mcp.server.Server's CallToolRequest handler
      -> ServerApp.call_tool
        -> validate_and_fill
        -> _resolve_path_params
        -> ADAPTERS[tool_name] = (build_args, parse_output)
        -> EnvDispatcher.run
          -> subprocess: micromamba run -n <env> <entry> <args>
        -> parse_output

Intended to run inside the Dockerfile.envs image, in the "server" micromamba
environment, e.g.:

    micromamba run -n server python scripts/live_proof.py

Before running any case, this script itself discovers every tool the real
server would list (``ServerApp.list_tools()`` — every manifest-backed tool
the registry makes available for this DEVICE, plus ``describe_tool``) and
fails loudly if ``CASES`` does not cover all of them. That is what makes
adding a new engine WITHOUT adding a live case visible here rather than
silent: ``tests/test_live_proof_script.py`` runs the equivalent check on the
host, without Docker, so the gap shows up in CI too.

Each case names a tool, the arguments to call it with, and the keys expected
in the parsed JSON payload on success. A case whose ``expect_keys`` is
``["error"]`` is an EXPECTED-FAILURE case: the driver asserts ``isError`` is
True and that the payload carries an ``error`` message, rather than asserting
success. No case currently uses this — every registered tool now has a
proven success path — but the mechanism stays, since a genuinely unproven
success path (as ``run_ipsae``'s was, until the staging fix below) is a real
state a future engine can land in, and it must be reported honestly rather
than silently omitted or faked.
"""

from __future__ import annotations

import asyncio
import json

from mcp import types

from protein_design_mcp import server as pdmcp

CASES: list[dict] = [
    {
        "tool": "run_prodigy",
        "arguments": {
            "complex_pdb": "tests/fixtures/test_pdbs/1BRS.pdb",
            "chain_a": "A",
            "chain_b": "D",
        },
        "expect_keys": ["binding_affinity_kcal_per_mol", "intermolecular_contacts"],
    },
    {
        # SETTLED LIVE (Task 7): mini_protein.pdb (used successfully by
        # run_mpnn below, which only needs backbone coordinates) is NOT
        # usable here. OpenMM's ForceField.createSystem requires every
        # residue's HEAVY ATOM SET to exactly match a known template, and
        # mini_protein.pdb's MET/LYS/VAL residues carry only N/CA/C/O/CB —
        # real side chains are truncated — so it fails with "No template
        # found for residue 0 (MET) ... matches NALA, but the residue is
        # missing 1 H atom." A real experimental fragment (1BRS chain D,
        # barstar) hits the same class of error from ordinary
        # crystallographic side-chain disorder (GLN 58 is missing its distal
        # OE1/NE2 in the deposited structure) — real PDB files routinely
        # need exactly the kind of gap-filling PDBFixer performs, which is
        # why it is pinned into the "md" environment even though today's
        # openmm_minimize.py script does not call it yet. This case instead
        # uses tests/fixtures/test_pdbs/mini_protein_complete.pdb, a small
        # synthetic poly-Gly/Ala chain built with every heavy atom its
        # residues require (GLY: N,CA,C,O; ALA: +CB, +OXT on the terminal
        # residue) — confirmed live to minimise cleanly.
        "tool": "run_openmm_minimize",
        "arguments": {
            "input_pdb": "tests/fixtures/test_pdbs/mini_protein_complete.pdb",
            "max_iterations": 50,
        },
        "expect_keys": ["energy_change_kj_mol", "outputs"],
    },
    {
        "tool": "run_mpnn",
        "arguments": {
            "backbone_pdb": "tests/fixtures/test_pdbs/mini_protein.pdb",
            "num_sequences": 2,
        },
        "expect_keys": ["designs", "num_designs"],
    },
    {
        # SETTLED LIVE (Task 7 fix round 1): originally reduced to a
        # failure-path case because the Task 6 adapter parsed run.stdout,
        # but ipsae==1.0.1's only entry point (ipsae.cli:main) never prints
        # its results table to stdout at all — it WRITES three files next
        # to the structure file. Fixed with a declarative staging
        # mechanism: the manifest's `engine.stage: ["structure"]` makes the
        # dispatcher copy the structure file into the scratch working
        # directory BEFORE the engine runs (see
        # protein_design_mcp.staging.stage_inputs), so "beside the input"
        # becomes "inside the scratch directory", where the manifest's new
        # `results_txt` output (multiple: true, since a by-residue detail
        # file lands next to it too) can collect it. The adapter now reads
        # that file instead of stdout, using the exact same header-name
        # column lookup as before. tests/fixtures/pae/example_pae.json is a
        # synthetic-but-structurally-valid PAE JSON (a 5x5 matrix matching
        # two_chain_complex.pdb's 5 residues) that ipsae==1.0.1 accepts
        # without complaint — confirmed live, this is now a genuine SUCCESS
        # case, not a reduced one.
        "tool": "run_ipsae",
        "arguments": {
            "pae_json": "tests/fixtures/pae/example_pae.json",
            "structure": "tests/fixtures/test_pdbs/two_chain_complex.pdb",
        },
        "expect_keys": ["ipsae", "chain_pair"],
    },
    {
        "tool": "describe_tool",
        "arguments": {"category": "scoring"},
        "expect_keys": ["tools"],
    },
]


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


async def _check_coverage() -> None:
    """Fail loudly if CASES does not exactly match the live server surface.

    Mirrors tests/test_live_proof_script.py's host-side check, but runs it
    against the REAL ``ServerApp.list_tools()`` this process is about to
    call through — so a manifest excluded at runtime for this DEVICE (e.g. a
    GPU-only tool on a CPU image) can never silently go untested, and
    neither can a case left over for a tool that no longer exists.
    """
    tools = await pdmcp._app.list_tools()
    registered = {t.name for t in tools}
    covered = {case["tool"] for case in CASES}

    missing = registered - covered
    if missing:
        raise SystemExit(
            f"FATAL: tool(s) with no live-proof case: {sorted(missing)}"
        )
    extra = covered - registered
    if extra:
        raise SystemExit(
            f"FATAL: CASES references unregistered tool(s): {sorted(extra)}"
        )


async def _run_case(case: dict) -> None:
    tool = case["tool"]
    expect_keys = case["expect_keys"]
    expect_error = expect_keys == ["error"]

    result = await call_tool(tool, case["arguments"])
    _print_result(tool, result)

    if expect_error:
        if not result.isError:
            raise SystemExit(f"FATAL: expected {tool!r} to fail, it did not")
    else:
        if result.isError:
            raise SystemExit(f"FATAL: expected {tool!r} to succeed, it did not")

    payload = json.loads(result.content[0].text)
    missing_keys = [key for key in expect_keys if key not in payload]
    if missing_keys:
        raise SystemExit(
            f"FATAL: {tool!r} result missing expected key(s) {missing_keys}: {payload}"
        )


async def main() -> None:
    await _check_coverage()

    for case in CASES:
        await _run_case(case)

    print("\n=== ALL CHECKS PASSED ===")


if __name__ == "__main__":
    asyncio.run(main())
