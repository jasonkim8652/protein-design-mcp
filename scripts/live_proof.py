"""Live end-to-end proof that EVERY registered tool's dispatch contract works.

Generalises ``live_proof_prodigy.py`` (which proved exactly one engine,
PRODIGY) into a table-driven driver covering every engine the manifest
registry exposes, plus the ``describe_tool`` meta-tool. Drives a REAL
``mcp.server.Server`` request handler -- built the same way
``protein_design_mcp.server`` builds its own (``Server`` + ``ServerApp`` +
``build_registry``, see ``_server_for_device`` below) -- so every call
traverses the exact path a real MCP client's ``tools/call`` request would:

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

THE TOOL SURFACE IS DEVICE-DEPENDENT: ``build_registry(device="cpu")``
excludes every ``requires.gpu: true`` manifest (see
``ToolRegistry._exclusion_reason``), so a single-device coverage check can
never see a GPU-only tool -- it would either block that tool from ever being
added to ``CASES`` (breaking the coverage assertion the moment it tried), or,
left out, silently prove nothing about it. So each case in ``CASES``
DECLARES the device it needs (``"device": "cpu"`` or ``"device": "cuda"``),
coverage is checked against the UNION of what ``build_registry`` returns
across ``DEVICES`` (plus ``describe_tool``, which is device-agnostic), and
each case is dispatched through a server built for ITS OWN declared device
(see ``_server_for_device``) rather than one server shared by every case
regardless of what it needs.

Before running any case, this script discovers every tool the real server
would list on either device and fails loudly if ``CASES`` does not cover
all of them. That is what makes adding a new engine WITHOUT adding a live
case visible here rather than silent: ``tests/test_live_proof_script.py``
runs the equivalent check on the host, without Docker, so the gap shows up
in CI too.

Each case names a tool, the device it needs, the arguments to call it with,
and the keys expected in the parsed JSON payload on success. A case whose
``expect_keys`` is ``["error"]`` is an EXPECTED-FAILURE case: the driver
asserts ``isError`` is True and that the payload carries an ``error``
message, rather than asserting success. No case currently uses this — every
registered tool now has a proven success path — but the mechanism stays,
since a genuinely unproven success path (as ``run_ipsae``'s was, until the
staging fix below) is a real state a future engine can land in, and it must
be reported honestly rather than silently omitted or faked.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from mcp import types
from mcp.server import Server

from protein_design_mcp.app import ServerApp, build_registry
from protein_design_mcp.meta_tools import DESCRIBE_TOOL_MANIFEST

# Every device server.py's own DEVICE resolution can select (env var
# override, or torch.cuda.is_available()) -- see server.py. Coverage and
# dispatch are checked against exactly these two, regardless of whatever
# device this script's own process happens to be running under, so a case
# is always checked/dispatched against the registry it actually declared.
DEVICES = ("cpu", "cuda")

CASES: list[dict] = [
    {
        "tool": "run_prodigy",
        "device": "cpu",
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
        "device": "cpu",
        "arguments": {
            "input_pdb": "tests/fixtures/test_pdbs/mini_protein_complete.pdb",
            "max_iterations": 50,
        },
        "expect_keys": ["energy_change_kj_mol", "outputs"],
    },
    {
        "tool": "run_mpnn",
        "device": "cpu",
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
        "device": "cpu",
        "arguments": {
            "pae_json": "tests/fixtures/pae/example_pae.json",
            "structure": "tests/fixtures/test_pdbs/two_chain_complex.pdb",
        },
        "expect_keys": ["ipsae", "chain_pair"],
    },
    {
        "tool": "describe_tool",
        "device": "cpu",
        "arguments": {"category": "scoring"},
        "expect_keys": ["tools"],
    },
    {
        # BoltzGen's `filtering` step runs no model -- pure CPU dataframe
        # ranking over a real (tiny, single-design) analysis directory this
        # tool's own wave produced with a live GPU run (design_to_target_iptm
        # 0.60088, design_ptm 0.9398 -- see tests/fixtures/boltzgen/ and
        # wave-C-report.md). Included even on a CPU-only image, unlike the
        # other GPU-required BoltzGen tools, because requires.gpu is false.
        "tool": "run_boltzgen_filter",
        "device": "cpu",
        "arguments": {
            "design_spec": "tests/fixtures/boltzgen/design_spec.yaml",
            "design_dir": "tests/fixtures/boltzgen/analysis_dir",
            "budget": 1,
            "top_budget": 1,
        },
        "expect_keys": ["selected_designs", "num_selected"],
    },
    {
        # BoltzGen's `design` step -- all-atom diffusion, GPU-required.
        # num_designs=1 keeps this quick; verified live on GPU 7 (2 designs,
        # ~2 minutes total including one-time model load -- see
        # wave-C-report.md).
        "tool": "run_boltzgen_design",
        "device": "cuda",
        "arguments": {
            "design_spec": "tests/fixtures/boltzgen/design_spec.yaml",
            "num_designs": 1,
        },
        "expect_keys": ["designs", "num_designs"],
    },
    {
        # BoltzGen's own inverse-folding head (--only_inverse_fold),
        # standalone -- redesigns chain A of a real backbone this same wave
        # generated live with run_boltzgen_design, verified live on GPU 7
        # (9.1s for 2 sequences on a 17-residue chain -- see
        # wave-C-report.md). Chain B (the target, not marked `design:`) is
        # expected back unchanged.
        "tool": "run_boltzgen_inverse_fold",
        "device": "cuda",
        "arguments": {
            "design_spec": "tests/fixtures/boltzgen/redesign_spec.yaml",
            "inverse_fold_num_sequences": 1,
        },
        "expect_keys": ["designs", "num_designs"],
    },
]


_SERVERS: dict[str, Server] = {}


def _server_for_device(device: str) -> Server:
    """Build (once, then cache) a real ``mcp.server.Server`` wired to a
    ``ServerApp`` whose registry was built for ``device``.

    Mirrors server.py's own wiring exactly (``Server`` + ``ServerApp`` +
    ``build_registry``) -- see server.py's module-level ``server``/``_app``
    and its ``list_tools``/``call_tool`` handlers -- but keyed PER DEVICE
    instead of the single ``DEVICE`` the running process resolves from its
    environment at import time. That is what makes a case declaring
    ``device="cuda"`` actually get dispatched through a registry that has
    the GPU-only tools, and a case declaring ``device="cpu"`` through one
    that doesn't, regardless of how this script itself was launched.
    """
    if device not in _SERVERS:
        srv = Server("protein-design-mcp")
        app = ServerApp(build_registry(device=device))

        @srv.list_tools()
        async def list_tools() -> list[types.Tool]:
            return await app.list_tools()

        @srv.call_tool(validate_input=False)
        async def call_tool_handler(name: str, arguments: dict[str, Any]):
            return await app.call_tool(name, arguments)

        _SERVERS[device] = srv
    return _SERVERS[device]


async def call_tool(name: str, arguments: dict, device: str) -> types.CallToolResult:
    """Invoke ``name`` the same way a real MCP client's request would,
    against the server built for ``device`` (see ``_server_for_device``).

    Fetches the ``types.CallToolRequest`` handler ``@srv.call_tool(...)``
    registered in ``request_handlers`` and calls it with a real
    ``CallToolRequest``, rather than calling ``ServerApp.call_tool`` (or
    ``app.call_tool``) directly.
    """
    server = _server_for_device(device)
    handler = server.request_handlers[types.CallToolRequest]
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


def _registered_tool_names_by_device() -> dict[str, set[str]]:
    """Tool name -> every device (of DEVICES) that registers it.

    ``describe_tool`` is a meta-tool with no manifest -- ``ServerApp.
    list_tools`` adds it unconditionally, regardless of what device its
    registry was built for (see app.py) -- so it maps to every entry of
    DEVICES here too.
    """
    registered: dict[str, set[str]] = {}
    for device in DEVICES:
        for tool in build_registry(device=device).tools():
            registered.setdefault(tool.name, set()).add(device)
    registered.setdefault(DESCRIBE_TOOL_MANIFEST.name, set()).update(DEVICES)
    return registered


def _check_coverage() -> None:
    """Fail loudly if CASES does not exactly match the UNION of the cpu and
    cuda live server surfaces.

    Mirrors tests/test_live_proof_script.py's host-side check. Checking the
    union (rather than the surface for one hardcoded device) is what lets a
    ``requires.gpu: true`` tool ever be added to CASES without permanently
    breaking this assertion, and what stops one from being silently left
    off it: a manifest excluded from the CPU registry but present on CUDA
    (or vice versa) can never go untested, and neither can a case left over
    for a tool that no longer exists on either device.
    """
    registered = _registered_tool_names_by_device()
    covered = {case["tool"] for case in CASES}

    missing = registered.keys() - covered
    if missing:
        detail = ", ".join(
            f"{name} (device={sorted(registered[name])})" for name in sorted(missing)
        )
        raise SystemExit(f"FATAL: tool(s) with no live-proof case: {detail}")

    extra = covered - registered.keys()
    if extra:
        raise SystemExit(
            f"FATAL: CASES references unregistered tool(s): {sorted(extra)}"
        )

    # A case can name a tool that IS covered overall but declare the WRONG
    # device for it (e.g. device="cpu" for a requires.gpu: true tool): that
    # would dispatch it against a registry that excludes it for an
    # unrelated (device) reason, and the resulting failure would look like
    # a broken engine rather than a mislabelled case.
    for case in CASES:
        tool, device = case["tool"], case["device"]
        if device not in registered[tool]:
            raise SystemExit(
                f"FATAL: {tool!r} case declares device={device!r}, but "
                f"{tool} is only registered on {sorted(registered[tool])}"
            )


async def _run_case(case: dict) -> None:
    tool = case["tool"]
    device = case["device"]
    expect_keys = case["expect_keys"]
    expect_error = expect_keys == ["error"]

    result = await call_tool(tool, case["arguments"], device)
    _print_result(f"{tool} (device={device})", result)

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
    _check_coverage()

    for case in CASES:
        await _run_case(case)

    print("\n=== ALL CHECKS PASSED ===")


if __name__ == "__main__":
    asyncio.run(main())
