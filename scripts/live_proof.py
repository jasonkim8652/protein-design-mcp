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
        # Real biological interface, not synthetic: 1BRS chain A (barnase)
        # / chain D (barstar), the same pair run_prodigy's own case above
        # uses. Confirmed live: hotspot_tags includes A59, A83, A27, A102 --
        # residues well established in the literature as barnase-barstar
        # interface hotspots -- so this is not just a shape check.
        "tool": "run_interface_residues",
        "device": "cpu",
        "arguments": {
            "complex_pdb": "tests/fixtures/test_pdbs/1BRS.pdb",
            "target_chain": "A",
            "binder_chains": ["D"],
        },
        "expect_keys": ["hotspot_tags", "residues", "n_interface_residues"],
    },
    {
        # mini_protein.pdb is a genuinely standalone single-chain file (no
        # other chains to accidentally treat as "bound"), matching this
        # tool's own "unbound target" input contract. msa: null is the
        # documented, valid, no-conservation path.
        "tool": "run_epitope_scan",
        "device": "cpu",
        "arguments": {
            "target_pdb": "tests/fixtures/test_pdbs/mini_protein.pdb",
            "chain": "A",
            "msa": None,
        },
        "expect_keys": ["hotspot_tags", "residues", "n_exposed_residues"],
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
    {
        # Task 7: Boltz-2, MSA-free, one short chain (the 20-residue PHF6
        # peptide used as this manifest's own schema example). Sampling
        # knobs pushed to the fast end of their allowed range
        # (recycling_steps=1, sampling_steps=10, diffusion_samples=1) --
        # this is a pipeline proof, not a benchmark. Runs the user's own
        # editable fork at ~/projects/lightning-boltz-dev, not upstream
        # Boltz (see the manifest's fork_notice).
        "tool": "run_boltz",
        "device": "cuda",
        "arguments": {
            "chains": [{"sequence": "NLYIQWLKDGGPSSGRPPPS", "msa": None, "copies": 1}],
            "recycling_steps": 1,
            "sampling_steps": 10,
            "diffusion_samples": 1,
        },
        "expect_keys": ["confidence_score", "ptm", "num_structures"],
    },
    {
        # Task 7: Chai-1, MSA-free, same short peptide as run_boltz above so
        # the two are directly comparable. Minimal trunk/diffusion settings
        # for speed; use_esm_embeddings/low_memory left at their (true)
        # defaults since neither costs extra wall time here.
        "tool": "run_chai1",
        "device": "cuda",
        "arguments": {
            "chains": [{"sequence": "NLYIQWLKDGGPSSGRPPPS", "msa": None, "copies": 1}],
            "num_trunk_recycles": 1,
            "num_diffn_timesteps": 10,
            "num_diffn_samples": 1,
            "num_trunk_samples": 1,
        },
        "expect_keys": ["aggregate_score", "ptm", "num_structures"],
    },
    {
        # Task 7: Protenix v1, MSA-free, same short peptide. cycle/step/
        # sample pushed to the fast end (still >=1, the schema's own
        # floor). model_name is always pinned to
        # protenix_base_default_v1.0.0 internally -- not a schema
        # parameter, so there is no selector here that could trigger a
        # different (undownloaded) checkpoint.
        "tool": "run_protenix",
        "device": "cuda",
        "arguments": {
            "chains": [{"sequence": "NLYIQWLKDGGPSSGRPPPS", "msa": None, "copies": 1}],
            "cycle": 1,
            "step": 10,
            "sample": 1,
            "seeds": [101],
        },
        "expect_keys": ["ptm", "num_structures", "model_pinned"],
    },
    {
        # Task 7: OpenFold3 (default "OpenBind" checkpoint, already
        # cached), MSA-free, same short peptide. num_diffusion_samples=1 is
        # the schema's own floor (also its built-in model-config default in
        # spirit -- see the manifest's diffusion-sample-cap doc).
        "tool": "run_openfold3",
        "device": "cuda",
        "arguments": {
            "chains": [{"sequence": "NLYIQWLKDGGPSSGRPPPS", "msa": None, "copies": 1}],
            "num_diffusion_samples": 1,
            "seeds": [42],
        },
        "expect_keys": ["avg_plddt", "ptm", "num_structures"],
    },
    {
        # Task 7: ESMFold2 -- no msa parameter at all, by design (see the
        # manifest doc: this SDK has no code path for a real multi-row
        # alignment). Reuses the 76-residue ubiquitin sequence already used
        # elsewhere in this file's fixtures/examples. num_diffusion_samples
        # lowered to 1 since only the first sample is ever written to
        # this tool's output PDB regardless of the setting (confirmed from
        # source, see the manifest doc) -- raising it would add compute
        # without changing what comes back through this tool.
        "tool": "run_esmfold2",
        "device": "cuda",
        "arguments": {
            "sequence": "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
            "num_recycles": 4,
            "num_diffusion_samples": 1,
            "num_sampling_steps": 10,
        },
        "expect_keys": ["mean_plddt", "num_residues", "sequence_length"],
    },
    {
        # Task 7: Promera, single-chain (monomer) target -- a one-entry
        # `chains` map predicts and scores a monomer, per the manifest's own
        # doc. `msa: null` (top-level, not per-chain) runs every chain
        # MSA-free deliberately -- confirmed live by Wave B on a 2-chain
        # complex; this case narrows to one chain for speed.
        # diffusion_steps=5 matches the install smoke test's own
        # fast-sanity-check value (the manifest doc explicitly flags 5 as
        # "not a value to use for a real prediction", which is exactly what
        # a live-proof case is).
        "tool": "run_promera",
        "device": "cuda",
        "arguments": {
            "chains": {
                "A1": {
                    "type": "protein",
                    "sequence": "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
                    "entity_id": 1,
                }
            },
            "msa": None,
            "recycling_steps": 1,
            "diffusion_samples": 1,
            "diffusion_steps": 5,
            "num_seeds": 1,
        },
        "expect_keys": ["complex_plddt", "ptm"],
    },
    {
        # Task 7: RoseTTAFold3, single chain, MSA-free (msa keyed per
        # chain_id, per the manifest's own policy -- every chain's choice
        # must be stated, null included). n_recycles=2 is the documented
        # HARD FLOOR (n_recycles=1 was confirmed live, per the manifest, to
        # crash with "Recycling generator produced no outputs") --
        # deliberately not going below it. num_steps=5 matches the install
        # smoke test's own fast-sanity-check value (manifest: "not a value
        # to use for a real prediction").
        "tool": "run_rf3",
        "device": "cuda",
        "arguments": {
            "chains": [
                {
                    "chain_id": "A",
                    "sequence": "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
                }
            ],
            "msa": {"A": None},
            "n_recycles": 2,
            "diffusion_batch_size": 1,
            "num_steps": 5,
        },
        "expect_keys": ["ptm", "iptm"],
    },
    {
        # Task 7: AlphaFold2-Multimer via ColabFold. SETTLED LIVE (Task 7,
        # round 1): a single-chain (monomer) query was tried first, and
        # succeeded, but ColabFold's own scores JSON for a monomer carries
        # no "iptm" key at all (only plddt/ptm/max_pae/pae) -- iptm is a
        # cross-chain metric ColabFold only computes for an actual
        # multimer, confirmed live by the real returned payload. Switched
        # to the manifest's own two-chain schema example (also the exact
        # shape Wave B verified live: "2-chain complex ... succeeded
        # (ptm: 0.26, iptm: 0.04)") so iptm is genuinely present.
        # MSA-free (msa: null forces --msa-mode single_sequence -- the only
        # mode this tool can ever reach; its remote mmseqs2_* modes are
        # structurally unreachable, see the adapter's own docstring).
        # num_models=1 runs only the fastest of the 5 already-cached
        # multimer_v3 parameter sets rather than all 5 -- no new download
        # is triggered, model_type itself is not a parameter this tool
        # exposes.
        "tool": "run_alphafold2_multimer",
        "device": "cuda",
        "arguments": {
            "sequences": [
                "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDA",
                "MASSQTNSAGGGKKD",
            ],
            "msa": None,
            "num_recycle": 1,
            "num_models": 1,
        },
        "expect_keys": ["ptm", "iptm"],
    },
    {
        # Task 7: MMseqs2 search, the only alignment-producing tool here.
        # Restricted to the smallest local database (small_bfd, 16.8GB) and
        # pair=false/search_templates=false to skip the ~78GB UniProt and
        # pdb_seqres searches entirely -- task-3-report.md confirmed a
        # search against small_bfd alone finishes in well under 20s on GPU
        # 7, against the ~87s a full 3-database call takes. Reuses the same
        # 76-residue ubiquitin sequence as run_esmfold2 above; unlike that
        # report's own live run (which found this sequence's hits all
        # dedupe to zero against itself -- ubiquitin is extremely
        # conserved), a zero (or non-zero) unpaired_hit_count is still a
        # genuine, valid result this case does not depend on either way.
        "tool": "run_mmseqs_search",
        "device": "cuda",
        "arguments": {
            "sequence": "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
            "unpaired_databases": ["small_bfd"],
            "pair": False,
            "search_templates": False,
        },
        "expect_keys": ["query_length", "unpaired_hit_count", "used_gpu"],
    },
    {
        # Wave D: Proteina-Complexa binder generation. Cheapest possible
        # smoke shape -- single-pass (no search), one length, 20 denoising
        # steps rather than this engine's own 400-step default.
        # EXPECTED-FAILURE for now: the `proteina` conda env this tool
        # dispatches into is missing several of Proteina-Complexa's own
        # declared dependencies (lightning, pydantic, typing_extensions,
        # fsspec among others -- `pip check` run live, 2026-09-22, lists
        # the full set), which blocks `complexa generate` regardless of
        # this case's own arguments -- see wave-D-report.md. Once that
        # environment is fixed, this case should be updated to assert a
        # real success payload (expect_keys=["rewards", "num_samples"])
        # instead.
        "tool": "run_proteina_complexa_generate",
        "device": "cuda",
        "arguments": {
            "task_name": "02_PDL1",
            "search_algorithm": "single-pass",
            "num_lengths": 1,
            "nsteps": 20,
        },
        "expect_keys": ["error"],
    },
    {
        # Wave D: Proteina-Complexa re-ranking of a finished generate run's
        # rewards CSV -- runs no model, requires.gpu is false. Fixture is a
        # 2-row CSV in generate.py's own rewards_{config}_{job}.csv shape.
        # EXPECTED-FAILURE for now, same broken `proteina` conda env as
        # run_proteina_complexa_generate above (blocks every `complexa`
        # subcommand, not something specific to this tool) -- see
        # wave-D-report.md. Once fixed, update to
        # expect_keys=["selected_designs", "num_selected"].
        "tool": "run_proteina_complexa_filter",
        "device": "cpu",
        "arguments": {
            "rewards_csv": "tests/fixtures/proteina_complexa/rewards_sample.csv",
        },
        "expect_keys": ["error"],
    },
    {
        # Wave D: Proteina-Complexa diversity analysis over a design set --
        # runs no neural model, requires.gpu is false. Reuses two existing
        # structure fixtures as a 2-design "set". EXPECTED-FAILURE for now,
        # same broken `proteina` conda env (see run_proteina_complexa_generate
        # above) -- see wave-D-report.md. Once fixed, update to
        # expect_keys=["foldseek_diversity", "mmseqs_diversity", "num_designs"].
        "tool": "run_proteina_complexa_analyze",
        "device": "cpu",
        "arguments": {
            "structure_paths": [
                "tests/fixtures/test_pdbs/mini_protein.pdb",
                "tests/fixtures/test_pdbs/two_chain_complex.pdb",
            ],
            "sequences": ["MKTAYIAKQRQISFVK", "MKTAYIAKQRQISFVL"],
        },
        "expect_keys": ["error"],
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
