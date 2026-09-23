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
message, rather than asserting success. Exactly one case uses this today:
``get_job_status`` queried with a deliberately nonexistent job id, which
MUST fail (task-14-report.md) — every other registered tool now has a
proven success path. The mechanism stays for that reason and because a
genuinely unproven success path (as ``run_ipsae``'s was, until the staging
fix below) is a real state a future engine can land in, and it must be
reported honestly rather than silently omitted or faked.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from mcp import types
from mcp.server import Server

from protein_design_mcp.app import ServerApp, build_registry
from protein_design_mcp.meta_tools import DESCRIBE_TOOL_MANIFEST, GET_JOB_STATUS_MANIFEST
from protein_design_mcp.utils.job_queue import get_job_queue

# Every device server.py's own DEVICE resolution can select (env var
# override, or torch.cuda.is_available()) -- see server.py. Coverage and
# dispatch are checked against exactly these two, regardless of whatever
# device this script's own process happens to be running under, so a case
# is always checked/dispatched against the registry it actually declared.
DEVICES = ("cpu", "cuda")

def _seed_real_completed_job() -> str:
    """Create ONE real, completed job through the exact same ``JobQueue``
    singleton ``get_job_status`` itself reads from
    (``protein_design_mcp.utils.job_queue.get_job_queue()`` -- the module
    ``job_status.get_design_status`` imports it from), so the
    ``get_job_status`` success case below queries a job that genuinely
    exists rather than a fabricated payload.

    ``get_job_status`` has no engine to dispatch through at all (see
    ``meta_tools.py`` / ``job_status.py``) and, confirmed by reading the
    rest of this server (task-14-report.md), nothing else in it EVER calls
    ``JobQueue.create_job()``/``complete_job()`` -- the composite pipeline
    that once produced jobs this way is gone, and no current tool writes to
    the job queue. So this harness has to seed the one real job its success
    case can query, the same way
    ``tests/test_job_queue.py::TestGetDesignStatus::test_get_status_completed_with_result``
    does. A COMPLETED job (rather than a running one with progress) is
    chosen deliberately: it exercises ``status``/``job_id``/``created_at``/
    ``result``/``completed_at`` -- every key a real terminal job carries --
    without ever touching ``job_status._estimate_time_remaining`` (only
    reached for a job with ``progress`` set), which is a live, per-job
    elapsed-time-rate estimate rather than a fixed value regardless.
    """
    queue = get_job_queue()
    job_id = queue.create_job()
    queue.complete_job(
        job_id,
        {"designs_completed": 1, "note": "live-proof: real seeded job, not a fabricated result"},
    )
    return job_id


_REAL_JOB_ID = _seed_real_completed_job()


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
        # ranking. Included even on a CPU-only image, unlike the other
        # GPU-required BoltzGen tools, because requires.gpu is false.
        #
        # FIXED (task 12, in-container proof): this case predated commit
        # f35079d (Wave E), which redesigned run_boltzgen_filter's inputs
        # from a single `design_dir` to three explicit staged file lists
        # (`generated_files`, `metrics_files`, `refold_structures`) --
        # nothing updated this case, so it dispatched with a `design_dir`
        # argument the manifest no longer accepts ("received unexpected
        # parameter(s): design_dir"), caught only by the in-container run,
        # near the very end of a long piece of work -- see
        # tests/test_live_proof_cases_validate.py, added the same day so
        # this class of drift is caught statically instead. generated_files
        # and refold_structures reuse the exact design_spec.cif/npz fixtures
        # run_boltzgen_design/run_boltzgen_fold's own live GPU runs produced
        # (tests/fixtures/boltzgen/generated_designs/,
        # tests/fixtures/boltzgen/refold/ -- see wave-E-report.md's own
        # design->fold->analyze->filter transcript). metrics_files is a NEW
        # fixture (tests/fixtures/boltzgen/analyzed/) produced by running
        # run_boltzgen_analyze live on the host (CPU, requires.gpu: false)
        # over those exact same generated_files/refold_structures --
        # confirmed its `id` column is "design_spec", matching the other
        # two file sets, so Filter's row lookup actually finds it.
        "tool": "run_boltzgen_filter",
        "device": "cpu",
        "arguments": {
            "design_spec": "tests/fixtures/boltzgen/design_spec.yaml",
            "generated_files": [
                "tests/fixtures/boltzgen/generated_designs/design_spec.cif",
                "tests/fixtures/boltzgen/generated_designs/design_spec.npz",
            ],
            "metrics_files": [
                "tests/fixtures/boltzgen/analyzed/aggregate_metrics_analyze.csv",
                "tests/fixtures/boltzgen/analyzed/ca_coords_sequences.pkl.gz",
            ],
            "refold_structures": ["tests/fixtures/boltzgen/refold/design_spec.cif"],
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
        # Task 14: Proteina-Complexa binder generation. Cheapest possible
        # smoke shape -- single-pass (no search), one length, 20 denoising
        # steps rather than this engine's own 400-step default. Task 8
        # already fixed the `proteina_complexa` env this dispatches into
        # (this manifest's own `prefix:`) and verified `complexa generate`
        # completes end to end with exactly this shape of arguments -- the
        # only reason this case still failed afterward (task-13) was a
        # SEPARATE, previously-undiscovered bug this task found: the
        # `generated_structures` output pattern
        # (run_proteina_complexa_generate.yaml) was `inference/*/*.pdb`, two
        # path segments, but generate.py's own save_predictions actually
        # writes THREE segments deep
        # (inference/<run>/<sample_dir>/<sample_dir>.pdb) -- confirmed live,
        # 2026-09-22, via a real call through the actual ServerApp.call_tool
        # path (host, GPU 7, CUDA_VISIBLE_DEVICES=7, EnvDispatcher(runner=
        # None) reaching proteina_complexa's own bin/ directly, same
        # technique task-8/task-10 used): the two-segment pattern matched
        # zero files every time, which made every real call fail with
        # "declared output 'generated_structures' matched no file" before
        # this adapter's own parse_output ever ran -- this tool's dispatch
        # path had never actually been exercised end to end before (task-8's
        # own live check ran the bare `complexa generate` CLI by hand, not
        # through this glob). Fixed to `inference/*/*/*.pdb` (see that
        # manifest). Confirmed live with the fix: isError=False, one 2-chain
        # PDB (chain A=115-residue PD-L1 target, chain B=147-residue
        # generated binder) and a real rewards_search_binder_local_pipeline_0.csv
        # (total_reward=0.0 for this single-pass, unweighted-by-default
        # sample -- a real, if uninteresting, value, not a placeholder).
        "tool": "run_proteina_complexa_generate",
        "device": "cuda",
        "arguments": {
            "task_name": "02_PDL1",
            "search_algorithm": "single-pass",
            "num_lengths": 1,
            "nsteps": 20,
        },
        "expect_keys": ["rewards", "num_samples"],
    },
    {
        # Task 14: Proteina-Complexa re-ranking of a finished generate run's
        # rewards CSV -- runs no model, requires.gpu is false.
        # task-13-report.md root-caused this case's old EXPECTED-FAILURE
        # status: filter.py globs its root_path for files matching
        # `rewards_{config_name}_*.csv` (config_name is always
        # "search_binder_local_pipeline" -- this adapter's own
        # `++base_config_name=search_binder_local_pipeline` override, never
        # caller-controlled), but the old fixture was named
        # `rewards_sample.csv`, which can never match that glob regardless
        # of environment -- a stale fixture, not an engine defect. Fixed by
        # using a REAL rewards CSV this task generated live (see
        # run_proteina_complexa_generate's case above) and committed under
        # its own honest, engine-produced name
        # (tests/fixtures/proteina_complexa/generated/
        # rewards_search_binder_local_pipeline_0.csv) -- the glob matches
        # because this file genuinely came out of a real `complexa generate`
        # run with that config, not because it was renamed to fit. This
        # fixture is read-only input here -- it does not depend on the
        # generate case above having run first in the same script execution.
        # Confirmed live: isError=False, num_selected=1, num_total_designs=1.
        "tool": "run_proteina_complexa_filter",
        "device": "cpu",
        "arguments": {
            "rewards_csv": "tests/fixtures/proteina_complexa/generated/rewards_search_binder_local_pipeline_0.csv",
        },
        "expect_keys": ["selected_designs", "num_selected"],
    },
    {
        # Task 14: Proteina-Complexa diversity analysis over a design set --
        # runs no neural model, requires.gpu is false. Uses the SAME real
        # generated structure as run_proteina_complexa_filter's case above
        # (tests/fixtures/proteina_complexa/generated/
        # job_0_n_262_id_0_single_orig0.pdb -- the actual PDB a real
        # run_proteina_complexa_generate call produced, chain A=target,
        # chain B=binder), so this is genuinely a step in the real
        # generate->filter->analyze chain rather than an unrelated synthetic
        # pair -- without making this case depend on the generate case
        # having run first (the fixture is a committed file, read
        # independently). `sequences` is the real A+B concatenated sequence
        # extracted from that same PDB (262 residues, matching
        # generate's own rewards CSV `aatype` column length) -- per the
        # manifest doc, the diversity computation itself derives sequence
        # straight from the structure regardless, so this is not
        # load-bearing for the numbers, only for the record. Only
        # structure_paths/sequences are required (refolding metrics are
        # optional and omitted here, per the manifest doc). A single design
        # degrades gracefully to the trivial (1.0, 1, 1) diversity result
        # rather than failing -- confirmed live by task-8's original chain
        # and reconfirmed here. Confirmed live: isError=False,
        # foldseek_diversity=mmseqs_diversity={"score": 1.0,
        # "num_clusters": 1, "num_samples": 1}, num_designs=1.
        "tool": "run_proteina_complexa_analyze",
        "device": "cpu",
        "arguments": {
            "structure_paths": [
                "tests/fixtures/proteina_complexa/generated/job_0_n_262_id_0_single_orig0.pdb",
            ],
            "sequences": [
                "AFTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNANGSPPPSPPRRDSHLNDSVLPVNGGDAPNPFIKSLNTTNTDDLLTNKDALTDSSDDPDLPSNGNSGTDNADLLPNNIALAPSNALPDNDGKPGKIKSGNPSFDPNTDNLNKTPTKPLNDINLQRPDDLIGDVLLLPNLVELDLTLDT",
            ],
        },
        "expect_keys": ["foldseek_diversity", "mmseqs_diversity", "num_designs"],
    },
    {
        # Task 11: get_job_status has no engine at all (wired like
        # describe_tool -- see meta_tools.py/wave-H-report.md), so its live
        # proof is a registry/dispatch check, not an engine run: a
        # deliberately unknown job_id, driven through the real
        # ServerApp.call_tool path, exercises the exact same lookup a real
        # job_id would hit (get_job_queue().get_job(...) -> None -> the
        # ValueError this tool's own wrapper translates into {"error": ...}),
        # confirmed live to return isError=True with a real "Job not found"
        # message. KEPT as an expected-failure case (task-14): querying a
        # job id that does not exist SHOULD fail, and this is exactly the
        # behavior worth pinning -- this is not the same class of weak case
        # as the Proteina-Complexa trio's old expected-failure cases, which
        # were tools that should have SUCCEEDED. See the next case for the
        # missing success path.
        "tool": "get_job_status",
        "device": "cpu",
        "arguments": {"job_id": "nonexistent-job-id-live-proof-check"},
        "expect_keys": ["error"],
    },
    {
        # Task 14: get_job_status's missing success path. Queries
        # _REAL_JOB_ID -- a job this script itself seeds (see
        # _seed_real_completed_job() above) through the exact same
        # JobQueue singleton get_job_status reads from, since nothing else
        # in this server ever creates one (the composite pipeline that used
        # to is gone; see that function's own docstring). A COMPLETED job
        # (not a running one with progress) is used deliberately: it proves
        # status/job_id/created_at/result/completed_at -- every key a real
        # terminal job carries -- without going anywhere near
        # job_status._estimate_time_remaining (only reached when a job has
        # ``progress`` set). That function's own docstring documents it
        # already derives its estimate from THIS job's own observed
        # progress rate rather than a fixed per-step table (the stale
        # rfdiffusion/proteinmpnn/esmfold table task-14's own brief warned
        # about was for an EARLIER version of this module and is not what
        # ships today -- confirmed by reading job_status.py directly); a
        # progress-based case was avoided anyway since its
        # estimated_time_remaining value is real but non-reproducible
        # (depends on wall-clock timing between two calls), which this
        # harness only ever checks for KEY presence, not value equality, so
        # it would not actually have been "pinning" anything even if used.
        # Confirmed live: isError=False, status="completed", a real
        # result dict, and a real completed_at timestamp.
        "tool": "get_job_status",
        "device": "cpu",
        "arguments": {"job_id": _REAL_JOB_ID},
        "expect_keys": ["status", "job_id", "created_at", "result", "completed_at"],
    },
    {
        # Task 16: AlphaFold 3 dispatches through engine.prefix
        # (/alphafold3_venv, extracted from romerolabduke/alphafast:latest
        # with docker create + docker cp) exactly like every other GPU
        # engine now -- no sibling docker run, no docker.sock. Runs the
        # image's own baked-in entrypoint, not the host repo's newer,
        # incompatible copy; outputs land at out/*, not out/job/*. MSA-free,
        # 20-residue Trp-cage, seeds=[1], num_recycles/num_diffusion_samples
        # dropped to 1 for speed. CONFIRMED LIVE in-container via
        # ServerApp.call_tool, 2026-09-23: isError=False,
        # ranking_score=ptm=0.11 (a real, if low-confidence, monomer
        # prediction -- iptm is null, correctly, since a single chain has no
        # cross-chain interface to report).
        "tool": "run_alphafold3",
        "device": "cuda",
        "arguments": {
            "chains": [
                {
                    "sequence": "NLYIQWLKDGGPSSGRPPPS",
                    "unpaired_msa": None,
                    "paired_msa": None,
                    "copies": 1,
                }
            ],
            "seeds": [1],
            "num_recycles": 1,
            "num_diffusion_samples": 1,
        },
        "expect_keys": ["ranking_score", "ptm", "iptm"],
    },
    {
        # Task 11: BoltzGen's `analysis` step (CPU dataframe/geometry work,
        # requires.gpu is false) over a REAL design -> fold chain this task
        # generated live (num_designs=1, sampling_steps=20 for fold) and
        # committed as fixtures -- tests/fixtures/boltzgen/generated_designs/
        # (run_boltzgen_design's own outputs.generated_designs) and
        # tests/fixtures/boltzgen/refold/ (run_boltzgen_fold's own
        # outputs.refolded_structures/refold_metrics on that same design).
        # designfolding_metrics left at its false default (no
        # design_refold_structures/metrics supplied) -- matching wave E's
        # own live-verified chain; that optional path is unit-tested but not
        # live-verified, per that wave's report. Confirmed live:
        # isError=False, num_designs_analyzed=1.
        "tool": "run_boltzgen_analyze",
        "device": "cpu",
        "arguments": {
            "design_spec": "tests/fixtures/boltzgen/design_spec.yaml",
            "generated_files": [
                "tests/fixtures/boltzgen/generated_designs/design_spec.cif",
                "tests/fixtures/boltzgen/generated_designs/design_spec.npz",
            ],
            "refold_structures": ["tests/fixtures/boltzgen/refold/design_spec.cif"],
            "refold_metrics": ["tests/fixtures/boltzgen/refold/design_spec.npz"],
        },
        "expect_keys": ["num_designs_analyzed"],
    },
    {
        # Task 11: BoltzGen's `design_folding` step (design refolded ALONE,
        # target stripped out) -- same real generated_files fixture as
        # run_boltzgen_fold below (see that case's comment for provenance).
        # sampling_steps dropped from the 200 default to 20 for speed.
        # Confirmed live on GPU 7: isError=False, one refolds entry with a
        # real design_ptm/design_iptm (no target-relative fields, correctly
        # -- the target is absent from this step's input entirely).
        "tool": "run_boltzgen_design_fold",
        "device": "cuda",
        "arguments": {
            "design_spec": "tests/fixtures/boltzgen/design_spec.yaml",
            "generated_files": [
                "tests/fixtures/boltzgen/generated_designs/design_spec.cif",
                "tests/fixtures/boltzgen/generated_designs/design_spec.npz",
            ],
            "sampling_steps": 20,
            "diffusion_samples": 1,
        },
        "expect_keys": ["refolds", "num_refolds"],
    },
    {
        # Task 11: BoltzGen's `folding` step (design refolded WITH its
        # target present -- the interface-confidence step everything
        # downstream ranks on). generated_files is a REAL design this task
        # produced live via run_boltzgen_design (design_spec.yaml,
        # num_designs=1) and committed as a fixture (both the .cif and its
        # .npz -- dropping either makes this tool unable to read the design
        # back, see the manifest doc) -- not synthetic, not reused from
        # another wave's fixture. sampling_steps dropped from the 200
        # default to 20 for speed. Confirmed live on GPU 7: isError=False,
        # one refolds entry with a real design_to_target_iptm=0.53.
        "tool": "run_boltzgen_fold",
        "device": "cuda",
        "arguments": {
            "design_spec": "tests/fixtures/boltzgen/design_spec.yaml",
            "generated_files": [
                "tests/fixtures/boltzgen/generated_designs/design_spec.cif",
                "tests/fixtures/boltzgen/generated_designs/design_spec.npz",
            ],
            "sampling_steps": 20,
            "diffusion_samples": 1,
        },
        "expect_keys": ["refolds", "num_refolds"],
    },
    {
        # Task 11: ColabFold's own MMseqs2 search (requires.gpu is false).
        # `backend` has no default -- it is a required, explicit choice
        # (local/remote). "local" cannot work on this host: ColabFold's own
        # UniRef30/envDB databases are not installed anywhere on it (task-10
        # confirmed this by exhaustive search) -- only "remote" is usable
        # here. THIS CALL LEAVES THE MACHINE: "remote" transmits the query
        # sequence to https://api.colabfold.com, the ColabFold project's own
        # public server, not this deployment -- see the manifest doc's
        # "backend" section before reusing this case's shape elsewhere.
        # Reuses the same 76-residue ubiquitin sequence as run_mmseqs_search
        # above. Confirmed live: isError=False, query_length=76, a real
        # ~3MB merged a3m with genuine UniRef100 hits returned in ~2s.
        "tool": "run_colabfold_search",
        "device": "cpu",
        "arguments": {
            "sequence": "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
            "backend": "remote",
        },
        "expect_keys": ["query_length"],
    },
    {
        # Task 11: ESM2-650M masked-marginal pseudo-log-likelihood scoring
        # (task-10's fixed dispatch, engine.prefix esm_env). Reuses the same
        # 76-residue ubiquitin sequence as run_esmfold2/run_mmseqs_search
        # above. Confirmed live on GPU 7: isError=False,
        # pseudo_log_likelihood=-1.04 (a physically sane value, matching
        # wave-H's own live finding).
        "tool": "run_esm_score",
        "device": "cuda",
        "arguments": {
            "sequence": "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
            "batch_size": 32,
        },
        "expect_keys": ["pseudo_log_likelihood", "per_residue_log_likelihood", "sequence_length"],
    },
    {
        # Task 11: FrameFlow unconditional monomer generation. min_length is
        # this schema's own floor (20) to keep the run fast; samples_per_
        # length=1, num_timesteps dropped from the 100 default to 10.
        # Confirmed live on GPU 7: isError=False, one 20-residue backbone
        # (length verified from the file's own CA atoms, not just the
        # filename).
        "tool": "run_frameflow",
        "device": "cuda",
        "arguments": {
            "min_length": 20,
            "max_length": 20,
            "samples_per_length": 1,
            "num_timesteps": 10,
        },
        "expect_keys": ["backbones", "num_backbones"],
    },
    {
        # Task 11: Genie 2 unconditional monomer generation, matching wave
        # F's own live-verified shape (1x50-residue sample). Confirmed live
        # on GPU 7: isError=False, backbones=[{"id": "50_0", "length": 50}].
        "tool": "run_genie2",
        "device": "cuda",
        "arguments": {
            "min_length": 50,
            "max_length": 50,
            "num_samples": 1,
            "batch_size": 1,
        },
        "expect_keys": ["backbones", "num_backbones"],
    },
    {
        # Task 11: Genie 3 target-conditioned binder generation
        # (genie2_fixed env, Bio.PDB available). target_pdb is
        # mini_protein.pdb (a genuine standalone 5-residue single chain,
        # already used by run_epitope_scan/run_openmm_minimize elsewhere in
        # this file) -- hotspots A2/A4 on it, binder fixed to 20 residues,
        # n_sample_step dropped from the 100 default to 10 for speed.
        # expand_interface left at its false default (condition on exactly
        # the given hotspots). Confirmed live on GPU 7: isError=False, one
        # 20-residue binder chain (A) plus the unchanged 5-residue target
        # (B), cond_strategy="hotspot".
        "tool": "run_genie3_binder",
        "device": "cuda",
        "arguments": {
            "target_pdb": "tests/fixtures/test_pdbs/mini_protein.pdb",
            "hotspot_residues": ["A2", "A4"],
            "binder_min_length": 20,
            "binder_max_length": 20,
            "num_samples": 1,
            "n_sample_step": 10,
        },
        "expect_keys": ["binders", "num_binders"],
    },
    {
        # Task 11: Genie 3 unconditional monomer generation (shares the
        # genie2 conda env's interpreter via an explicit PYTHONPATH -- see
        # the manifest's engine comment). n_sample_step dropped from the 100
        # default to 10 for speed; model_variant left at its "v1"/DDIM
        # default. Confirmed live on GPU 7: isError=False, one 60-residue
        # backbone.
        "tool": "run_genie3_scaffold",
        "device": "cuda",
        "arguments": {
            "min_length": 60,
            "max_length": 60,
            "num_samples": 1,
            "batch_size": 1,
            "n_sample_step": 10,
        },
        "expect_keys": ["backbones", "num_backbones"],
    },
    {
        # Task 11: La-Proteina unconditional, all-atom monomer generation
        # (LD1/AE1 checkpoint pair). nsteps=20 matches this manifest's own
        # documented fast-smoke-test value (its doc explicitly notes this is
        # "far below the recommended value ... only appropriate for
        # confirming the path works"). Confirmed live on GPU 7:
        # isError=False, one 50-residue backbone
        # (id="job_0_n_50_id_0", matching wave F's own live finding).
        "tool": "run_la_proteina",
        "device": "cuda",
        "arguments": {
            "lengths": [50],
            "num_samples": 1,
            "max_nsamples_per_batch": 1,
            "nsteps": 20,
        },
        "expect_keys": ["backbones", "num_backbones"],
    },
    {
        # Task 11: MultiFlow unconditional generation WITH ProteinMPNN
        # codesign and its own ESMFold self-consistency refold
        # (multiflow_fixed env, deepspeed installed -- per commit 9af9079,
        # engine caches are now per-engine and persistent, so the ESMFold
        # weight download this refold triggers costs once, not once per
        # call). min_length is this schema's own floor (20); num_timesteps
        # dropped from the 500 default to 10 for speed. Confirmed live on
        # GPU 7: isError=False, one 20-residue sample with a real codesigned
        # sequence AND a real self_consistency score
        # (bb_rmsd=0.61, mean_plddt=82.5 -- the refold genuinely completed,
        # not the defensive {} fallback).
        "tool": "run_multiflow",
        "device": "cuda",
        "arguments": {
            "min_length": 20,
            "max_length": 20,
            "samples_per_length": 1,
            "num_timesteps": 10,
        },
        "expect_keys": ["samples", "num_samples"],
    },
    {
        # Task 11: Protpardelle-1c target-conditioned binder generation
        # (cc83, the BindCraft-benchmark backbone-only multi-chain model --
        # this schema's own default). target_pdb is mini_protein.pdb (the
        # same standalone 5-residue chain run_genie3_binder/
        # run_rfdiffusion_binder/run_rfdiffusion2/run_rfdiffusion3_binder
        # below all use as a target) -- contig "A1-5;/;20-20" keeps the
        # whole 5-residue chain fixed and generates a 20-residue second
        # chain; hotspots left null (a real, documented "no hotspot bias"
        # choice, not an oversight) since this fixture's target has no
        # literature-known interface to point at. Confirmed live on GPU 7:
        # isError=False, one sample with chain A (target, unchanged at 5
        # residues) and chain B (generated, 20 residues).
        "tool": "run_protpardelle",
        "device": "cuda",
        "arguments": {
            "target_pdb": "tests/fixtures/test_pdbs/mini_protein.pdb",
            "contig": "A1-5;/;20-20",
            "total_lengths": [[5, 5], [20, 20]],
            "hotspots": None,
            "num_samples": 1,
            "batch_size": 1,
        },
        "expect_keys": ["samples", "num_samples"],
    },
    {
        # Task 11: RFdiffusion2, conda backend (this schema's own default --
        # task 9 fixed rfd2_fixed so this works with no Docker socket; see
        # that manifest's engine comment). target_pdb is mini_protein.pdb,
        # whose whole 5-residue chain A is exactly what contig
        # "A1-5_10-10" (RFdiffusion2's own underscore-separated grammar)
        # names as the fixed target segment, generating a 10-residue
        # binder. diffusion_steps=15 matches task 9's own confirmed-live
        # value. Confirmed live on GPU 7: isError=False, num_structures=1.
        "tool": "run_rfdiffusion2",
        "device": "cuda",
        "arguments": {
            "target_pdb": "tests/fixtures/test_pdbs/mini_protein.pdb",
            "contig": "A1-5_10-10",
            "backend": "conda",
            "num_designs": 1,
            "diffusion_steps": 15,
            "ckpt_variant": "140",
        },
        "expect_keys": ["num_structures"],
    },
    {
        # Task 11: RFdiffusion3's binder (PPI) path -- RosettaCommons'
        # rc-foundry rfd3, comma-separated contig grammar (diffused segment
        # first, unlike RFdiffusion 1.1.0's target-first convention).
        # target_pdb is mini_protein.pdb; contig "10-10,/0,A1-5" generates a
        # 10-residue binder against the whole 5-residue target chain, with
        # select_hotspots "A2,A4". num_timesteps dropped from the 200
        # default to 10 for speed, matching wave G's own live-verified
        # shape. Confirmed live on GPU 7: isError=False, num_structures=1,
        # a real metrics dict (clash counts, radius of gyration, ...) and
        # diffused_index_map.
        "tool": "run_rfdiffusion3_binder",
        "device": "cuda",
        "arguments": {
            "target_pdb": "tests/fixtures/test_pdbs/mini_protein.pdb",
            "contig": "10-10,/0,A1-5",
            "select_hotspots": "A2,A4",
            "diffusion_batch_size": 1,
            "num_timesteps": 10,
        },
        "expect_keys": ["num_structures", "metrics", "diffused_index_map", "ckpt_path"],
    },
    {
        # Task 11: RFdiffusion3's monomer/scaffold path -- same engine as
        # run_rfdiffusion3_binder above, no target chain (unconditional,
        # length only). length="25" matches wave G's own live-verified
        # value; diffusion_batch_size dropped to 1 (from the 8 default) and
        # num_timesteps to 10 (from 200) for speed. Confirmed live on GPU 7:
        # isError=False, num_structures=1, a real metrics dict.
        "tool": "run_rfdiffusion3_scaffold",
        "device": "cuda",
        "arguments": {
            "length": "25",
            "diffusion_batch_size": 1,
            "num_timesteps": 10,
        },
        "expect_keys": ["num_structures", "metrics", "ckpt_path"],
    },
    {
        # Task 11: RFdiffusion 1.1.0's legacy binder-design path (the
        # /file_server/data/jk661/pioneer/RFdiffusion checkout, PYTHONPATH +
        # nvrtc/JIT-fusion workaround baked into the manifest -- see its
        # engine comment). target_pdb is mini_protein.pdb; contig
        # "A1-5/0 10-10" (RFdiffusion 1.1.0's own space-separated,
        # target-first grammar -- NOT the same grammar as run_rfdiffusion2's
        # or run_rfdiffusion3_binder's contig above, despite superficial
        # similarity) fixes the whole 5-residue target chain and generates a
        # 10-residue binder, hotspot_res A2/A4. diffusion_steps=15 is this
        # schema's own CONFIRMED LIVE hard floor (T<15 raises an
        # AssertionError before any GPU work starts -- see the manifest
        # doc). Confirmed live on GPU 7: isError=False, num_structures=1,
        # checkpoint_used/contig_used echoed back from the engine's own log.
        "tool": "run_rfdiffusion_binder",
        "device": "cuda",
        "arguments": {
            "target_pdb": "tests/fixtures/test_pdbs/mini_protein.pdb",
            "contig": "A1-5/0 10-10",
            "hotspot_res": ["A2", "A4"],
            "num_designs": 1,
            "diffusion_steps": 15,
        },
        "expect_keys": ["num_structures", "checkpoint_used", "contig_used"],
    },
    {
        # Task 11: PyRosetta's InterfaceAnalyzerMover (task-10's fixed
        # dispatch, engine.prefix repointed at the working ~/.conda/envs/
        # BindCraft install -- the dedicated pyrosetta env's own wheel is
        # missing its compiled extension and can never work, see the
        # manifest's engine comment). Same complex/chains as run_prodigy's
        # own case above (1BRS, barnase/barstar, chains A/D). Every
        # repacking/packstat/shape-complementarity knob left at its default.
        # requires.gpu is false -- confirmed live (CPU): isError=False,
        # dG=209.63, dSASA=1573.97,
        # shape_complementarity=0.72, interface_hbonds=13 -- matching
        # task-10's own live finding (dG=210.05) to within run-to-run
        # packing-stochasticity noise.
        "tool": "run_rosetta_interface",
        "device": "cpu",
        "arguments": {
            "complex_pdb": "tests/fixtures/test_pdbs/1BRS.pdb",
            "interface": "A_D",
        },
        "expect_keys": ["dG", "dSASA", "shape_complementarity", "interface_hbonds"],
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

    ``describe_tool`` and ``get_job_status`` are both meta-tools with no
    manifest -- ``ServerApp.list_tools`` adds them unconditionally,
    regardless of what device its registry was built for (see app.py) --
    so they map to every entry of DEVICES here too. (Task 11: this used to
    add only ``describe_tool``, which silently left ``get_job_status``
    permanently uncoverable -- CASES could never list it without
    ``_check_coverage`` rejecting it as "references unregistered tool(s)",
    even though it is a real tool a client can call through the same
    handler.)
    """
    registered: dict[str, set[str]] = {}
    for device in DEVICES:
        for tool in build_registry(device=device).tools():
            registered.setdefault(tool.name, set()).add(device)
    registered.setdefault(DESCRIBE_TOOL_MANIFEST.name, set()).update(DEVICES)
    registered.setdefault(GET_JOB_STATUS_MANIFEST.name, set()).update(DEVICES)
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


@dataclass(frozen=True)
class CaseOutcome:
    """One case's result -- collected, never raised.

    ``_run_case`` used to ``raise SystemExit`` the instant a case's outcome
    didn't match its expectation, which ``main()`` never caught. That is a
    real defect (see task-13-report.md): a hard exit on the FIRST mismatch
    means this script can never produce a complete picture once any one
    case's real-world outcome differs from what was true when the case was
    written -- which is exactly what an in-container run exists to find,
    and is close to inevitable the first time every case is actually
    executed somewhere new. ``_run_case`` now returns a ``CaseOutcome``
    instead of raising, so ``main()`` can run every case, print a full
    table, and only THEN decide the process exit code -- non-zero if
    anything failed, but only after every case has actually run.
    """

    tool: str
    device: str
    ok: bool
    detail: str = ""


async def _run_case(case: dict) -> CaseOutcome:
    tool = case["tool"]
    device = case["device"]
    expect_keys = case["expect_keys"]
    expect_error = expect_keys == ["error"]

    try:
        result = await call_tool(tool, case["arguments"], device)
    except Exception as exc:  # noqa: BLE001 - one case crashing must not stop the rest
        print(f"\n=== {tool} (device={device}) ===")
        print(f"CRASHED before a result was returned: {exc!r}")
        return CaseOutcome(tool, device, ok=False, detail=f"crashed: {exc!r}")

    _print_result(f"{tool} (device={device})", result)

    if expect_error:
        if not result.isError:
            return CaseOutcome(tool, device, ok=False, detail="expected to fail, it did not")
        return CaseOutcome(tool, device, ok=True)

    if result.isError:
        return CaseOutcome(tool, device, ok=False, detail="expected to succeed, it did not")

    payload = json.loads(result.content[0].text)
    missing_keys = [key for key in expect_keys if key not in payload]
    if missing_keys:
        return CaseOutcome(
            tool, device, ok=False,
            detail=f"result missing expected key(s) {missing_keys}",
        )
    return CaseOutcome(tool, device, ok=True)


def _print_summary(outcomes: list[CaseOutcome]) -> None:
    passed = [o for o in outcomes if o.ok]
    failed = [o for o in outcomes if not o.ok]
    print(f"\n=== SUMMARY: {len(passed)}/{len(outcomes)} passed ===")
    for outcome in outcomes:
        mark = "PASS" if outcome.ok else "FAIL"
        suffix = f" -- {outcome.detail}" if outcome.detail else ""
        print(f"  [{mark}] {outcome.tool} (device={outcome.device}){suffix}")
    if failed:
        print(f"\n{len(failed)} case(s) did not match their expectation:")
        for outcome in failed:
            print(f"  - {outcome.tool} (device={outcome.device}): {outcome.detail}")


async def main() -> int:
    """Run every case, print a full table, and return the process exit
    code -- 0 only if every case matched its expectation. Every case runs
    regardless of earlier failures (see ``CaseOutcome``'s docstring)."""
    _check_coverage()

    outcomes = [await _run_case(case) for case in CASES]
    _print_summary(outcomes)

    if all(o.ok for o in outcomes):
        print("\n=== ALL CHECKS PASSED ===")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
