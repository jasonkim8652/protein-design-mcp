# run_genie3_binder

**Category:** binder_generation  
**Engine:** `genie3`  
**Environment:** `/home/jk661/.conda/envs/genie2`  
**GPU required:** yes

> This file is generated from `src/protein_design_mcp/manifests/run_genie3_binder.yaml`. Edit the manifest, then run `python scripts/generate_tool_docs.py`.

## Summary

Generate a binder against a target with Genie 3's all-atom SE(3)-equivariant diffusion model, conditioned on a target structure and hotspot residues you supply directly (MSA-free by construction -- Genie 3 never reads an alignment during generation, only during evaluation/reward steps this tool does not expose). Only hotspot conditioning is available on this host (see the doc); the "extended interface" mode needs Biopython, which is not installed in the environment Genie 3 runs under here.

## What this is
Genie 3's target-conditioned binder-design generation path
(`genie3.cli generate`, dataset source `target`): given a target structure
and a set of hotspot residues on it, samples a new chain's coordinates
(and, with `predict_sidechain: true`, side-chain atoms) conditioned on
that interface.

## `target_pdb` and `hotspot_residues` -- the parameters that decide whether this tool is usable at all
`target_pdb` is a structure file containing ONLY the target (one or more
chains) -- no existing binder chain. `hotspot_residues` is a list of
`{chain_id}{residue_index}` tags picking specific residues ON THAT
STRUCTURE to condition the binder against, e.g. `["A19", "A76", "A80"]`
for chain A residues 19, 76 and 80 (an optional trailing insertion-code
letter is also accepted, e.g. `"A113A"`). These are read directly from
Genie 3's own `parse_hotspot_tags` grammar (`genie3/generation/utils/
interface/extended.py`, read from source) -- a malformed tag like `"19"`
(no chain) or `"chainA19"` is rejected by THIS tool's own schema pattern
before it ever reaches the engine, rather than failing deep inside a
subprocess.

## Only hotspot conditioning is available here -- confirmed, not assumed
Genie 3's own binder-design pipeline (`scripts/problem/binder_design/
prepare.py`) normally also computes an "extended interface" -- surface
residues near each hotspot, via a Shrake-Rupley solvent-accessible-surface
calculation (`compute_extended_interface`, which imports `Bio.PDB`).
Confirmed live: the `genie2` environment Genie 3 runs under on this host
does not have Biopython installed (`ModuleNotFoundError: No module named
'Bio'`). Rather than reimplementing an SASA algorithm outside what this
tool has verified, this tool conditions on EXACTLY the hotspot residues
you supply (`cond_strategy: "hotspot"`, Genie 3's own literal, unexpanded
interface definition) -- not a larger neighborhood. If your target has a
known broader interface, supply every residue in it directly as
`hotspot_residues`.

## MSA -- never built, confirmed from source
`target_msa_filepath` (which Genie 3's own `prepare.py` builds via
`colabfold_batch`) is read ONLY by evaluation/reward code
(`genie3/evaluation/model/fold/*.py`, `genie3/generation/diffusion/
reward/colabfold.py`'s `mode: "msa"` reward path) -- confirmed by reading
every reference to it in the codebase. Core generation
(`create_np_features_from_target_config`, what this tool actually calls)
never reads it. This tool therefore never builds an alignment at all,
which is also why it never contacts a remote MSA server with a target
sequence that may be under embargo.

## When to use this instead of the alternatives
- `run_protpardelle` is the direct sibling -- another target-conditioned
  binder generator, using RFdiffusion-style contig strings instead of a
  target-PDB-plus-hotspots JSON problem, and a materially different
  (non-latent-diffusion) architecture.
- Neither this tool nor `run_protpardelle` scores the result. Refold with
  `run_chai1` or `run_boltz` with the target chain present, then score
  the interface with `run_ipsae`.
- Design a sequence for the target chain if you need one --
  `predict_sidechain: true` gives side-chain ATOMS for the binder chain
  Genie 3 itself placed, not an independently verified sequence; for that,
  fold the result and run `run_mpnn` on it instead.

## What is NOT exposed, and why
- **`predict_sequence`** is fixed to `false` -- sequence design is
  `run_mpnn`'s job, the same rule as every other generative tool here.
- **Multi-round / iterative binder design** (Genie 3's own
  `cond_strategy: iter_common`, which recomputes an interface from prior
  rounds' successes) is an orchestration pattern over several calls to
  this same generation step, not a single generation call -- out of scope
  for one tool invocation.
- **Beam search / reward-guided sampling** (`generation.inference.search`)
  is not exposed -- it requires ColabFold (the `colabfold` reward mode) or
  other scoring machinery this tool does not run; score candidates
  yourself afterward instead.

## What you must supply
`target_pdb`, `hotspot_residues`, `binder_min_length`, `binder_max_length`
-- every one of these defines the design task itself and has no safe
default.

## What you get back
`binders`: one entry per generated PDB, each `{"id", "length",
"binder_chain_id", "chain_lengths"}`. Each PDB holds the FULL COMPLEX
(see the `outputs:` note above) -- `length` and `binder_chain_id` name
ONLY the generated binder chain specifically (confirmed live: Genie 3
always writes it as the file's first chain, since
`create_np_features_from_target_config` concatenates the binder's
features before the target's, read from source); `chain_lengths` gives
every chain's own length, target included, so you can verify what stayed
fixed yourself rather than trust this description. `num_binders`, and
under `outputs` the path to every generated PDB.

## Parameters

| Parameter | Type | Required | Default | Constraints | Description |
|---|---|---|---|---|---|
| `target_pdb` | string | yes | `—` | pattern: `\.(pdb\|cif)$` | Structure file containing ONLY the target (one or more chains, no existing binder chain). hotspot_residues below refer to this file's own chain identifiers and residue numbers. |
| `hotspot_residues` | array | yes | `—` | minItems: `1`<br>maxItems: `50` | Residues on target_pdb to condition the binder against, each "{chain_id}{residue_index}" (optionally with a trailing insertion-code letter), e.g. "A19". Used exactly as given -- see the doc for why this tool does not expand them into a larger interface. A malformed tag (missing chain, non-numeric residue index) is rejected here rather than failing inside the engine. |
| `binder_min_length` | integer | yes | `—` | minimum: `5`<br>maximum: `300` | Shortest binder chain length to sample, inclusive. Every sample's length is drawn uniformly from [binder_min_length, binder_max_length] independently. |
| `binder_max_length` | integer | yes | `—` | minimum: `5`<br>maximum: `300` | Longest binder chain length to sample, inclusive. Set equal to binder_min_length to fix a single length. |
| `num_samples` | integer | no | `2` | minimum: `1`<br>maximum: `200` | Total number of binder samples to draw against this target. |
| `model_variant` | string | no | `v1` | enum: `['v1', 'legacy']` | Which trained checkpoint and sampler to use -- see run_genie3_scaffold's doc for what each variant is (identical checkpoints, same tradeoffs). `legacy` ignores every DDIM-only parameter below. |
| `direction_scale` | number | no | `1.0` | minimum: `0.0`<br>maximum: `5.0` | DDIM sampler only (ignored for model_variant=legacy). Scales the predicted denoising direction at each step; Genie 3's own config requires an explicit value (no shipped default). |
| `eta` | number | no | `1.0` | minimum: `0.0`<br>maximum: `1.0` | DDIM sampler only (ignored for model_variant=legacy). Stochasticity: 0 is deterministic (ODE-like), 1 is DDPM-equivalent stochastic sampling. Genie 3's own default is 1.0. |
| `n_sample_step` | integer | no | `100` | minimum: `1`<br>maximum: `1000` | DDIM sampler only (ignored for model_variant=legacy). Number of denoising steps out of the model's fixed 1000-step training schedule. Genie 3's own default is 100; more steps is finer-grained at roughly linear cost. |
| `noise_scale` | number | no | `1.0` | minimum: `0.0`<br>maximum: `2.0` | Scales injected noise at each denoising step. Applies to both samplers. Genie 3's own default is 1.0 for both. |
| `predict_sidechain` | boolean | no | `False` | — | DDIM sampler only (ignored for model_variant=legacy). Whether the model also predicts side-chain atoms for the binder chain (all-atom output) rather than backbone frames only. Genie 3's own default is false. |
| `seed` | integer | no | `0` | minimum: `0` | Random seed. |
