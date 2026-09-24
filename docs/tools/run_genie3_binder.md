# run_genie3_binder

**Category:** binder_generation  
**Engine:** `genie3`  
**Environment:** `/home/jk661/.conda/envs/genie2_fixed`  
**GPU required:** yes

> This file is generated from `src/protein_design_mcp/manifests/run_genie3_binder.yaml`. Edit the manifest, then run `python scripts/generate_tool_docs.py`.

## Summary

Generate a binder against a target with Genie 3's all-atom SE(3)-equivariant diffusion model, conditioned on a target structure and hotspot residues you supply directly (MSA-free by construction -- Genie 3 never reads an alignment during generation, only during evaluation/reward steps this tool does not expose). Conditions on exactly the hotspot residues you supply by default; set expand_interface: true to instead condition on Genie 3's own SASA-based "extended interface" (surface residues near your hotspots), now available on this host.

## What this is
Genie 3's target-conditioned binder-design generation path
(`genie3.cli generate`, dataset source `target`): given a target structure
and a set of hotspot residues on it, samples a new chain's coordinates
conditioned on that interface. Backbone frames only -- see "What is NOT
exposed" for why the side-chain stage is unreachable here.

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

## `expand_interface` -- exact hotspots (default) vs. Genie 3's own extended interface
By default (`expand_interface: false`) this tool conditions on EXACTLY the
hotspot residues you supply (`cond_strategy: "hotspot"`, Genie 3's own
literal, unexpanded interface definition) -- not a larger neighborhood.
Genie 3's own binder-design pipeline (`scripts/problem/binder_design/
prepare.py`) normally also computes an "extended interface" -- surface
residues near each hotspot, via a Shrake-Rupley solvent-accessible-surface
calculation (`genie3.generation.utils.interface.extended
.compute_extended_interface`, which needs `Bio.PDB`). That is now
available here too: set `expand_interface: true` to run the exact same
computation `prepare.py` runs (`version_num=1`, so your literal hotspots
are always folded into the expanded set too) and condition on the result
instead (`cond_strategy: "extended"`). Confirmed live against a real
target (108-residue barnase, 3 real interface hotspots): expanded to 16
surface residues within `interface_cutoff_angstrom` of any hotspot atom.
`interface_cutoff_angstrom`, `interface_rsa_threshold` and
`interface_abs_sasa_threshold` are `compute_extended_interface`'s own
tunable knobs, exposed here rather than hardcoded; all three are ignored
when `expand_interface` is false. The reply's `cond_strategy` and
`extended_interface_residues` (also written to the `interface_conditioning`
output on every call) report exactly what was used, so this is never
silent about which mode ran. If your target has a known broader interface
you want conditioned on exactly, supply every residue in it directly as
`hotspot_residues` instead of using `expand_interface`.

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
- Design a sequence for the binder if you need one -- this tool emits
  backbone frames, so fold the result and run `run_mpnn` on it.

## What is NOT exposed, and why
- **`predict_sequence`** is fixed to `false` -- sequence design is
  `run_mpnn`'s job, the same rule as every other generative tool here.
- **`predict_sidechain`** is not exposed, although `run_genie3_scaffold`
  does expose it. Genie 3's side-chain pass is a second stage guarded by
  `assert config.dataset.source == "unconditional"`
  (`genie3/generation/workflow.py`), and binder generation runs with
  `source == "target"` (`runner.py`, `postprocess.py`), so the assertion can
  never hold here. The guard sits AFTER `[1 / 2] Main stage completed!`, so
  setting it would run the whole generation and then discard it -- confirmed
  live, 52 s of GPU lost to an AssertionError. Fold the backbone and run
  `run_mpnn` for an all-atom model instead.
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
fixed yourself rather than trust this description. `num_binders`,
`cond_strategy` (`"hotspot"` or `"extended"` -- which one actually ran),
`extended_interface_residues` (null unless `expand_interface` was true,
else the expanded tag list actually conditioned on), and under `outputs`
the path to every generated PDB plus `interface_conditioning` (the same
conditioning data as a file).

## Parameters

| Parameter | Type | Required | Default | Constraints | Description |
|---|---|---|---|---|---|
| `target_pdb` | string | yes | `—` | pattern: `\.(pdb\|cif)$` | Structure file containing ONLY the target (one or more chains, no existing binder chain). hotspot_residues below refer to this file's own chain identifiers and residue numbers. WHERE THIS COMES FROM -- The target you want a binder against -- your own structure file, or a path a previous step returned. Target ONLY, with no existing binder chain. |
| `hotspot_residues` | array | yes | `—` | minItems: `1`<br>maxItems: `50` | Residues on target_pdb to condition the binder against, each "{chain_id}{residue_index}" (optionally with a trailing insertion-code letter), e.g. "A19". Used exactly as given -- see the doc for why this tool does not expand them into a larger interface. A malformed tag (missing chain, non-numeric residue index) is rejected here rather than failing inside the engine. |
| `binder_min_length` | integer | yes | `—` | minimum: `5`<br>maximum: `300` | Shortest binder chain length to sample, inclusive. Every sample's length is drawn uniformly from [binder_min_length, binder_max_length] independently. |
| `binder_max_length` | integer | yes | `—` | minimum: `5`<br>maximum: `300` | Longest binder chain length to sample, inclusive. Set equal to binder_min_length to fix a single length. |
| `num_samples` | integer | no | `2` | minimum: `1`<br>maximum: `200` | Total number of binder samples to draw against this target. |
| `model_variant` | string | no | `v1` | enum: `['v1', 'legacy']` | Which trained checkpoint and sampler to use -- see run_genie3_scaffold's doc for what each variant is (identical checkpoints, same tradeoffs). `legacy` ignores every DDIM-only parameter below. |
| `direction_scale` | number | no | `1.0` | minimum: `0.0`<br>maximum: `5.0` | DDIM sampler only (ignored for model_variant=legacy). Scales the predicted denoising direction at each step; Genie 3's own config requires an explicit value (no shipped default). |
| `eta` | number | no | `1.0` | minimum: `0.0`<br>maximum: `1.0` | DDIM sampler only (ignored for model_variant=legacy). Stochasticity: 0 is deterministic (ODE-like), 1 is DDPM-equivalent stochastic sampling. Genie 3's own default is 1.0. |
| `n_sample_step` | integer | no | `100` | minimum: `1`<br>maximum: `1000` | DDIM sampler only (ignored for model_variant=legacy). Number of denoising steps out of the model's fixed 1000-step training schedule. Genie 3's own default is 100; more steps is finer-grained at roughly linear cost. |
| `noise_scale` | number | no | `1.0` | minimum: `0.0`<br>maximum: `2.0` | Scales injected noise at each denoising step. Applies to both samplers. Genie 3's own default is 1.0 for both. |
| `seed` | integer | no | `0` | minimum: `0` | Random seed for the diffusion trajectory. Two runs with the same seed and the same parameters return the same backbones, so change it to sample a different set rather than re-running and expecting variety. It does not affect quality, only which samples you get. |
| `expand_interface` | boolean | no | `False` | — | false (the default) conditions on EXACTLY hotspot_residues, unchanged from this tool's original behavior. true runs Genie 3's own SASA-based "extended interface" expansion (compute_extended_interface, confirmed live -- see the doc) and conditions on that larger surface patch instead; your literal hotspots are always included in it too. Use this when you have a few known hotspots but want the binder to also engage the surrounding surface, rather than only the exact residues you named. |
| `interface_cutoff_angstrom` | number | no | `6.0` | minimum: `1.0`<br>maximum: `20.0` | Ignored unless expand_interface is true. A surface residue is included in the expanded interface if any of its heavy atoms is within this distance of any heavy atom of a hotspot residue. compute_extended_interface's own default is 6.0. |
| `interface_rsa_threshold` | number | no | `0.25` | minimum: `0.0`<br>maximum: `1.0` | Ignored unless expand_interface is true. Relative solvent accessibility above which a residue counts as "surface" and is eligible for the expanded interface at all (a buried residue within interface_cutoff_angstrom is still excluded). compute_extended_interface's own default is 0.25. |
| `interface_abs_sasa_threshold` | number | no | `10.0` | minimum: `0.0`<br>maximum: `300.0` | Ignored unless expand_interface is true. Absolute SASA (Angstroms^2) floor used alongside interface_rsa_threshold in the same surface classification. compute_extended_interface's own default is 10.0. |
