# run_rfdiffusion3_scaffold

**Category:** monomer_generation  
**Engine:** `rfd3`  
**Environment:** `/home/jk661/.conda/envs/foundry`  
**GPU required:** yes

> This file is generated from `src/protein_design_mcp/manifests/run_rfdiffusion3_scaffold.yaml`. Edit the manifest, then run `python scripts/generate_tool_docs.py`.

## Summary

Generate a de novo monomer backbone with RFdiffusion3 (RosettaCommons `rc-foundry`'s rfd3, inference_engine=rfdiffusion3) -- RosettaCommons' current production scaffold model. No target/binding partner (use run_rfdiffusion3_binder for that); either fully unconditional at a given length, or scaffolding around a fixed functional motif pulled from an existing small structure. Backbone only (pair with run_mpnn). CONFIRMED LIVE (unconditional path) on this host, GPU 7.

## What this is
RFdiffusion3, the monomer/scaffold path: no target chain, so no
`select_hotspots` (contrast `run_rfdiffusion3_binder`, the binder/PPI
sibling on the exact same engine and JSON schema). Two uses:
- **Unconditional**: give only `length`; RFdiffusion3 generates a
  backbone of that length with no other constraint. CONFIRMED LIVE
  (2026-09-22, GPU 7): a 25-residue unconditional sample ran end to end.
- **Motif scaffolding**: give `motif_pdb` and `motif_contig` together to
  fix a functional motif (e.g. a binding loop or catalytic site) pulled
  from an existing small structure, with the surrounding scaffold
  diffused around it.

## When to use this instead of the alternatives
This is the monomer_generation-category sibling of a three-generation
binder_generation lineage this server also exposes
(`run_rfdiffusion_binder`, `run_rfdiffusion2`, `run_rfdiffusion3_binder|
see that tool's doc for the full three-way comparison). For scaffold-only
work (no target), the other options in this server are `run_genie2`,
`run_frameflow`, `run_multiflow` and `run_la_proteina` -- all different
underlying generative models with different strengths; this tool's
distinguishing feature versus those is exact fixed-motif scaffolding
through the same pydantic-validated conditioning schema
`run_rfdiffusion3_binder` uses for targets.

## `motif_contig` -- same grammar as run_rfdiffusion3_binder's `contig`
Comma-separated tokens, no spaces: a diffused-length range (e.g.
`30-30`), a bare `/0` chain break, or a motif segment pulled from
`motif_pdb` (e.g. `A10-15`). Worked example -- a 50-residue scaffold
around a fixed 6-residue motif (chain A, residues 10-15):
`motif_contig: "20-20,A10-15,24-24"`. Requires `motif_pdb` to be set;
this tool raises a clear error if one is given without the other rather
than letting the engine fail with a less specific pydantic error.

## `redesign_motif_sidechains`
When true (and `motif_contig`/`motif_pdb` are set), the motif's own side
chains are redesigned (fixed backbone, diffused sequence) rather than
copied verbatim from `motif_pdb`. False (the default) keeps the motif's
real side chains, which matters when the motif's function (e.g. a
binding surface or catalytic geometry) depends on specific side-chain
identities.

## Important caveats
- **Backbone only** -- run `run_mpnn` before trusting any sequence RFD3
  itself emits.
- Symmetry (cyclic/dihedral oligomer scaffolding, which RFdiffusion3's
  schema also supports) is not exposed here -- not exercised in this
  tool's verification.

## What you get back
`num_structures`, `metrics` (from the rank-0 sample's sidecar: clash
counts, secondary-structure fractions, radius of gyration), `ckpt_path`,
and under `outputs` the paths to every `structure_cif` and its paired
`metadata_json`.

## Parameters

| Parameter | Type | Required | Default | Constraints | Description |
|---|---|---|---|---|---|
| `length` | string | yes | `—` | pattern: `^\d+(-\d+)?$` | Total backbone length: a fixed integer (e.g. "80") or a "min-max" range (e.g. "60-100", sampled once per design). When motif_contig is also set, this is the length of the FULL built structure (motif plus diffused scaffold together), not the diffused portion alone. |
| `motif_pdb` | string | no | `—` | pattern: `\.(pdb\|cif)$` | Path to a structure containing the functional motif to scaffold around. Required together with motif_contig (motif scaffolding); omit both for unconditional generation. |
| `motif_contig` | string | no | `—` | pattern: `^([A-Za-z]\d+(-\d+)?\|/\d+\|\d+(-\d+)?)(,([A-Za-z]\d+(-\d+)?\|/\d+\|\d+(-\d+)?))*$` | RFdiffusion3's contig grammar (see the doc's "motif_contig" section): comma-separated tokens, no spaces -- a diffused-length range like "20-20", a bare "/0" chain break, or a motif segment like "A10-15" pulled from motif_pdb. Required together with motif_pdb; omit both for unconditional generation. |
| `redesign_motif_sidechains` | boolean | no | `False` | — | Only meaningful when motif_contig/motif_pdb are set. When true, redesign the motif's side chains (fixed backbone, diffused sequence) instead of keeping them exactly as given. False (the default) keeps the real motif side chains -- usually required when the motif's function depends on specific side-chain identities. |
| `diffusion_batch_size` | integer | no | `8` | minimum: `1`<br>maximum: `64` | Number of independent structures to sample in one call. 8 is RFdiffusion3's own default; cost scales roughly linearly. Drop it to 1 for a fast sanity check on a new length/motif combination. |
| `num_timesteps` | integer | no | `200` | minimum: `1`<br>maximum: `1000` | Number of diffusion denoising steps. 200 is RFdiffusion3's own default; fewer is much faster and noticeably lower quality -- use a small value only for a structural sanity check, not a design you intend to keep. |
| `step_scale` | number | no | `1.5` | minimum: `1.0`<br>maximum: `2.0` | Diffusion step size. Higher values give less diverse, more designable structures; lower values give more diversity at some designability cost. 1.5 is RFdiffusion3's own default. |
| `seed` | integer | no | `—` | minimum: `0` | Random seed for diffusion sampling. Omit (the default, null) for a fresh random seed each call; set it for a reproducible sample set at the same diffusion_batch_size. |
