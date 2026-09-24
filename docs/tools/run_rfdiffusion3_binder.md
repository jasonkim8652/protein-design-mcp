# run_rfdiffusion3_binder

**Category:** binder_generation  
**Engine:** `rfd3`  
**Environment:** `/home/jk661/.conda/envs/foundry`  
**GPU required:** yes

> This file is generated from `src/protein_design_mcp/manifests/run_rfdiffusion3_binder.yaml`. Edit the manifest, then run `python scripts/generate_tool_docs.py`.

## Summary

Generate de novo binder backbones against a target with RFdiffusion3 (RosettaCommons `rc-foundry`'s rfd3, inference_engine=rfdiffusion3) -- RosettaCommons' current production binder-design model. Backbone only (pair with run_mpnn). Conditions through one pydantic-validated JSON schema shared with run_rfdiffusion3_scaffold: contig plus select_hotspots for the target and interface, in a comma-separated grammar that looks similar to but is NOT the same as run_rfdiffusion_binder's or run_rfdiffusion2's space-separated contig grammar -- see the doc before reusing a contig string across these tools.

## What this is
RFdiffusion3 (RosettaCommons `rc-foundry`'s `rfd3` package,
`inference_engine=rfdiffusion3`), a ground-up rebuild of the RFdiffusion
lineage on a pydantic-validated, atom-level conditioning schema rather
than the older per-generation ad-hoc contig strings. This is the binder
(protein-protein interface) path: `contig` plus `select_hotspots` against
an existing target structure. The unconditional/motif-scaffolding
sibling path is `run_rfdiffusion3_scaffold` -- same engine, same JSON
schema, no target chain.

## When to use this instead of the alternatives -- three generations, one lineage
- `run_rfdiffusion_binder` (RFdiffusion 1.1.0): the legacy baseline.
  Simpler, longest-battle-tested space-separated contig grammar
  (`contigs`/`hotspot_res`), no native ligand/nucleic-acid awareness.
- `run_rfdiffusion2`: the all-atom, ligand/nucleic-acid-aware successor,
  conditioned through the SAME grammar family as `run_rfdiffusion_binder`
  (`contigmap.contigs`/`ppi.hotspot_res`) -- not this tool's grammar.
  Its documented invocation needs Apptainer, not installed on this host;
  this server runs it through a config-loading workaround instead, and a
  full generation could not be verified live here (see that tool's own
  doc for the exact blocker) -- prefer THIS tool unless you specifically
  need RFdiffusion2.
- **This tool**: RosettaCommons' current production binder model, a
  completely different, pydantic-validated JSON conditioning schema (see
  "`contig` and `select_hotspots`" below) -- comma-separated with `/0`
  chain breaks and NO spaces, unlike the other two generations' grammars.
  No Apptainer dependency; CONFIRMED LIVE end to end on this host
  (GPU 7). Prefer this one for new binder-design work unless you
  specifically need RFdiffusion 1.1.0 or RFdiffusion2 behaviour.
- The `_binder` (this tool) vs `_scaffold` (`run_rfdiffusion3_scaffold`)
  split within RFdiffusion3 itself is purely "is there a target chain
  present" -- both run the identical `rfd3 design` entry point and JSON
  schema; `run_rfdiffusion3_scaffold` never has a target so it has no
  `select_hotspots`.

## `contig` and `select_hotspots` -- RFdiffusion3's own grammar
Comma-separated tokens (NO spaces, unlike RFdiffusion 1.1.0/RFdiffusion2's
space-separated grammar), each one of:
- a diffused-length segment: `<min>-<max>` (e.g. `50-50` for a fixed
  50-residue binder) -- by convention placed FIRST in `contig` for a
  binder design (the reverse of RFdiffusion 1.1.0's target-first
  convention).
- `/0`, a chain break, as its own token.
- a target segment: `<Chain><start>-<end>` or a single residue
  `<Chain><residue>` (e.g. `A1-150`, `A62`), pulled from `target_pdb`.
Worked example -- a 50-residue binder against the whole of target chain A
(150 residues): `contig: "50-50,/0,A1-150"`. `length` optionally
constrains the OVERALL total (motif + diffused) length; when the
diffused segment in `contig` is already a fixed range, `length` is
usually redundant with it and can be left unset.
`select_hotspots` uses the SAME per-residue token grammar (no length
ranges needed there): `"A62,A89,A121"`. CONFIRMED LIVE with
`target_pdb`/`contig`/`select_hotspots` together (GPU 7, 2026-09-22): a
real binder-mode sample ran end to end and produced a structure plus
metrics JSON.

## `redesign_motif_sidechains`
When true, the target's own side chains within `contig` are redesigned
(fixed backbone, diffused sequence) rather than copied verbatim from
`target_pdb`. False (the default) keeps the target exactly as given --
the ordinary binder-design setting, since you almost always want the
real target, not a redesigned one.

## Partial diffusion (`partial_t`)
Set to noise an existing binder+target complex for this many Angstroms
(not timesteps, unlike RFdiffusion 1.1.0's `partial_t`) and denoise it
back out -- diversifies an existing complex while mostly preserving its
fold. Requires `target_pdb` to already contain something for `contig` to
select (RFdiffusion3 partial diffusion always starts from a real
structure, never from a bare length). Omit for ordinary from-scratch
generation.

## What you must supply
`target_pdb` and `contig` together (this tool's target-conditioned
binder path; use `run_rfdiffusion3_scaffold` for no-target generation).
`select_hotspots` is optional but strongly recommended for the same
reason as in the other two generations -- without it, the binder can
attach anywhere the contig geometry allows.

## Important caveats
- **Backbone only.** RFdiffusion3 does emit a sequence (via
  `read_sequence_from_sequence_head`, always on in this wrapper) but it
  is not validated the way ProteinMPNN's is -- treat it the same as the
  other two generations' output and run `run_mpnn` before trusting any
  sequence.
- Ligand and nucleic-acid conditioning, symmetry, and the
  hydrogen-bond/RASA-accessibility conditioning fields RFdiffusion3's
  schema also supports are not exposed here -- none were exercised in
  this tool's verification and this server does not invent flags it has
  not verified.

## What you get back
`num_structures`, `metrics` (from the rank-0 -- first -- sample's own
sidecar JSON: clash counts, secondary-structure fractions, radius of
gyration), `diffused_index_map` (target-residue renumbering, from the
same sample), `ckpt_path`, and under `outputs` the paths to every
`structure_cif` and its paired `metadata_json`.

## Parameters

| Parameter | Type | Required | Default | Constraints | Description |
|---|---|---|---|---|---|
| `target_pdb` | string | yes | `—` | pattern: `\.(pdb\|cif)$` | Path to the target structure (PDB or mmCIF). WHERE THIS COMES FROM -- The target you want a binder against -- your own structure file, or a path a previous step returned. |
| `contig` | string | yes | `—` | pattern: `^([A-Za-z]\d+(-\d+)?\|/\d+\|\d+(-\d+)?)(,([A-Za-z]\d+(-\d+)?\|/\d+\|\d+(-\d+)?))*$` | RFdiffusion3's own contig grammar (see the doc's "contig and select_hotspots" section) -- comma-separated tokens, NO spaces: a diffused-length range like "50-50", a bare "/0" chain break, or a target segment like "A1-150" pulled from target_pdb. By convention the diffused (binder) segment comes FIRST for a binder design, the opposite order from run_rfdiffusion_binder's contigs. |
| `select_hotspots` | string | no | `—` | pattern: `^[A-Za-z]\d+(-\d+)?(,[A-Za-z]\d+(-\d+)?)*$` | Target residues the binder must contact, comma-separated tokens of "<ChainID><ResidueNumber>" or "<ChainID><start>-<end>" (no spaces). Omit (the default, null) to run without hotspot conditioning -- a real but not recommended choice; the binder can then attach anywhere the contig geometry allows. |
| `length` | string | no | `—` | pattern: `^\d+(-\d+)?$` | Optional overall length constraint ("min-max" or a fixed integer) on the full built structure (diffused + target residues together). Usually redundant with an already-fixed diffused range in contig and can be left unset; use it when contig's diffused segment is itself a range (e.g. "40-60") and you additionally want to cap the total. |
| `redesign_motif_sidechains` | boolean | no | `False` | — | When true, redesign the target's side chains within contig (fixed backbone, diffused sequence) instead of keeping them exactly as given in target_pdb. False (the default, and the ordinary binder-design setting) keeps the real target unmodified. |
| `partial_t` | number | no | `—` | minimum: `0.0`<br>maximum: `15.0` | Enables partial diffusion: add this many Angstroms of coordinate noise to the EXISTING complex in target_pdb (per contig's selection) and denoise it back out, diversifying it while mostly preserving its fold. Angstroms, not timesteps (unlike run_rfdiffusion_binder's partial_t). Upstream recommends <= 15. Omit (the default) for ordinary from-scratch generation. |
| `diffusion_batch_size` | integer | no | `8` | minimum: `1`<br>maximum: `64` | Number of independent structures to sample in one call. 8 is RFdiffusion3's own default; cost scales roughly linearly. Drop it to 1 for a fast sanity check on a new contig/hotspot combination. |
| `num_timesteps` | integer | no | `200` | minimum: `1`<br>maximum: `1000` | Number of diffusion denoising steps. 200 is RFdiffusion3's own default; fewer is much faster and noticeably lower quality -- use a small value only for a structural sanity check, not a design you intend to keep. |
| `step_scale` | number | no | `1.5` | minimum: `1.0`<br>maximum: `2.0` | Diffusion step size. Higher values give less diverse, more designable structures; lower values give more diversity at some designability cost (per RFdiffusion3's own config comments). 1.5 is RFdiffusion3's own default. |
| `seed` | integer | no | `—` | minimum: `0` | Random seed for diffusion sampling. Omit (the default, null) for a fresh random seed each call; set it for a reproducible sample set at the same diffusion_batch_size. |
