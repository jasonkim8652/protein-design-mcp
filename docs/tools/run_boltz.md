# run_boltz

**Category:** structure_prediction  
**Engine:** `boltz`  
**Environment:** `/home/jk661/.conda/envs/boltz`  
**GPU required:** yes

> This file is generated from `src/protein_design_mcp/manifests/run_boltz.yaml`. Edit the manifest, then run `python scripts/generate_tool_docs.py`.

## Summary

Predict a multi-chain protein structure with Boltz-2, given an explicit alignment (or none) per chain. Runs the user's own fork of Boltz, not upstream. Reports confidence (ptm, iptm, pLDDT) and writes a PAE matrix usable by run_ipsae. The affinity head is not exposed here -- it is protein-ligand only and returns meaningless numbers for a protein-protein interface; use run_prodigy or run_ipsae for interface quality instead.

## What this is
Boltz-2 all-atom structure prediction, run through the user's own editable
fork at `~/projects/lightning-boltz-dev` rather than upstream
`boltz-community/boltz` -- state this when reporting or comparing results,
since the fork's behaviour is not guaranteed identical to upstream.

## What it is for
Predicting how a set of protein chains fold together: a binder with its
target, a target alone, an oligomer -- whatever `chains` describes. Fast
(single-digit seconds for a small monomer at reduced sampling settings;
tens of seconds to a few minutes at the defaults) relative to the other
structure predictors here.

## When to use this instead of the alternatives
- `run_chai1`, `run_protenix`,
  `run_openfold3` are the direct siblings -- same
  job shape (chains + optional per-chain MSA), different underlying
  model. Comparing their outputs on the SAME `msa` input is exactly what
  this server's MSA-tool-first design exists to make possible; do not
  change engines and alignment source in the same comparison.
- `run_esmfold2` takes no MSA at all and is faster
  still, at some cost to accuracy on hard interfaces.
- This tool never runs Boltz-2's affinity head. If you need a
  protein-ligand binding-affinity estimate, that head exists in the
  underlying engine but is deliberately not wired up here -- for a
  protein-protein interface it returns a number that is not meaningful.
  Use `run_prodigy` (absolute free energy, calibrated on natural
  complexes) or `run_ipsae` (predictor confidence in the interface,
  from this tool's own `pae_npz` output) instead.

## What you must supply
`chains`: one entry per chain in the assembly, each
`{"sequence": "<protein AA string>", "msa": "<path>" | null, "copies": <int, default 1>}`.
List every chain you want folded together in this call -- the target and
the binder together for a complex prediction, the binder alone to score it
in isolation. Nothing is inferred from anything else you have called.
`msa` has no default: pass `null` to run that chain MSA-free (a real,
supported choice, with reduced accuracy), or a path to the `unpaired_a3m`
file `run_mmseqs_search` wrote for that exact sequence. Passing a paired
a3m, or an a3m from a different search tool (e.g. `run_colabfold_search`) here is a format/database mismatch this tool
cannot detect -- it will usually just predict worse, not fail. This wrapper never
lets Boltz reach its own MSA server or run a local search itself, no
matter what `msa` is set to.

## Important caveats
- **Ran the user's own fork, not upstream Boltz** -- always state this
  when reporting results from this tool.
- The affinity head is not exposed (see above) -- this is a permanent
  design decision for this tool, not a missing feature.
- `subsample_msa`'s CLI help text claims a default of `True`; verified
  against the fork's own source that this is a plain flag with no
  explicit default, which resolves to `False` when omitted. This tool's
  own default below (`false`) matches the VERIFIED runtime behaviour, not
  the help text.
- Model is pinned to Boltz-2; Boltz-1 selection is not exposed.

## What you get back
`confidence_score`, `ptm`, `iptm`, `protein_iptm`, `complex_plddt`,
`complex_iplddt`, `complex_pde`, `complex_ipde` (from the rank-0 model),
`num_structures`, a `fork_notice` string, and under `outputs` the paths to
every predicted structure and its per-model confidence/PAE/PDE/pLDDT
files.

## Parameters

| Parameter | Type | Required | Default | Constraints | Description |
|---|---|---|---|---|---|
| `chains` | array | yes | `—` | minItems: `1`<br>maxItems: `12` | One entry per chain in the predicted assembly: {"sequence": "<protein amino-acid string, uppercase, standard 20 plus X/B/Z/J/U/O>", "msa": "<path to run_mmseqs_search's unpaired_a3m output for this sequence>" or null, "copies": <positive integer, default 1, identical copies of this chain -- use for a homo-oligomer instead of repeating the entry>}. Chain composition (who's present, alone or in complex) is never inferred -- list exactly the chains you want folded together. "msa" is required on every entry; there is no default. |
| `recycling_steps` | integer | no | `3` | minimum: `0`<br>maximum: `10` | Number of recycling passes through the trunk before diffusion. More recycling can improve hard interfaces at roughly linear extra cost; 3 is Boltz's own default and a reasonable middle ground. |
| `sampling_steps` | integer | no | `200` | minimum: `1`<br>maximum: `1000` | Number of diffusion denoising steps per sample. Fewer steps is much faster and noticeably lower quality; 200 is Boltz's own default and the setting its reported benchmarks use. Drop this for a fast sanity check, not for a result you intend to keep. |
| `diffusion_samples` | integer | no | `1` | minimum: `1`<br>maximum: `25` | Number of independent structures to sample. Boltz ranks and returns all of them (job_model_0 is the highest-confidence); more samples costs roughly linear GPU time. 1 is Boltz's own default; raise it to get a diverse set to filter downstream. |
| `step_scale` | number | no | `1.5` | minimum: `0.1`<br>maximum: `5.0` | Diffusion step size, which sets the temperature of the sampling process: lower gives more diverse samples, higher gives more deterministic ones (recommended range 1-2 per Boltz's own --help). 1.5 is Boltz-2's own default (Boltz-1's differs, at 1.638, but this tool always runs Boltz-2). |
| `use_potentials` | boolean | no | `False` | — | Whether to steer sampling with Boltz's physical-validity potentials (reduces clashes and bond-geometry violations at some cost to diversity and speed). False is Boltz's own default. |
| `output_format` | string | no | `mmcif` | enum: `['pdb', 'mmcif']` | File format for the written structures. mmcif (Boltz's own default) carries more metadata (e.g. per-residue confidence in B-factor columns is written either way, but mmcif is the format Boltz's own downstream tooling expects); pdb is more broadly compatible with older analysis tools. |
| `max_msa_seqs` | integer | no | `8192` | minimum: `1`<br>maximum: `16384` | Maximum number of MSA rows Boltz will use from the supplied alignment, applied after any of run_mmseqs_search's own filtering. 8192 is Boltz's own default. Lowering it speeds up featurization on a very deep alignment at some cost to co-evolutionary signal. |
| `subsample_msa` | boolean | no | `False` | — | Whether to randomly subsample the MSA down to num_subsampled_msa rows before featurization (a regularisation Boltz's training used). VERIFIED against the fork's source that omitting this flag resolves to false at runtime, despite the CLI's own --help text claiming a default of true -- this tool's default matches the verified runtime behaviour. |
| `num_subsampled_msa` | integer | no | `1024` | minimum: `1`<br>maximum: `16384` | Row count to subsample down to when subsample_msa is true; ignored otherwise. 1024 is Boltz's own default. |
| `seed` | integer | no | `42` | minimum: `0` | Random seed for diffusion sampling. Fixed by default so repeated calls with identical inputs are reproducible; vary it to get a different sample set at the same diffusion_samples count. |
