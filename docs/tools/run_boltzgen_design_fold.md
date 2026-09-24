# run_boltzgen_design_fold

**Category:** structure_prediction  
**Engine:** `boltzgen`  
**Environment:** `/home/jk661/miniforge3/envs/boltzgen`  
**GPU required:** yes

> This file is generated from `src/protein_design_mcp/manifests/run_boltzgen_design_fold.yaml`. Edit the manifest, then run `python scripts/generate_tool_docs.py`.

## Summary

Refold a design ALONE (target chain(s) removed) and check it still folds into its own generated shape -- BoltzGen's own `design_folding` pipeline step (mode of the same Predict/`fold.yaml` task `run_boltzgen_fold` wraps, with the design isolated from its target). A self-consistency check, not an interface confidence estimate: a design that scores well WITH its target present (`run_boltzgen_fold`) but folds into something different ALONE is a design the target coordinates are propping up, not one with real standalone secondary/tertiary structure. Feed it the WHOLE `outputs.generated_designs` (or `outputs.inverse_folded_designs`) list from run_boltzgen_design or run_boltzgen_inverse_fold -- both the `.cif` and its `.npz` are required, not just the structure file.

## What this is
BoltzGen's `design_folding` pipeline step -- the SAME underlying task as
`run_boltzgen_fold` (`boltzgen.task.predict.predict.Predict` over
`fold.yaml`, confidence-model checkpoint `boltz2_conf_final.ckpt`), run
with two extra flags BoltzGen's own CLI sets for this mode
(`writer.designfolding=true`, `data.cfg.return_designfolding=true`):
before refolding, the TARGET chain(s) are stripped out of the input
structure, and only the designed residues are kept. The comparison
metrics this mode is FOR (design-alone-vs-with-target consistency) are
computed downstream, in `run_boltzgen_analyze`'s
`designfolding-*`-prefixed columns -- this tool by itself only refolds
the isolated design and reports its own confidence.

## `--protocol` has no effect on this step, verified the same way as
`run_boltzgen_fold`: diffing `boltzgen configure --steps design_folding`'s
resolved `fold.yaml` across all six protocols shows no protocol touching
`"design_folding"` in `cli/boltzgen.py`'s `protocol_configs` dict at all
(only `"folding"` gets a `protein-redesign` override, and this tool is a
different step). BoltzGen's own CLI decides WHETHER to run this step at
all from the protocol (`protein-anything`/`protein-small_molecule` run it
by default, the peptide/nanobody/antibody protocols do not) -- but that
choice is exactly what calling this tool, or not, already IS here, so
there is nothing left for a `protocol` parameter to control.

## When to use this instead of the alternatives
- `run_boltzgen_fold` refolds the design WITH its target present -- the
  interface confidence estimate everything downstream actually ranks on.
  Run this tool IN ADDITION when you also want the self-consistency check
  (BoltzGen's own `protein-anything`/`protein-small_molecule` protocols
  run both by default).
- This is not a general-purpose structure predictor -- its output feeds
  `run_boltzgen_analyze` in the exact shape that step expects (BoltzGen's
  own `.npz` metadata format under `fold_out_design_npz/`), not a generic
  confidence JSON.

## What you must supply
- `design_spec`: the SAME design specification YAML the designs came
  from. Required only because `boltzgen run` validates every design spec
  it is given before running any step at all; its content does not
  change this step's own numbers.
- `generated_files`: the WHOLE `outputs.generated_designs` list from
  `run_boltzgen_design`, or `outputs.inverse_folded_designs` from
  `run_boltzgen_inverse_fold` -- every `.cif` AND every `.npz` that call
  returned, unfiltered.

## What you get back
`refolds`: one entry per design, `{"id", "design_ptm", "design_iptm",
"design_iiptm", "min_interaction_pae", "min_design_to_target_pae",
"interaction_pae", "iptm", "ptm", "complex_plddt", "complex_iplddt",
"complex_pde", "complex_ipde", "num_samples", "best_sample_index"}` --
note several target-relative fields `run_boltzgen_fold` reports
(`design_to_target_iptm`, `protein_iptm`, `target_ptm`, `ligand_iptm`,
`design_residue_iptm`) are not meaningful here (the target is absent from
the input entirely) and are omitted rather than reported as a
degenerate/placeholder number. `num_refolds`, and under `outputs` the
paths to every refolded (design-only) structure and the full per-sample
metrics.

## Parameters

| Parameter | Type | Required | Default | Constraints | Description |
|---|---|---|---|---|---|
| `design_spec` | string | yes | `—` | pattern: `\.(yaml\|yml)$` | The design specification YAML the designs came from. Its content has no effect on this step's own numbers -- required only because `boltzgen run` validates every design spec it is given before running any step, `--steps design_folding` included. YOU WRITE THIS FILE: no tool on this server produces a BoltzGen design spec, so it cannot come from a previous step of a planned workflow. A workflow that reaches any run_boltzgen_* tool has to carry a spec path the caller authored -- confirmed by a run that planned run_boltzgen_fold after RFdiffusion3 and failed on this parameter, because there was nothing upstream that could have supplied one. |
| `generated_files` | array | yes | `—` | minItems: `2` | The WHOLE `outputs.generated_designs` (from run_boltzgen_design) or `outputs.inverse_folded_designs` (from run_boltzgen_inverse_fold) list -- every `.cif` AND `.npz` path that call returned, passed through unfiltered. Both file types for at least one design are required (hence minItems: 2); dropping the `.npz` half leaves this tool with no way to read the design back. |
| `folding_checkpoint` | string | no | `huggingface:boltzgen/boltzgen-1:boltz2_conf_final.ckpt` | — | Path or huggingface:repo:file reference for the folding (confidence model) checkpoint -- the SAME one `run_boltzgen_fold` uses, this mode only changes what structure is fed to it. BoltzGen's own default, already cached on this host (~2.0G). |
| `recycling_steps` | integer | no | `3` | minimum: `0`<br>maximum: `10` | Number of recycling passes through the trunk before diffusion. 3 is BoltzGen's own default for this step. |
| `sampling_steps` | integer | no | `200` | minimum: `1`<br>maximum: `1000` | Number of diffusion denoising steps per refold sample. Fewer is faster and lower quality; 200 is BoltzGen's own default for this step. |
| `diffusion_samples` | integer | no | `5` | minimum: `1`<br>maximum: `25` | Number of independent refold samples per design. This tool reports the highest-confidence one as its headline metrics, but `refold_metrics` carries the full spread. 5 is BoltzGen's own default for this step. |
| `use_kernels` | string | no | `auto` | enum: `['auto', 'true', 'false']` | Whether to use BoltzGen's fused CUDA kernels. "auto" (BoltzGen's own default) enables them when the GPU's compute capability is >= 8.0 -- confirmed live on this host's GPUs (capability 8.9), kernels are used. |
| `moldir` | string | no | `huggingface:boltzgen/inference-data:mols.zip` | — | Path or huggingface:repo:file reference for BoltzGen's canonical molecule/CCD library, needed to resolve residue and ligand chemistry. BoltzGen's own default, already cached on this host. |
| `num_workers` | integer | no | `1` | minimum: `0`<br>maximum: `32` | DataLoader worker process count. 1 is a safe default for a single interactive call; raise it only if data loading, not GPU compute, is the bottleneck. |
