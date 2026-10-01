# run_boltzgen_fold

**Category:** structure_prediction  
**Engine:** `boltzgen`  
**Environment:** `/opt/conda/envs/boltzgen`  
**GPU required:** yes

> This file is generated from `src/protein_design_mcp/manifests/run_boltzgen_fold.yaml`. Edit the manifest, then run `python scripts/generate_tool_docs.py`.

## Summary

Refold designed proteins with BoltzGen's MSA-free confidence model, either with the target (with_target=true) or alone (false). Accepts native design_spec + complete paired CIF/NPZ generated_files, or an external PDB/CIF backbone (including gzip) + explicit design_chains + designed_sequences mapping. External inputs are converted by BoltzGen's native parser and writer without inference; the same folding model then predicts coordinates and confidence. Returns native generated_designs and design_spec_yaml for analyze, refolded structures and sample confidence metrics.

## What this is
BoltzGen's `folding` pipeline step
(`boltzgen.task.predict.predict.Predict` over `fold.yaml`, writer
`FoldingWriter`), run in isolation via `boltzgen run <design_spec>
--steps folding`. Refolds each design WITH its target chain(s) present
(unlike `run_boltzgen_fold` with `with_target: false`, which refolds the design ALONE) using
BoltzGen's own confidence model -- the same architecture family as
Boltz-2, but this is BoltzGen's own weights, not a call to `run_boltz`.
Reports the interface confidence metrics that make this the step
everything downstream (`run_boltzgen_analyze`, `run_boltzgen_filter`)
actually ranks on.

## `--protocol` has no effect on this step for the six protocols that
matter here (verified empirically the same way as `run_boltzgen_design`:
diffing `boltzgen configure --steps folding`'s resolved `fold.yaml` shows
every protocol identical except `protein-redesign`, which sets
`data.design_mask_templates=true` -- a redesign-specific mode this tool
does not support). `protocol` is not exposed here for the same reason it
is not exposed on `run_boltzgen_design`: a parameter with no effect on
the path this tool actually takes would be misleading to document as a
real choice.

## When to use this instead of the alternatives
- `run_boltzgen_fold` with `with_target: false` refolds the design ALONE, target absent -- a
  self-consistency check on the design's own shape, not an interface
  confidence estimate. Run both if you want each design's `analyze` step
  (`run_boltzgen_analyze`) to include `designfolding-*` columns.
- `run_chai1`/`run_boltz` are general-purpose
  structure predictors that take an MSA; this tool is BoltzGen-specific
  and MSA-free, and its output feeds `run_boltzgen_analyze` in the exact
  shape that step expects -- do not substitute a different predictor's
  output here, `run_boltzgen_analyze` reads BoltzGen's own `.npz`
  metadata format, not a generic confidence JSON.

## What you must supply
Choose exactly one input mode and state with_target explicitly:
- Native: design_spec plus generated_files containing one or more complete
  matching .cif/.npz pairs. A selected subset is valid; pass both files for
  every selected design. Folder paths and unmatched files are rejected.
- External: structure (PDB/CIF, optionally gzip), design_chains (actual
  chain IDs), and designed_sequences (map each design chain to its sequence).
  For example, an RFdiffusion3 complex and ProteinMPNN sequence use
  structure: complex.cif.gz, design_chains: [A], designed_sequences: {A: ACDE...}.
  Sequences follow observed residue order and must have exactly the same
  length. Canonical protein chains only. Design residues need complete
  N/CA/C/O backbones; rebuild Genie3 CA traces before this handoff.

External conversion preserves the observed backbone, changes only the named
chains' sequences, and removes their old sidechains. BoltzGen's native YAML
parser, featurizer and DesignWriter derive and serialize genuine metadata;
no sequence generation, structure prediction or invented confidence values
are involved in conversion. Missing sidechain coordinates stay unresolved.
BoltzGen assigns native output chain IDs, which can differ from the input
IDs. outputs.chain_mapping_json records input_to_generated and input_to_refolded
mappings; binder-alone refolds include only design chains. Use these mappings
when selecting output chains. The exported design_spec uses generated chain IDs.
The normal folding step then runs with the caller's checkpoint and settings.

## Downstream analysis
Pass outputs.design_spec_yaml and selected complete outputs.generated_designs
pairs to run_boltzgen_analyze, together with this call's corresponding
refolded_structures and refold_metrics. External converted designs retain
the same ID across all outputs. The exported specification contains concrete
sequences for CLI validation; the CIF/NPZ pair carries the actual design mask.

## What you get back
`refolds`: one entry per design, `{"id", "design_ptm", "design_iptm",
"design_to_target_iptm", "design_iiptm", "design_residue_iptm",
"min_interaction_pae", "min_design_to_target_pae", "interaction_pae",
"iptm", "ptm", "protein_iptm", "target_ptm", "ligand_iptm",
"complex_plddt", "complex_iplddt", "complex_pde", "complex_ipde",
"num_samples", "best_sample_index"}` -- the metrics of the
highest-confidence internal sample (0.8*iptm + 0.2*ptm, matching which
sample `refolded_structures` wrote). `num_refolds`, and under `outputs`
the paths to every refolded structure and the full per-sample metrics.
PAE here is in Angstroms, lower is better; every `*iptm`/`*ptm`/`*plddt`
field is 0-1, higher is better.

## Parameters

| Parameter | Type | Required | Default | Constraints | Description |
|---|---|---|---|---|---|
| `with_target` | boolean | yes | `—` | — | Whether the target chain(s) are PRESENT while the design is refolded. Two different experiments, and the one this server refuses to pick for you -- docs/TOOLS.md's own rule is that whether a prediction runs with the target present or on the binder alone is the caller's decision. true  -- refold the design IN COMPLEX with its target. An estimate of          the INTERFACE: does the model place this binder on this target. false -- refold the design ALONE, target removed. A self-consistency          check on the design's own shape: does the sequence fold back          into what the generator drew, independent of any binding.  Run both when you want run_boltzgen_analyze to carry both the `folding-*` and the `designfolding-*` columns; run_boltzgen_filter can then rank on either. No default: defaulting would make the more consequential of the two the silent one. |
| `design_spec` | string | no | `—` | pattern: `\.(yaml\|yml)$` | The design specification YAML the designs came from. Its content has no effect on this step's own numbers -- required only because `boltzgen run` validates every design spec it is given before running any step, `--steps folding` included. Use outputs.design_spec_yaml from run_boltzgen_design or run_boltzgen_inverse_fold. The tools create this file internally; the caller does not need to write YAML for this handoff. |
| `generated_files` | array | no | `—` | minItems: `2` | Native mode: one or more complete matching .cif/.npz pairs from outputs.generated_designs or outputs.inverse_folded_designs. A selected subset is valid. Each design must include both files, exactly once. Supply design_spec; do not combine with structure or designed_sequences. |
| `structure` | string | no | `—` | pattern: `\.(pdb\|cif)(\.gz)?$` | External protein complex backbone, such as RFdiffusion3 or rebuilt Genie3 output. Alternative to design_spec + generated_files. Supply design_chains and designed_sequences. Design residues require N, CA, C and O coordinates. |
| `design_chains` | array | no | `—` | minItems: `1` | Explicit structure chain IDs carrying the designed sequence; required with structure. |
| `designed_sequences` | object | no | `—` | — | Map every design_chains ID to its ProteinMPNN or other designed sequence, in structure residue order. Exactly one canonical uppercase amino acid per observed protein residue. Other chains keep their structure sequences. |
| `folding_checkpoint` | string | no | `huggingface:boltzgen/boltzgen-1:boltz2_conf_final.ckpt` | — | Path or huggingface:repo:file reference for the folding (confidence model) checkpoint. BoltzGen's own default, already cached on this host (~2.0G). |
| `recycling_steps` | integer | no | `3` | minimum: `0`<br>maximum: `10` | Number of recycling passes through the trunk before diffusion. More recycling can improve hard interfaces at roughly linear extra cost; 3 is BoltzGen's own default for this step. |
| `sampling_steps` | integer | no | `200` | minimum: `1`<br>maximum: `1000` | Number of diffusion denoising steps per refold sample. Fewer is faster and lower quality; 200 is BoltzGen's own default for this step (verified live: a full 200-step refold of a 179-token complex took 52.6s on this host's GPU 7, well within this tool's timeout). |
| `diffusion_samples` | integer | no | `5` | minimum: `1`<br>maximum: `25` | Number of independent refold samples per design. This tool reports the highest-confidence one as its headline metrics (see "What you get back"), but `refold_metrics` carries the full spread for all of them. 5 is BoltzGen's own default for this step. |
| `use_kernels` | string | no | `auto` | enum: `['auto', 'true', 'false']` | Whether to use BoltzGen's fused CUDA kernels. "auto" (BoltzGen's own default) enables them when the GPU's compute capability is >= 8.0 -- confirmed live on this host's GPUs (capability 8.9), kernels are used. |
| `moldir` | string | no | `huggingface:boltzgen/inference-data:mols.zip` | — | Path or huggingface:repo:file reference for BoltzGen's canonical molecule/CCD library, needed to resolve residue and ligand chemistry. BoltzGen's own default, already cached on this host. |
| `num_workers` | integer | no | `0` | minimum: `0`<br>maximum: `0` | DataLoader worker process count. Defaults to 0 to load in the prediction process. Only 0 is supported to prevent multiprocessing queue stalls when container shared memory fills during tensor transfer; use num_workers=0. |
