# run_esmfold2

**Category:** structure_prediction  
**Engine:** `esm`  
**Environment:** `/home/jk661/.conda/envs/esmfold2`  
**GPU required:** yes

> This file is generated from `src/protein_design_mcp/manifests/run_esmfold2.yaml`. Edit the manifest, then run `python scripts/generate_tool_docs.py`.

## Summary

Predict a single protein chain's structure from its sequence alone (ESMFold2, esm==3.4.0's EvolutionaryScale SDK model). No alignment is built or accepted -- this model takes none, unlike every other tool in this category. Fastest single-sequence predictor available here; use it for a quick look or when no MSA is available at all. For a multi-chain complex or when you already have (or can build) an alignment, use run_promera, run_rf3, or run_alphafold2_multimer instead.

## What this is
ESMFold2, EvolutionaryScale's language-model-conditioned structure
predictor (`esm==3.4.0`, `esm.models.esmfold2.EsmFold2Model`), loaded from
the `biohub/ESMFold2` HuggingFace checkpoint. It folds one protein chain
directly from its sequence: an ESM-C protein language model backbone
feeds a diffusion-based structure head, with no multiple-sequence
alignment anywhere in the pipeline.

## No `msa` parameter -- on purpose, not an oversight
ESMFold2 takes no alignment at all. Read from source
(`esm/models/esmfold2/protein_utils.py::prepare_protein_features` and
`hf_adapter.py::infer_protein`): the featurizer builds a depth-1
"pseudo-MSA" from the single input sequence only -- there is no code path
in this SDK that accepts a real, multi-row alignment as an input to
folding. Every other `structure_prediction` tool in this server takes an
explicit `msa` (`null` for deliberately MSA-free, or a path); this tool
has no such parameter because there is nothing for it to control.

## What it is for
The fastest structure prediction available in this server (cold start
~19s, then ~2s/prediction for a 65-residue chain -- confirmed live on GPU
7) with no database or alignment dependency whatsoever. Use it as a quick
first look at a designed monomer, or whenever an MSA genuinely cannot be
built (an orphan sequence, or a caller that wants to test alignment-free
robustness deliberately).

## When to use this instead of the alternatives
- This tool folds exactly one chain, alone. For a multi-chain complex
  (e.g. scoring a designed binder against its target), use
  `run_promera`, `run_rf3`, or `run_alphafold2_multimer` -- all of which
  co-fold multiple chains and take an explicit alignment.
- If you already have, or can build with `run_mmseqs_search` (not yet
  implemented), an alignment for this sequence, a co-folding tool fed
  that alignment will generally be more accurate than this MSA-free
  prediction. Reach for ESMFold2 for speed, not for the best possible
  single-chain structure.

## What you must supply
`sequence` -- one protein chain's amino acid sequence, nothing else.

## What you get back
`mean_plddt` (0-100 scale, AlphaFold's familiar convention -- confirmed
live on GPU 7, not assumed: the model's own internal confidence head
computes pLDDT on a 0-1 scale (`_categorical_mean(..., start=0.0,
end=1.0)` in `esm/models/esmfold2/model.py`), but the SDK's own
`ProteinChain.to_pdb_string()` rescales it to 0-100 when writing the
PDB's B-factor column, which is what this tool actually reads back to
compute `mean_plddt` -- a live run's B-factors were observed in the
38-46 range, not 0-1), `num_residues`, `sequence_length`, and under
`outputs` the path to `structure_pdb`, whose B-factor column carries the
same per-residue pLDDT.

## Important caveats
- `num_diffusion_samples` batches several structure candidates
  internally, but this tool's PDB output is always the FIRST sample
  (confirmed by reading `esm/models/esmfold2/protein_utils.py::
  output_to_pdb`: `coords[:, 0]` then `[0]`) -- raising it does not give
  you more structures back through this tool, only more internal compute
  per call. Exposed anyway because it is a genuine, documented model
  config field (`EsmFold2Config.num_diffusion_samples`), not something
  invented for this wrapper.
- ESMFold2's `forward()` also accepts lower-level diffusion-sampler
  overrides (`noise_scale`, `step_scale`, `max_inference_sigma`,
  `lm_mask_pct`, `msa_max_depth`, `msa_column_mask_rate`). Unlike
  `num_loops`/`num_diffusion_samples`/`num_sampling_steps`, none of these
  carry a docstring or a documented default in `EsmFold2Config` -- there
  is nothing here to write an honest, non-`type-is-not-a-description`
  description for, so they are not exposed. Noted here rather than
  silently dropped.
- The wrapper script verifies at runtime that the loaded `esm` package
  actually lives inside this engine's own environment (not
  `~/.local`) and fails loudly if not, rather than silently running the
  wrong model -- see the `env.PYTHONNOUSERSITE` comment in this manifest
  for why that check exists at all.

## Parameters

| Parameter | Type | Required | Default | Constraints | Description |
|---|---|---|---|---|---|
| `sequence` | string | yes | `—` | pattern: `^[ACDEFGHIKLMNPQRSTVWYXBZJUO]+$` | One protein chain's amino acid sequence, uppercase single-letter codes only (the standard 20 plus the ambiguity codes X/B/Z/J/U/O). This is the ONLY input ESMFold2 takes -- no structure, no alignment, no chain partners. |
| `num_recycles` | integer | no | `20` | minimum: `0`<br>maximum: `64` | Number of trunk refinement loops before the structure head runs (the model's own config field is named `num_loops`; this tool keeps the more familiar `num_recycles` name other predictors here use for the same idea). Each loop re-runs the pairformer trunk on its own previous output, roughly linear cost per loop. 20 is `EsmFold2Config.num_loops`'s own default. 0 skips refinement entirely (a legitimate, fast, lower-quality boundary -- not an error). |
| `num_diffusion_samples` | integer | no | `8` | minimum: `1`<br>maximum: `32` | How many structure candidates the diffusion head samples internally per call (`EsmFold2Config.num_diffusion_samples`'s own default is 8). Only the first sample is ever written to this tool's output PDB (see the doc's "Important caveats") -- raising this adds compute without changing what you get back through this tool. |
| `num_sampling_steps` | integer | no | `68` | minimum: `1`<br>maximum: `200` | Number of denoising steps the diffusion head runs. More steps can sharpen the predicted geometry at roughly linear cost; 68 is `EsmFold2StructureHeadConfig.inference_num_steps`'s own default, read from the model's shipped config. |
