# run_rf3

**Category:** structure_prediction  
**Engine:** `rf3`  
**Environment:** `/home/jk661/.conda/envs/foundry`  
**GPU required:** yes

> This file is generated from `src/protein_design_mcp/manifests/run_rf3.yaml`. Edit the manifest, then run `python scripts/generate_tool_docs.py`.

## Summary

Co-fold one or more protein chains together with RoseTTAFold3 and return its structure and AF3-style confidence (ptm, iptm, has_clash, ranking_score). RF3 never fetches or builds its own alignment -- supply one per chain via `msa`, or run every chain MSA-free deliberately with `msa: null`. Shares its `foundry` environment and checkpoint cache with RFdiffusion3 (a different, generative tool, not this one).

## What this is
RoseTTAFold3 (`rf3`, RosettaCommons' `foundry` monorepo, PyPI
`rc-foundry`), an AlphaFold-3-class co-folding model. Given a set of
protein chains it predicts one joint structure and reports AF3-shaped
confidence metrics (`ptm`, `iptm`, per-chain-pair PAE/PDE, `has_clash`,
`ranking_score`).

## `msa` is never automatic -- confirmed from source, not assumed
RF3 does not fetch or build an alignment itself (`rf3/inference_engines/
rf3.py`: MSA directories are set ONLY from the `LOCAL_MSA_DIRS`
environment variable; with it unset, RF3 logs a warning and predicts
single-sequence). This tool never sets `LOCAL_MSA_DIRS` -- instead, the
wrapper script writes each chain's supplied a3m path directly into RF3's
own JSON input as a per-chain `msa_paths` entry (`rf3/utils/
inference.py::InferenceInput.from_json_dict`, read from source), which
RF3 reads without any directory/hashing convention. Confirmed live: RF3
folds a chain fine with no MSA at all (single-sequence smoke test, GPU
7). `msa: null` runs deliberately MSA-free; there is no `"auto"`.

## `chains` -- chain composition is explicit, never inferred
A one-entry list predicts a monomer; several entries co-fold a complex.
Only protein chains are supported by this tool (each is written as RF3's
own `POLYPEPTIDE(L)` component type). RF3 itself also supports nucleic
acids and ligand components via the same JSON schema -- not exposed here,
since that surface was not exercised in this server's install/smoke
testing and this tool does not invent flags it has not verified.

## When to use this instead of the alternatives
- `run_promera` also co-folds, but reports Promera's own `iCS`/`ipSAE`
  confidence rather than AF3-shaped fields, and is a materially different
  model. `run_alphafold2_multimer` co-folds via ColabFold's AF2-Multimer
  weights instead of RF3's own. All three take an explicit `msa`; none
  build one.
- For a single chain with no partners, `run_esmfold2` is faster and takes
  no alignment machinery at all (it has no `msa` parameter, by design).
- Once you have a structure from any of these, `run_ipsae` gives a
  model-agnostic interface confidence from the PAE matrix; RF3's own
  `iptm` here is the same kind of number as a co-folding tool's built-in
  confidence, not a second opinion.

## What you must supply
`chains` and `msa` (every chain's alignment choice stated, `null` or a
path, never omitted).

## What you get back
From RF3's own `<id>_summary_confidences.json` (confirmed live, but
**this schema is unstable upstream** -- it is hand-assembled per-field by
RF3 itself, not a versioned contract, per
`rf3/inference_engines/rf3.py::compile_af3_style_confidence_outputs`):
`ptm`, `iptm`, `has_clash`, `ranking_score`, `overall_plddt`,
`overall_pae`, `overall_pde`. This adapter passes the file's content
through as-is rather than picking individual fields, so it survives most
upstream additions but NOT a rename of a field this tool's own doc
promises -- if a future `rc-foundry` release renames one of the fields
above, expect this doc (not the adapter, which does no field-by-field
parsing) to need an update. Under `outputs`, paths to `structure_cif` and
`summary_confidences_json` (the full per-atom `<id>_confidences.json` and
per-model `<id>_ranking_scores.csv` RF3 also writes are not collected by
this tool).

## Important caveats
- `n_recycles` has a hard floor of 2: `n_recycles=1` was confirmed live to
  crash (`IndexError: pop from an empty deque` /
  `RuntimeError: Recycling generator produced no outputs` --
  RF3's recycling generator needs at least 2 iterations). This is a real
  engine limitation, not a policy choice by this tool.
- RF3's own template conditioning, ligand/nucleic-acid components,
  `early_stopping_plddt_threshold`, and `fallback_conformer_to_input_coords`
  are not exposed here -- all are outside the protein-only, no-template
  surface this tool was verified against.

## Parameters

| Parameter | Type | Required | Default | Constraints | Description |
|---|---|---|---|---|---|
| `chains` | array | yes | `—` | minItems: `1` | List of protein chains to co-fold, each {"chain_id": "<short token, e.g. 'A'>", "sequence": "<AA sequence, standard 20 + XBZJUO>"}. `chain_id` values must be unique within the list. One entry predicts a monomer; several co-fold a complex -- chain composition is never inferred, this list is exactly the assembly RF3 predicts. |
| `msa` | — | yes | `—` | — | null runs every chain in `chains` MSA-free, deliberately (RF3 never fetches or builds an alignment itself). To supply alignments, a JSON object whose keys are EXACTLY the `chain_id` values in `chains` (every chain's choice must be stated) and whose values are each null (that chain runs MSA-free) or a path to that chain's own single plain a3m -- run_mmseqs_search's unpaired_a3m for that chain's sequence is valid here. There is no "auto". |
| `n_recycles` | integer | no | `10` | minimum: `2`<br>maximum: `40` | Number of trunk recycling iterations. Must be at least 2 -- RF3's recycling generator crashes below that (confirmed live, see the doc's "Important caveats"). Higher can improve accuracy at roughly linear cost; 10 is RF3's own shipped default (`RF3InferenceEngine.__init__`). |
| `diffusion_batch_size` | integer | no | `5` | minimum: `1`<br>maximum: `32` | Number of independent structure samples the diffusion head draws in parallel; RF3 returns its own top-ranked sample. 5 is RF3's own shipped default. |
| `num_steps` | integer | no | `50` | minimum: `1`<br>maximum: `200` | Number of diffusion denoising steps. More steps can sharpen geometry at roughly linear cost; 50 is RF3's own shipped default. The install smoke test used 5 for a fast sanity check only -- not a value to use for a real prediction. |
