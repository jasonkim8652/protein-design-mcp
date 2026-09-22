# run_openfold3

**Category:** structure_prediction  
**Engine:** `openfold3`  
**Environment:** `None`  
**GPU required:** yes

> This file is generated from `src/protein_design_mcp/manifests/run_openfold3.yaml`. Edit the manifest, then run `python scripts/generate_tool_docs.py`.

## Summary

Predict a multi-chain protein structure with OpenFold3 (OpenBind release), given an explicit alignment (or none) per chain. Apache-2.0 code, weights and training data. num_diffusion_samples is capped at 5 on this hardware -- 100 samples is documented to OOM a 48 GB card. Its per-sample confidence_json carries a directly usable PAE matrix for run_ipsae.

## What this is
OpenFold3 all-atom structure prediction, using the default "OpenBind"
checkpoint (`openbind-2025-06-30-174k`, Apache-2.0, downloaded from an
unsigned/anonymous public S3 bucket -- not a gated weight).

## What it is for
Predicting how a set of protein chains fold together, same job shape as
the other co-folding tools here.

## When to use this instead of the alternatives
- `run_boltz`, `run_chai1`, `run_protenix` (not yet implemented) are the
  direct siblings -- same job shape, different model. Compare them on
  the SAME `msa` input; do not change engine and alignment source in the
  same comparison.
- This tool's `confidence_json` output (one file per sample) VERIFIED
  live to carry a top-level `pae` key with an NxN matrix -- a genuine,
  directly usable `run_ipsae` input, alongside `run_boltz`'s `pae_npz`.

## What you must supply
`chains`: one entry per chain in the assembly, each
`{"sequence": "<protein AA string>", "msa": "<path>" | null, "copies": <int, default 1>}`.
List every chain you want folded together in this call. `msa` has no
default: pass `null` to run that chain MSA-free, or a path to the
`unpaired_a3m` file `run_mmseqs_search` wrote for that exact sequence.

## Important caveat -- the diffusion-sample cap
`num_diffusion_samples` is capped at 5 here. OpenFold3's own issue
tracker (#71) reports a 75.94 GiB allocation attempt on a 48 GB card at
100 samples; this host's L40S has 46 GB. 5 also happens to be the
built-in model-config default (`architecture.shared.diffusion.
no_full_rollout_samples = 5`), so the cap does not restrict the common
case -- it exists to make the ceiling explicit rather than let a caller
discover it as an out-of-memory crash.

## What you get back
`avg_plddt`, `ptm`, `iptm`, `gpde`, `has_clash`, `sample_ranking_score`,
`chain_ptm`, `chain_pair_iptm` (from the lowest seed/sample pair),
`num_structures`, `num_diffusion_samples_cap`, and under `outputs` the
paths to every predicted structure and its per-sample confidence files.

## Parameters

| Parameter | Type | Required | Default | Constraints | Description |
|---|---|---|---|---|---|
| `chains` | array | yes | `—` | minItems: `1`<br>maxItems: `12` | One entry per chain in the predicted assembly: {"sequence": "<protein amino-acid string, uppercase, standard 20 plus X/B/Z/J/U/O>", "msa": "<path to run_mmseqs_search's unpaired_a3m output for this sequence>" or null, "copies": <positive integer, default 1, identical copies of this chain>}. Chain composition is never inferred -- list exactly the chains you want folded together. "msa" is required on every entry; there is no default. |
| `num_diffusion_samples` | integer | no | `5` | minimum: `1`<br>maximum: `5` | Number of independent structures to sample. Capped at 5 on this hardware (see the doc's caveat) -- this is not an arbitrary restriction, it is also OpenFold3's own model-config default. |
| `seeds` | array | no | `[42]` | minItems: `1`<br>maxItems: `5` | Random seeds; one full set of num_diffusion_samples structures is generated per seed. [42] is OpenFold3's own single-seed default. Add more seeds to check sensitivity to initialization rather than only to sampling. |
