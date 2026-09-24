# run_chai1

**Category:** structure_prediction  
**Engine:** `chai_lab`  
**Environment:** `/home/jk661/.conda/envs/chai1`  
**GPU required:** yes

> This file is generated from `src/protein_design_mcp/manifests/run_chai1.yaml`. Edit the manifest, then run `python scripts/generate_tool_docs.py`.

## Summary

Predict a multi-chain protein structure with Chai-1, given an explicit alignment (or none) per chain. Uses ESM embeddings by default (no weight download surprise -- already cached on this host). Lowest integration risk of the co-folding tools here (Apache-2.0, L40S is an explicitly supported SKU). Does not produce a PAE matrix on disk -- use its own per_chain_pair_iptm for interface ranking, not run_ipsae.

## What this is
Chai-1 all-atom structure prediction. ESM-2-3B sequence embeddings are
computed and used by default (`use_esm_embeddings`); this is separate
from, and not a substitute for, the MSA you supply through `chains`.

## What it is for
Predicting how a set of protein chains fold together, same job shape as
the other co-folding tools here: `chains` lists exactly who's present.

## When to use this instead of the alternatives
- `run_boltz`, `run_protenix`, `run_openfold3` are the direct siblings -- same job shape, different
  model. Compare them on the SAME `msa` input; do not change engine and
  alignment source in the same comparison.
- **This tool cannot feed `run_ipsae`.** Chai-1's own CLI never writes
  its computed PAE matrix to disk (verified from source: the internal
  folding routine in `chai1.py` keeps `pae_scores` only in the in-memory
  `StructureCandidates` return value, which the CLI entry point never
  saves). Use `per_chain_pair_iptm` from this tool's own output to rank
  interfaces, or predict with `run_boltz`/`run_protenix`/`run_openfold3` when you need a
  PAE-based ipSAE score.
- Templates and explicit contact/pocket constraints are not exposed by
  this tool (Chai-1 supports both, but scoping this tool to the
  sequence+MSA+structure path keeps it consistent with every other
  co-folding tool here, none of which expose templates yet either).

## What you must supply
`chains`: one entry per chain in the assembly, each
`{"sequence": "<protein AA string>", "msa": "<path>" | null, "copies": <int, default 1>}`.
List every chain you want folded together in this call. `msa` has no
default: pass `null` to run that chain MSA-free, or a path to the
`unpaired_a3m` file `run_mmseqs_search` wrote for that exact sequence.
This wrapper converts the supplied a3m into Chai-1's own `.aligned.pqt`
format itself and never lets Chai-1 reach its own MSA server or run a
search, no matter what `msa` is set to.

## What you get back
`aggregate_score`, `ptm`, `iptm`, `per_chain_ptm`, `per_chain_pair_iptm`,
`has_inter_chain_clashes` (from the rank-0 sample), `num_structures`, a
`caveat` restating the no-PAE limitation, and under `outputs` the paths
to every predicted structure and its per-model scores file.

## Parameters

| Parameter | Type | Required | Default | Constraints | Description |
|---|---|---|---|---|---|
| `chains` | array | yes | `—` | minItems: `1`<br>maxItems: `12` | One entry per chain in the predicted assembly: {"sequence": "<protein amino-acid string, uppercase, standard 20 plus X/B/Z/J/U/O>", "msa": "<path to run_mmseqs_search's unpaired_a3m output for this sequence>" or null, "copies": <positive integer, default 1, identical copies of this chain>}. Chain composition is never inferred -- list exactly the chains you want folded together. "msa" is required on every entry; there is no default. |
| `num_trunk_recycles` | integer | no | `3` | minimum: `0`<br>maximum: `10` | Number of recycling passes through the trunk before diffusion. 3 is Chai-1's own default. More recycling can help hard interfaces at roughly linear extra cost. |
| `num_diffn_timesteps` | integer | no | `200` | minimum: `1`<br>maximum: `1000` | Number of diffusion denoising steps. 200 is Chai-1's own default. Fewer steps is much faster and noticeably lower quality -- drop this for a fast sanity check, not for a result you intend to keep. |
| `num_diffn_samples` | integer | no | `5` | minimum: `1`<br>maximum: `25` | Number of independent structures to sample per trunk sample. 5 is Chai-1's own default. Chai-1 ranks and returns all of them (model_idx_0 is the highest aggregate_score). |
| `num_trunk_samples` | integer | no | `1` | minimum: `1`<br>maximum: `5` | Number of independent trunk (MSA/pair representation) samples, each producing its own num_diffn_samples structures. 1 is Chai-1's own default; raising it explores more trunk-level diversity at roughly linear extra cost, and nests output structures one directory level deeper (trunk_<i>/). |
| `use_esm_embeddings` | boolean | no | `True` | — | Whether to compute and use ESM-2-3B sequence embeddings as an extra model input, separate from the MSA. True is Chai-1's own default and the setting its reported benchmarks use; the weights are already cached on this host so there is no first-call download surprise. |
| `low_memory` | boolean | no | `True` | — | Whether to trade some speed for lower peak GPU memory (chunks certain internal computations). True is Chai-1's own default and safe to leave on; set false only if you have GPU memory to spare and want the faster path. |
| `seed` | integer | no | `42` | minimum: `0` | Random seed for trunk and diffusion sampling. Fixed by default so repeated calls with identical inputs are reproducible. |
