# run_protenix

**Category:** structure_prediction  
**Engine:** `protenix`  
**Environment:** `None`  
**GPU required:** yes

> This file is generated from `src/protein_design_mcp/manifests/run_protenix.yaml`. Edit the manifest, then run `python scripts/generate_tool_docs.py`.

## Summary

Predict a multi-chain protein structure with Protenix v1, given an explicit alignment (or none) per chain. Always pins the v1.0.0 weights explicitly (the proprietary v2 is never selected). Reports the richest confidence output of the co-folding tools here (per-chain and per- chain-pair ptm/iptm/plddt/gpde), and can optionally dump full per-token confidence data for downstream scoring.

## What this is
Protenix v1 all-atom structure prediction. `pip install protenix`
resolves to package version 2.0.0 today, whose own default model_name
already happens to be `protenix_base_default_v1.0.0` -- but this tool
always pins `-n protenix_base_default_v1.0.0` explicitly on every call
rather than relying on that default, because a future package release
could change it and the proprietary, opt-in-only `protenix-v2` must
never be selected silently.

## What it is for
Predicting how a set of protein chains fold together, same job shape as
the other co-folding tools here.

## When to use this instead of the alternatives
- `run_boltz`, `run_chai1`, `run_openfold3` (not yet implemented) are the
  direct siblings -- same job shape, different model. Compare them on the
  SAME `msa` input; do not change engine and alignment source in the
  same comparison.
- This is the only co-folding tool here that reports per-chain-PAIR ptm
  (`chain_pair_iptm` is in the raw confidence file even though this
  tool's own summary surfaces only the scalar `chain_ptm`/`chain_iptm`
  lists) -- inspect `outputs.confidence_json` directly if you need the
  full per-pair matrix.
- `run_ipsae` needs a PAE JSON. This tool's `full_data_json` output
  (written when `need_atom_confidence` is true) VERIFIED live to carry a
  `token_pair_pae` matrix, but under Protenix's own key name, not an
  AF3-style `pae` key -- it is not a drop-in `run_ipsae` input as-is.
  `run_boltz`'s `pae_npz` output is the verified drop-in path to
  `run_ipsae` today.

## What you must supply
`chains`: one entry per chain in the assembly, each
`{"sequence": "<protein AA string>", "msa": "<path>" | null, "copies": <int, default 1>}`.
List every chain you want folded together in this call. `msa` has no
default: pass `null` to run that chain MSA-free, or a path to the
`unpaired_a3m` file `run_mmseqs_search` wrote for that exact sequence.

## Important caveat
Protenix's own `--use_msa` flag is job-wide, not per-chain: if any
protein chain in a job lacks a resolvable alignment path AND `--use_msa`
is true, Protenix calls its OWN remote MSA service
(`https://protenix-server.com/api/msa`, verified from source). To keep
every `msa: null` chain genuinely local and MSA-free while still letting
other chains in the same call use a real alignment, this wrapper writes
a placeholder single-record (query-only) a3m for every `null` chain
before invoking Protenix, so its own search path is never reached
regardless of what any individual chain's `msa` is set to.

## What you get back
`plddt`, `ptm`, `iptm`, `gpde`, `has_clash`, `ranking_score`, `chain_ptm`,
`chain_iptm` (from the lowest seed/sample pair), `num_structures`,
`model_pinned`, and under `outputs` the paths to every predicted
structure and its per-sample confidence file(s).

## Parameters

| Parameter | Type | Required | Default | Constraints | Description |
|---|---|---|---|---|---|
| `chains` | array | yes | `—` | minItems: `1`<br>maxItems: `12` | One entry per chain in the predicted assembly: {"sequence": "<protein amino-acid string, uppercase, standard 20 plus X/B/Z/J/U/O>", "msa": "<path to run_mmseqs_search's unpaired_a3m output for this sequence>" or null, "copies": <positive integer, default 1, identical copies of this chain>}. Chain composition is never inferred -- list exactly the chains you want folded together. "msa" is required on every entry; there is no default. |
| `cycle` | integer | no | `10` | minimum: `1`<br>maximum: `20` | Number of Pairformer recycling cycles. 10 is Protenix's own default. More cycles can improve hard interfaces at roughly linear extra cost. |
| `step` | integer | no | `200` | minimum: `1`<br>maximum: `1000` | Number of diffusion denoising steps. 200 is Protenix's own default. Fewer steps is much faster and noticeably lower quality -- drop this for a fast sanity check, not for a result you intend to keep. |
| `sample` | integer | no | `5` | minimum: `1`<br>maximum: `25` | Number of independent structures to sample per seed. 5 is Protenix's own default; more samples costs roughly linear GPU time. |
| `seeds` | array | no | `[101]` | minItems: `1`<br>maxItems: `5` | Random seeds; one full set of `sample` structures is generated per seed. [101] is Protenix's own single-seed default. Add more seeds to check sensitivity to initialization rather than only to sampling. |
| `dtype` | string | no | `bf16` | enum: `['bf16', 'fp32']` | Inference numeric precision. bf16 (Protenix's own default) is faster and uses less GPU memory with negligible accuracy cost on this hardware; fp32 trades speed for maximum numerical precision. |
| `need_atom_confidence` | boolean | no | `True` | — | Whether to additionally dump full per-token/per-atom confidence data (see the full_data_json output) alongside the summary. True by default so the richer file is available whenever you need it; set false to skip writing it and save a little disk/time on a call where you only need the summary fields. |
