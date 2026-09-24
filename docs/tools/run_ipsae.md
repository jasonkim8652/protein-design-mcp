# run_ipsae

**Category:** scoring  
**Engine:** `ipsae`  
**Environment:** `scoring`  
**GPU required:** no

> This file is generated from `src/protein_design_mcp/manifests/run_ipsae.yaml`. Edit the manifest, then run `python scripts/generate_tool_docs.py`.

## Summary

Score how confident a structure predictor was about a protein-protein interface, from the PAE matrix it already produced (ipSAE). This is the field's standard discriminator between real and spurious binders and is the metric to rank designs by. It scores a prediction you already have: run a co-folding tool first and feed this its PAE output.

## What this is
ipSAE (interface Score from Aligned Errors) reads a structure predictor's
predicted aligned error matrix and returns a per-chain-pair confidence in the
interface itself, rather than in the fold as a whole.

## What it is for
Ranking designed binders. Benchmarking across AF2-IG, Boltz-1, Boltz-2,
Chai-1, ColabFold and Protenix found ipSAE roughly 1.4x more precise than
ipTM at separating true binders from non-binders, and it remains the field
standard as of late 2026.

## When to use this instead of the alternatives
- `run_prodigy` returns an absolute binding free energy on a physical scale,
  but is calibrated on natural complexes and mis-ranks de novo designs. Use
  PRODIGY for a sanity floor, ipSAE for ranking.
- The raw `iptm` a co-folding tool reports is the older, less precise form of
  the same idea. Prefer ipSAE when you have the PAE matrix.
- ipSAE tells you nothing about the physics of the interface. For buried
  surface area, hydrogen bonds and shape complementarity you need a
  physics-based interface analysis, which is not yet implemented here.

## What you must supply
The PAE JSON a predictor emitted, and the structure it goes with. The
structure must contain AT LEAST TWO CHAINS -- ipSAE scores the interface
between a pair, so a single-chain structure is refused before the engine
runs. A binder generated without a chain break comes back fused to its
target as one chain and lands here; fold the binder and target as separate
chains instead.

## What you get back
`ipsae`, `iptm_af`, `pdockq` and the `chain_pair` they describe, and under
`outputs` the paths to `results_txt` — ipSAE's own results table plus its
by-residue detail file, both collected as a list. ipSAE emits one row per
chain pair; for a complex with more than two chains, only the first chain-pair row
is returned, with no indication of which pair that is beyond `chain_pair`
itself. Score the specific pair you care about by supplying a
structure/PAE containing only those two chains.

## Parameters

| Parameter | Type | Required | Default | Constraints | Description |
|---|---|---|---|---|---|
| `pae_file` | string | yes | `—` | pattern: `\.(json\|npz)$` | The predicted-aligned-error matrix from the structure predictor that produced `structure`. WHICH FILE depends on the predictor, and they are not interchangeable formats. Three of this server's predictors are known to work, each with a DIFFERENT output, all confirmed by running them:    run_alphafold2_multimer  ->  its `pae_json` output        (.json)   run_boltz                ->  its `pae_npz` output         (.npz)   run_alphafold3           ->  its `confidences_json` output (.json)  For AlphaFold 3 it is `confidences_json`, NOT `summary_confidences_json` -- the latter holds scalars, not the matrix. run_protenix and run_chai1 DO produce a PAE, and ipSAE still cannot read theirs: it parses any .cif + .json pair as AlphaFold 3 format and looks for the keys `pae` and `atom_plddts`, while Protenix writes `pae`/`plddt` and Chai-1 writes `token_pair_pae`/`atom_plddt`. Passing either gets a KeyError deep in the engine rather than a clear refusal, so fold with one of the three above when you intend to score here. This parameter used to be called `pae_json` and reject anything but `.json`, which refused Boltz's PAE even though run_boltz's own output says it is "for run_ipsae". A `.npz` must stay BESIDE its confidence file. ipSAE does not take the summary as an argument -- it derives the path by replacing "pae" with "confidence" and ".npz" with ".json" in this one -- so a PAE copied away from its sibling fails on a file you never named. Pass the path run_boltz returned rather than a copy. `structure` must contain AT LEAST TWO CHAINS. ipSAE scores the interface between a chain pair, so a monomer gives it nothing to score -- and it does not say so: on a single-chain prediction it dies with `cannot access local variable 'n0res_byres_all'`, which names nothing a caller can act on. Fold with the target present before scoring. |
| `structure` | string | yes | `—` | pattern: `\.(pdb\|cif)$` | The predicted structure the PAE matrix belongs to. WHERE THIS COMES FROM -- The structure whose PAE you are passing, from the same predictor run -- `run_alphafold2_multimer`, `run_boltz` (its structures output) or `run_alphafold3` (its model_cif output). |
| `pae_cutoff` | number | no | `10.0` | minimum: `1.0`<br>maximum: `30.0` | PAE cutoff in Angstroms. |
| `dist_cutoff` | number | no | `10.0` | minimum: `1.0`<br>maximum: `30.0` | Distance cutoff in Angstroms. |
