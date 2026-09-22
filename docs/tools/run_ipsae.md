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
The PAE JSON a predictor emitted, and the structure it goes with.

## What you get back
`ipsae`, `iptm_af`, `pdockq` and the `chain_pair` they describe.

## Parameters

| Parameter | Type | Required | Default | Constraints | Description |
|---|---|---|---|---|---|
| `pae_json` | string | yes | `—` | pattern: `\.json$` | PAE matrix JSON produced by a structure predictor. |
| `structure` | string | yes | `—` | pattern: `\.(pdb\|cif)$` | The predicted structure the PAE matrix belongs to. |
| `pae_cutoff` | number | no | `10.0` | minimum: `1.0`<br>maximum: `30.0` | PAE cutoff in Angstroms. |
| `dist_cutoff` | number | no | `10.0` | minimum: `1.0`<br>maximum: `30.0` | Distance cutoff in Angstroms. |
