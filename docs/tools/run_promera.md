# run_promera

**Category:** structure_prediction  
**Engine:** `promera`  
**Environment:** `/home/jk661/.conda/envs/promera`  
**GPU required:** yes

> This file is generated from `src/protein_design_mcp/manifests/run_promera.yaml`. Edit the manifest, then run `python scripts/generate_tool_docs.py`.

## Summary

Co-fold one or more protein chains together and score the result with Promera's own confidence heads (iCS, ipSAE, pLDDT, pTM/ipTM) in a single pass. Unlike run_ipsae, which scores any predictor's PAE after the fact, this tool both PREDICTS the structure and reports its own confidence in that specific prediction -- the same relationship run_chai1 (not yet implemented)'s own ipTM has to its own structure. Promera's `Design` task (minibinder/nanobody generation) is a separate, composite pipeline and is not exposed by this tool.

## What this is
Promera (github.com/bjing2016/promera, MIT licence, Bowen Jing), a
biomolecular co-folding model built on the `tinyprot` structure/feature
library. Given a set of protein chains it predicts one joint structure and
reports its own confidence in that structure: per-chain pLDDT/pTM,
complex-level pTM/ipTM, and -- the metrics this tool exists for --
per-chain-pair `iCS` (interface contact score) and `ipSAE`.

## `run_promera`'s confidence output is NOT the same thing as `run_ipsae`
`run_ipsae` reads the PAE matrix ANY structure predictor already produced
and scores the interface in it after the fact -- it is model-agnostic by
design. This tool's `ipsae`/`iCS` numbers are Promera's own built-in
confidence in the structure IT just predicted, computed inside the same
forward pass, the same relationship a co-folding tool's own `iptm` field
has to its own prediction. Do not read this tool's confidence output as a
second opinion on someone else's structure, and do not feed this tool's
structure through `run_ipsae` expecting a materially different number for
the same interface -- if you want an independent check, use
`run_rosetta_interface` (not yet implemented) or `run_prodigy` instead,
which score the physics, not the model's own confidence.

## When to use this instead of the alternatives
- Use this when you want ONE co-folded structure of several chains AND its
  own interface confidence in one call. `run_rf3` and
  `run_alphafold2_multimer` co-fold too but report confidence in their own
  native shape (RF3's AF3-style `iptm`/`ptm`; ColabFold's `ipTM`/`pTM`),
  not `iCS`/`ipSAE` directly.
- `run_esmfold2` (not this tool) is the right choice for a single chain,
  alone, with no alignment support at all.

## `chains` -- chain composition is explicit, never inferred
A one-entry map predicts (and scores) a monomer; a multi-entry map
co-folds a complex. This is Promera's own target-schema format
(`tinyprot.structure.Structure.from_schema`), written directly to the
JSON file Promera reads -- not reinterpreted by this tool. `entity_id`
marks chains that are identical copies of one molecule (homo-oligomers);
give distinct chains distinct integers.

## `msa` -- never automatic, and keyed to EXACTLY the chains you gave
`null` runs every chain in `chains` MSA-free, deliberately (Promera's own
single-sequence dummy MSA fallback -- confirmed live: a 2-chain complex
folded fine with no MSA at all, `ipsae` populated at ~0 as expected for a
low-quality MSA-free smoke prediction). To supply real alignments, pass a
JSON object whose keys are EXACTLY the keys of `chains` (every chain's
choice must be stated, one way or the other -- this tool rejects a
partial map rather than silently running the missing chains MSA-free) and
whose values are each either `null` (that one chain, specifically, runs
MSA-free) or a path to that chain's own single plain a3m --
`run_mmseqs_search`'s `unpaired_a3m` for that exact sequence is valid
here. Promera looks its MSA cache up by a SHA-256 hash of the exact
sequence string in `chains` (`tinyprot.msa.hash_sequence` /
`load_msa_from_dir`, read from source, not guessed) -- an a3m built for a
different sequence than the one in `chains` is silently treated as "no
MSA found" rather than raising, so double check the sequence matches
exactly.

## What you must supply
`chains` and `msa` (every chain's alignment choice stated).

## What you get back
Promera's own `_conf.json` content for this target: `complex_plddt`,
`complex_ptm`, `chain_plddt`, `ptm` (per chain), `complex_iptm`, `iptm`
(per chain pair, complexes only), `ipsae` (per chain pair), `iCS`
(interface contact score per chain pair -- empty for a single-chain
target), `msa_depth` and `msa_path` per chain. Under `outputs`, paths to
`structure_cif` and `confidence_json`.

## Important caveats
- `Design` (minibinder/VHH nanobody generation via a LigandMPNN fork) is
  Promera's own composite task and is not installed or wired by this
  tool, per this server's policy of not exposing orchestrators over steps
  already available individually.
- Promera's own performance/precision knobs (`amp`, `amp_diffusion`,
  `pad_tokens_to_multiple`, `dynamic_schedule`, `sort_by_size`) and
  output-bookkeeping knobs (`skip_existing`, `save_traj`,
  `save_distogram`, `save_full_confidence`) are not exposed here -- none
  change the predicted structure or its confidence, only how the job is
  scheduled/what side files it additionally writes.
- The editable install this manifest mounts lives under this session's
  scratchpad -- see the `engine.mounts` comment in this manifest for the
  durability caveat before relying on this past the current session.

## Parameters

| Parameter | Type | Required | Default | Constraints | Description |
|---|---|---|---|---|---|
| `chains` | object | yes | `—` | — | Promera's own per-target schema map: each key is a chain label you choose (e.g. "A1", "B1") and each value is {"type": "protein", "sequence": "<AA sequence, standard 20 + XBZJUO>", "entity_id": <int>}. entity_id marks identical copies of one molecule (homo-oligomers) with the same integer; give distinct chains distinct integers. A one-entry map predicts a monomer; several entries co-fold a complex -- chain composition is never inferred, this map is exactly the assembly Promera predicts and scores. Written directly as Promera's own target JSON schema file (tinyprot Structure.from_schema format). |
| `msa` | — | yes | `—` | — | null runs every chain in `chains` MSA-free, deliberately. To supply alignments, a JSON object whose keys are EXACTLY the keys of `chains` (every chain's choice must be stated) and whose values are each null (that chain runs MSA-free) or a path to that chain's own single plain a3m (e.g. run_mmseqs_search's unpaired_a3m for that exact sequence -- Promera looks its cache up by a hash of the sequence text, so an a3m for the wrong sequence is silently treated as "no MSA" rather than an error). There is no "auto": this tool never fetches or builds an alignment itself. |
| `recycling_steps` | integer | no | `4` | minimum: `1`<br>maximum: `20` | Number of trunk recycling iterations. Higher can improve accuracy at roughly linear cost. 4 is Promera's own shipped default (`promera/inference/cofolding.yaml`). |
| `diffusion_samples` | integer | no | `5` | minimum: `1`<br>maximum: `32` | Number of independent structure samples the diffusion head draws; the returned structure/confidence is Promera's own top-ranked sample. 5 is Promera's own shipped default. |
| `diffusion_steps` | integer | no | `200` | minimum: `1`<br>maximum: `400` | Number of denoising steps per diffusion sample. More steps can sharpen geometry at roughly linear cost; 200 is Promera's own shipped default. The install smoke test used 5 for a fast sanity check only -- not a value to use for a real prediction. |
| `num_seeds` | integer | no | `1` | minimum: `1`<br>maximum: `8` | How many independent random seeds to run this target under (each producing its own full set of `diffusion_samples`). 1 is Promera's own shipped default; raise it to check how much the prediction varies across seeds, at proportional cost. |
