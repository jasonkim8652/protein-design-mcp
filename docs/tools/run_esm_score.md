# run_esm_score

**Category:** scoring  
**Engine:** `esm`  
**Environment:** `/home/jk661/.conda/envs/esm_env`  
**GPU required:** yes

> This file is generated from `src/protein_design_mcp/manifests/run_esm_score.yaml`. Edit the manifest, then run `python scripts/generate_tool_docs.py`.

## Summary

Score how "natural" a protein sequence looks to ESM2-650M, as a masked- marginal pseudo-log-likelihood. This is a developability proxy -- it reflects similarity to the natural sequences the language model was trained on -- and NOT a binding predictor: it says nothing about whether a binder actually contacts or binds its target. It scores a sequence you already have (e.g. a design's own generated sequence, or one run_mpnn produced for a fixed backbone); it does not fold or design anything.

## What this is
ESM2-650M (`esm2_t33_650M_UR50D`, the same weights Proteina-Complexa's own
composite "evaluate" step used to load via `AutoModelForMaskedLM` before
that step was excluded from this server as a composite -- see
docs/superpowers/specs/2026-09-21-atomistic-tool-refresh-design.md) scored
as a masked-marginal pseudo-log-likelihood (PLL): for each position in the
sequence, that position's token is replaced with `<mask>` and the model's
predicted log-probability of the TRUE residue at that position, under that
masked context, is recorded. The sequence's PLL is the mean of these
per-position values. This is the field-standard definition of
"pseudo-likelihood" for a masked language model (Meier et al. 2021's
"wildtype marginal" is a cheaper one-forward-pass approximation of the
same idea; this tool runs the full masked-marginal version, not that
approximation, so its cost scales with sequence length).

## What this is NOT -- read this before using it for ranking binders
This tool answers "does this sequence look like a protein ESM2 has seen
the statistical shape of before", not "does this sequence bind its
target" or "will this sequence fold and express". A high PLL sequence can
still fail to bind, fail to fold, or fail to express; a lower-PLL sequence
(further from natural sequence statistics -- common for a de novo
binder's designed interface) is not automatically a bad design. Never use
this tool's output as the sole or primary ranking criterion for binder
designs -- use `run_ipsae` (predictor confidence in an interface) or
`run_prodigy`/`run_rosetta_interface` (physical interface quality) for
that. Use this tool as one input among several, or as a coarse sanity
check on a redesigned sequence's overall "naturalness" (e.g. after
`run_mpnn`).

## When to use this instead of the alternatives
- `run_ipsae` and a co-folding tool's own `iptm` measure predictor
  CONFIDENCE in a specific interface -- a different question from this
  tool's sequence-only "naturalness" score, and the better choice for
  ranking binder candidates against each other.
- `run_prodigy` and `run_rosetta_interface` score the PHYSICS of an
  interface from a structure -- also a different question. This tool
  needs no structure at all, only a sequence, so it is the cheapest and
  fastest of the three to run.

## What you must supply
`sequence`: a single protein chain, uppercase, the standard 20 amino acids
only. Ambiguity codes (X/B/Z/J/U/O) are rejected -- a masked-marginal
score at a position whose "true" residue is not one of the 20 canonical
amino acids is not meaningful (there is no well-defined "log-probability
of X" to report), so this tool refuses the input outright rather than
return a silently degraded number for those positions.

## What you get back
`pseudo_log_likelihood` (the sequence-level PLL, this tool's headline
number -- higher/less negative means more "natural" under ESM2-650M),
`per_residue_log_likelihood` (one value per position, same order as
`sequence`, useful for spotting exactly which residues look unusual),
`sequence_length`, `device` (`cuda` or `cpu`, whichever this run actually
used), and a `caveat` string restating the "developability proxy, not a
binding predictor" warning above.

## Parameters

| Parameter | Type | Required | Default | Constraints | Description |
|---|---|---|---|---|---|
| `sequence` | string | yes | `—` | pattern: `^[ACDEFGHIKLMNPQRSTVWY]{1,2000}$` | Protein sequence to score, uppercase, standard 20 amino acids only (no X/B/Z/J/U/O ambiguity codes -- see the doc for why). 2000-residue cap keeps a single call's masked-marginal cost (one forward pass per position, batched -- see batch_size) bounded; split a longer construct into domains if you need to score more than that. |
| `batch_size` | integer | no | `32` | minimum: `1`<br>maximum: `256` | How many masked positions to score per forward pass. Purely a speed/GPU-memory knob -- it does not change the score, since each position is still scored independently with only that ONE position masked. Higher is faster but uses more GPU memory; lower this if a long sequence runs out of memory. 32 is a middle-ground default for a 650M-parameter model on a single modern GPU. |
