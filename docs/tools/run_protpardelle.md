# run_protpardelle

**Category:** binder_generation  
**Engine:** `protpardelle`  
**Environment:** `/home/jk661/.conda/envs/pp1c`  
**GPU required:** yes

> This file is generated from `src/protein_design_mcp/manifests/run_protpardelle.yaml`. Edit the manifest, then run `python scripts/generate_tool_docs.py`.

## Summary

Generate a binder against a target with Protpardelle-1c's all-atom diffusion model, using an RFdiffusion-style contig string to fix the target chain(s) and specify the generated chain's length (backbone by default; side chains too with an all-atom model variant). MSA-free by construction -- there is no alignment parameter because this is a structure diffusion model, not a sequence model.

## What this is
Protpardelle-1c's `python -m protpardelle.sample` multi-chain
conditional-generation path, run in single-target mode: a contig string
fixes one or more target chains (taken from `target_pdb`) and specifies a
length range for one new chain, and the model samples that new chain's
backbone (or, with an all-atom `model`, its side chains too) conditioned
on the target.

## `contig` and `total_lengths` -- the parameters that decide whether this tool is usable at all
`contig` follows Protpardelle-1c's own contig grammar (similar to
RFdiffusion's, but `;/;` denotes a chain break instead of a bare space) --
read from Protpardelle-1c's own README and `data/motif.py`, and validated
by this schema before it ever reaches the engine:
- `A1-128;/;120-120`: condition on `target_pdb` chain A residues 1-128,
  generate a new chain of EXACTLY 120 residues.
- `A1-79;/;B1-141;/;70-150`: condition on chains A and B, generate a new
  chain between 70 and 150 residues.
- A bare `<min>-<max>` segment (no chain letter) is a flexible-length
  scaffold segment inserted around fixed motif segments, e.g.
  `0-20;A1-50;0-20;/;100-100` conditions on a 50-residue span from chain
  A with up to 20 flexible residues on either side, then generates a
  100-residue chain.

`total_lengths` is a list of `[min, max]` pairs, ONE PER CHAIN IN THE SAME
ORDER `contig`'s chain-break-separated (`;/;`) segments introduce them --
for every TARGET chain, `[min, max]` should equal its exact residue count
from `target_pdb` (Protpardelle-1c's own README: "the total lengths for
the target chain(s) should match their number of residues"); the final
entry is the generated chain's own length range.

## `hotspots`
A list of `"{chain_id}{residue_index}"` tags (e.g. `"A19"`) naming
residues on the TARGET to bias generation toward -- Protpardelle-1c's own
binder-generation convention, passed straight through as a comma-joined
string. `null` runs without hotspot bias (still target-conditioned via
`contig`, just with no additional steering toward specific residues).

## `model` -- which trained checkpoint, not a decoration
Every one of these is the ONLY checkpoint for its epoch on this host (see
Protpardelle-1c's own README table):
- `cc83` (default): backbone-only, multi-chain conditional generation --
  "the BindCraft benchmark model" in Protpardelle-1c's own words, and the
  one used in its own binder-generation example.
- `cc95`: the same architecture as cc83, finetuned with heavier hotspot
  dropout -- Protpardelle-1c's README recommends it when you want the
  model to rely less heavily on the exact hotspots given.
- `cc94`: all-atom (produces side chains, not just backbone), finetuned on
  multichain data, but WITHOUT hotspot conditioning -- `hotspots` has no
  effect with this model selected (Protpardelle-1c's own README:
  "multichain data but no hotspots").
- `cc78`: an experimental model whose residue indices are tied across
  chains -- Protpardelle-1c's own README says this "favors homodimers"
  rather than a target/binder pair with different chains; included since
  it is a real, documented multi-chain checkpoint, not because this tool
  recommends it for binder design specifically.

## When to use this instead of the alternatives
- `run_genie3_binder` is the direct sibling -- another target-conditioned
  binder generator, using a target-PDB-plus-hotspots JSON problem instead
  of a contig string, and a materially different (non-latent) diffusion
  architecture.
- Neither this tool nor `run_genie3_binder` scores the result. Refold with
  `run_chai1` or `run_boltz` with the target chain present, then score
  the interface with `run_ipsae`.
- Design (or verify) a sequence for the generated chain with `run_mpnn` --
  this tool's `num_mpnn_seqs` is always 0 (sequence design is run_mpnn's
  job, same rule as every other generative tool here), even with an
  all-atom `model` that places side-chain atoms of its own.

## What is NOT exposed, and why
- **`num_mpnn_seqs`** is always 0 -- sequence design is `run_mpnn`'s job.
- **`ssadj`** (fold-conditioning) is never set -- Protpardelle-1c's own
  README states "fold-conditional model weights will be released at a
  later date"; there is nothing to condition on yet.
- **Partial diffusion** (`motif_contigs: partial_diffusion`) is a
  different sampling task -- refining an EXISTING structure rather than
  conditioning on a fixed target chain -- and is not what this tool's
  `contig` grammar expresses; always disabled here.
- **Unconditional / single-chain motif scaffolding** (no target,
  `motifs`/`motif_contigs` without a chain break) is `run_genie2`'s,
  `run_frameflow`'s, `run_multiflow`'s and `run_genie3_scaffold`'s
  territory -- this tool is binder-specific (always at least one target
  chain from `target_pdb`, via the contig's own chain break).

## What you must supply
`target_pdb`, `contig`, `total_lengths` -- every one of these defines the
design task itself and has no safe default. `hotspots` may be `null`
(stated explicitly, never inherited from a default).

## What you get back
`samples`: one entry per generated PDB, each `{"id", "chains": [{
"chain_id", "length"}, ...]}` for every chain in that file (target
chain(s) included, unchanged across samples). `num_samples`, and under
`outputs` the path to every generated PDB.

## Parameters

| Parameter | Type | Required | Default | Constraints | Description |
|---|---|---|---|---|---|
| `target_pdb` | string | yes | `—` | pattern: `\.pdb$` | Structure file containing the target chain(s) contig and hotspots refer to. May contain extra residues beyond what contig selects -- Protpardelle-1c reads only the residues the contig names. WHERE THIS COMES FROM -- The target you want a binder against -- your own structure file, or a path a previous step returned. |
| `contig` | string | yes | `—` | pattern: `^(?:/\|[A-Za-z]?\d+-\d+)(?:;(?:/\|[A-Za-z]?\d+-\d+))*$` | Protpardelle-1c's own contig grammar (`;/;` for a chain break). The break is REQUIRED for a binder and must have a segment on BOTH sides. Confirmed by running the engine: `B2-505;/;80-120` returns two chains (A:504, B:108); `B2-505;80-120` runs but returns ONE fused 613-mer, not a binder; `B2-505;/` dies inside the engine on int('/'). All three are decided here rather than deep inside the engine. See the doc's worked examples. |
| `total_lengths` | array | yes | `—` | minItems: `1`<br>maxItems: `10` | List of [min, max] length pairs, one per chain in contig's own order (chain-break-separated segments) -- see the doc for why target chains' pairs must equal their exact residue count. Structural validity (each entry really is a 2-integer [min, max] pair, and the count matches contig's own chain count) is checked by the engine wrapper, which reports a clear error rather than letting a mismatch fail deep inside the model. |
| `hotspots` | — | yes | `—` | — | null runs without hotspot bias. A list of "{chain_id}{residue_index}" tags (e.g. "A19") biases generation toward those target residues -- see the doc's "hotspots" section. Each tag is validated against Protpardelle-1c's own grammar before dispatch; a malformed tag raises a clear error naming it. |
| `model` | string | no | `cc83` | enum: `['cc83', 'cc95', 'cc94', 'cc78']` | Which trained checkpoint to sample from -- see the doc's "model" section for what each one is. cc83 is Protpardelle-1c's own BindCraft-benchmark, backbone-only multi-chain model. |
| `step_scale` | number | no | `1.2` | minimum: `0.1`<br>maximum: `3.0` | Score scale during denoising. Higher values correspond to lower-temperature (less diverse, more consensus-like) sampling; lower values raise diversity. Protpardelle-1c's own recommended sweep range is 0.8-1.6, with 1.2 a good default in most cases (its own README). |
| `schurn` | number | no | `200.0` | minimum: `0.0`<br>maximum: `500.0` | Stochasticity magnitude injected during denoising; 0 is noise-free (deterministic ODE sampling when combined with step_scale 1.0). Protpardelle-1c's own README: schurn 200 works well for single-chain models but can hurt multi-chain all-atom models -- lower it (e.g. toward 0) if using model cc94 and samples look poor. |
| `crop_cond_start` | number | no | `0.0` | minimum: `0.0`<br>maximum: `1.0` | Fraction of total denoising steps before crop-conditional guidance begins being applied. Protpardelle-1c's own default is 0.0 (applied from the start). |
| `translation` | array | no | `[0.0, 0.0, 0.0]` | minItems: `3`<br>maxItems: `3` | Translate the input motif by [x, y, z] Angstroms before conditioning. Protpardelle-1c's own default is no translation; almost never needs changing. |
| `num_samples` | integer | no | `2` | minimum: `1`<br>maximum: `200` | Total number of samples to draw against this target. |
| `batch_size` | integer | no | `8` | minimum: `1`<br>maximum: `64` | Number of structures denoised in parallel per forward pass. Raise for throughput if GPU memory allows. |
| `seed` | integer | no | `—` | minimum: `0` | Random seed. Left unset, Protpardelle-1c's own CLI does not fix one (a fresh run varies call to call). |
