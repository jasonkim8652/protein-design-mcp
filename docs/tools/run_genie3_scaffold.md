# run_genie3_scaffold

**Category:** monomer_generation  
**Engine:** `genie3`  
**Environment:** `/home/jk661/.conda/envs/genie2`  
**GPU required:** yes

> This file is generated from `src/protein_design_mcp/manifests/run_genie3_scaffold.yaml`. Edit the manifest, then run `python scripts/generate_tool_docs.py`.

## Summary

Generate an unconditional monomer backbone with Genie 3, an all-atom SE(3)-equivariant diffusion model (backbone frames only here -- no sequence, no side chains; design a sequence for the result with run_mpnn). Only unconditional generation is exposed; Genie 3's motif-scaffolding and binder-design entry points are separate (the latter is `run_genie3_binder` (not yet implemented)).

## What this is
Genie 3's unconditional generation path (`genie3.cli generate`, dataset
source `unconditional`), run with either of the two checkpoints downloaded
on this host.

## `model_variant` -- which trained model and sampler, not a decoration
- `v1` (default): the current model (`pretrained/v1/checkpoints/
  step=600000.ckpt`), sampled with Genie 3's DDIM sampler. `direction_scale`,
  `eta`, `n_sample_step`, `noise_scale` and `predict_sidechain` all apply.
- `legacy`: the Genie-2-compatible checkpoint (`pretrained/legacy/
  checkpoints/step=400000.ckpt`), sampled with the simpler DDPM sampler,
  which takes only `noise_scale` -- `direction_scale`, `eta`,
  `n_sample_step` and `predict_sidechain` are silently ignored when this
  variant is selected (DDPM has no equivalent knobs), documented here
  rather than left to be discovered by a caller wondering why they had no
  effect.

## The DDIM knobs, read from source
(`genie3/generation/diffusion/sampler/ddim.py`, read directly -- these are
not documented in the README):
- `direction_scale`: scales the predicted denoising direction at each
  step. No shipped default (Genie 3's own config requires a value); higher
  values follow the model's predicted direction more strongly.
- `eta`: stochasticity, 0 (fully deterministic, ODE-like) to 1
  (DDPM-equivalent stochastic sampling). Genie 3's own default is 1.0.
- `n_sample_step`: number of denoising steps, out of a fixed 1000-step
  training schedule (can be fewer). Genie 3's own default is 100; more
  steps is finer-grained integration at roughly linear cost.
- `noise_scale`: scales injected noise at each step. Genie 3's own default
  is 1.0.
- `predict_sidechain`: whether the model also predicts sidechain atoms
  (all-atom output) rather than backbone frames only. Genie 3's own
  default is false.

## When to use this instead of the alternatives
- `run_genie2`, `run_frameflow` and `run_la_proteina`
  (not yet implemented) are the other unconditional monomer generators
  (diffusion, flow-matching and latent-flow-matching respectively). No
  single one is strictly better across all lengths.
- `run_multiflow` (not yet implemented) additionally co-designs a
  sequence; this tool produces backbones only.
- For a target-conditioned binder, use `run_genie3_binder`
  (not yet implemented) or `run_protpardelle` (not yet implemented)
  instead -- this tool has no notion of a target at all.

## What is NOT exposed, and why
- **`predict_sequence`** (Genie 3's own built-in codesign head) is fixed
  to `false` -- sequence design is `run_mpnn`'s job, the same rule that
  excludes it from every other generative tool in this wave.
- **Motif scaffolding and binder design** are Genie 3's own separate
  applications with entirely different dataset/conditioning config, not
  exposed by this tool.

## What you must supply
Nothing beyond the length range -- this is unconditional generation.

## What you get back
`backbones`: one entry per generated PDB, each `{"id", "length"}` (`length`
counted from the file's own CA atoms). `num_backbones`, and under
`outputs` the path to every generated PDB.

## Parameters

| Parameter | Type | Required | Default | Constraints | Description |
|---|---|---|---|---|---|
| `min_length` | integer | yes | `—` | minimum: `5`<br>maximum: `512` | Shortest length to sample, inclusive. |
| `max_length` | integer | yes | `—` | minimum: `5`<br>maximum: `512` | Longest length to sample, inclusive. Set equal to min_length to sample one length only. |
| `length_step` | integer | no | `50` | minimum: `1`<br>maximum: `512` | Gap between sampled lengths, from min_length to max_length. Ignored when min_length equals max_length. |
| `num_samples` | integer | no | `2` | minimum: `1`<br>maximum: `200` | Number of samples to draw AT EACH sampled length (not a total count -- multiplies with the number of lengths in [min_length, max_length]). |
| `batch_size` | integer | no | `4` | minimum: `1`<br>maximum: `64` | Number of structures denoised in parallel per forward pass. |
| `model_variant` | string | no | `v1` | enum: `['v1', 'legacy']` | Which trained checkpoint and sampler to use. See the doc's "model_variant" section -- `legacy` ignores every DDIM-only parameter below. |
| `direction_scale` | number | no | `1.0` | minimum: `0.0`<br>maximum: `5.0` | DDIM sampler only (ignored for model_variant=legacy). Scales the predicted denoising direction at each step; Genie 3's own config requires an explicit value (no shipped default). Higher values follow the model's prediction more strongly. |
| `eta` | number | no | `1.0` | minimum: `0.0`<br>maximum: `1.0` | DDIM sampler only (ignored for model_variant=legacy). Stochasticity: 0 is deterministic (ODE-like), 1 is DDPM-equivalent stochastic sampling. Genie 3's own default is 1.0. |
| `n_sample_step` | integer | no | `100` | minimum: `1`<br>maximum: `1000` | DDIM sampler only (ignored for model_variant=legacy). Number of denoising steps out of the model's fixed 1000-step training schedule. Genie 3's own default is 100; more steps is finer-grained at roughly linear cost. |
| `noise_scale` | number | no | `1.0` | minimum: `0.0`<br>maximum: `2.0` | Scales injected noise at each denoising step. Applies to both samplers. Genie 3's own default is 1.0 for both. |
| `predict_sidechain` | boolean | no | `False` | — | DDIM sampler only (ignored for model_variant=legacy). Whether the model also predicts sidechain atoms (all-atom output) rather than backbone frames only. Genie 3's own default is false. |
| `seed` | integer | no | `0` | minimum: `0` | Random seed. |
