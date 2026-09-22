# run_frameflow

**Category:** monomer_generation  
**Engine:** `experiments`  
**Environment:** `/home/jk661/.conda/envs/frameflow`  
**GPU required:** yes

> This file is generated from `src/protein_design_mcp/manifests/run_frameflow.yaml`. Edit the manifest, then run `python scripts/generate_tool_docs.py`.

## Summary

Generate an unconditional monomer backbone with FrameFlow, an SE(3) flow-matching model (no sequence, no side chains -- design a sequence for the result with run_mpnn). Only unconditional generation is exposed; FrameFlow's motif-scaffolding entry point (a different Hydra config, inference_scaffolding.yaml) is not wired into this tool.

## What this is
FrameFlow's `experiments/inference_se3_flows.py -cn inference_unconditional`,
a flow-matching model over rigid-body frames (SE(3)) that samples backbone
coordinates for a chosen length, with no conditioning.

## `length_subset` must be nulled explicitly -- confirmed live
`inference_unconditional.yaml`'s own default sets
`inference.samples.length_subset` to a fixed `[70, 100, 200, 300]` --
confirmed live that this silently overrides `min_length`/`max_length`
regardless of what they are set to (an early version of this tool always
produced all four of those lengths no matter what was asked for). This
tool always passes `inference.samples.length_subset=null` first, so
`min_length`/`max_length`/`length_step` are what actually controls the
run.

## `inference.output_dir` does not work -- confirmed, and contained
Verified live (survey): passing `inference.output_dir=<path>` does NOT
change where FrameFlow writes -- it always writes to a cwd-relative
`./inference_outputs/...`, even though the saved `config.yaml` shows the
override took effect in the config object itself. This is contained here
because the dispatcher already runs this engine with its cwd set to a
fresh, disposable scratch directory for every call (see
`dispatch.env.EnvDispatcher`) -- so "writes wherever cwd is" and "writes
inside the scratch workdir this call owns" are the same thing. This tool
does not attempt to override `output_dir` at all, since doing so has no
effect and would be misleading to show in the argv.

## `checkpoint_variant` -- which trained model, not a decoration
FrameFlow ships three published checkpoints, each a genuinely different
trained model, not a config preset:
- `pdb` (default): trained on PDB monomers. This is `inference_unconditional
  .yaml`'s OWN default `ckpt_path` for unconditional sampling (confirmed
  from the config file itself).
- `pdb_amortization`: an amortization-trained variant. FrameFlow's own
  README states "by default, we use and recommend the amortization model"
  for MOTIF-SCAFFOLDING specifically -- not this tool's unconditional-only
  surface, but the checkpoint is a real, loadable alternative for
  unconditional sampling too, so it is exposed rather than hidden.
- `scope`: trained on SCOPe domain data (`data.dataset=scope` at training
  time) instead of full PDB chains -- a different structural distribution,
  useful when you want fold-domain-like outputs rather than full-length
  PDB-style chains.

## When to use this instead of the alternatives
- `run_genie2` and `run_genie3_scaffold` are the direct siblings -- other
  unconditional monomer generators. FrameFlow is a flow-matching model;
  Genie 2/3 are diffusion models. No single one is strictly better across
  all lengths; compare outputs if it matters for your use case.
- `run_multiflow` additionally co-designs a sequence with ProteinMPNN in
  the same call; this tool produces backbones only, designed afterward
  with `run_mpnn`.
- For a target-conditioned binder rather than a free-standing monomer, use
  `run_genie3_binder` or `run_protpardelle` instead.

## What you must supply
Nothing beyond the length range -- this is unconditional generation.

## What you get back
`backbones`: one entry per generated PDB, each `{"id", "length"}` (`length`
counted from the file's own CA atoms). `num_backbones`, and under `outputs`
the path to every generated `sample.pdb` (trajectory files are not
collected).

## Parameters

| Parameter | Type | Required | Default | Constraints | Description |
|---|---|---|---|---|---|
| `min_length` | integer | yes | `—` | minimum: `20`<br>maximum: `512` | Shortest length to sample, inclusive. FrameFlow's own unconditional benchmark in its README samples [70, 100, 200, 300]; lengths well outside that range are unverified but not blocked here. |
| `max_length` | integer | yes | `—` | minimum: `20`<br>maximum: `512` | Longest length to sample, inclusive. Set equal to min_length to sample one length only. |
| `length_step` | integer | no | `10` | minimum: `1`<br>maximum: `512` | Gap between sampled lengths, from min_length to max_length. Ignored when min_length equals max_length. |
| `samples_per_length` | integer | no | `1` | minimum: `1`<br>maximum: `200` | Number of samples to draw AT EACH sampled length (FrameFlow's own config name for this field, and its own default is 10 -- this tool defaults lower since this multiplies with the number of lengths in [min_length, max_length]; total runtime is approximately samples_per_length * number_of_lengths * per-sample cost (about 9s per 50-residue sample on this host's GPU 7, per the install survey). |
| `num_timesteps` | integer | no | `100` | minimum: `1`<br>maximum: `1000` | Number of flow-integration steps from noise to structure. FrameFlow's own default is 100. More steps means a finer-grained integration of the same flow field -- typically better geometry at roughly linear cost; fewer steps is faster but coarser. This is the direct analogue of a diffusion model's sampling-step count. |
| `min_t` | number | no | `0.01` | minimum: `0.0001`<br>maximum: `0.5` | The minimum flow time the integrator runs to (it stops just short of t=0 rather than exactly at it, for numerical stability). FrameFlow's own default is 0.01. Lower values integrate closer to the fully-denoised state at some risk of numerical instability near t=0; there is little reason to raise it above the default. |
| `self_condition` | boolean | no | `True` | — | Whether each integration step also conditions on the model's own structure prediction from the previous step (self-conditioning). FrameFlow's own default is true, and self-conditioning generally improves sample quality in this model family; set false only to isolate its effect. |
| `checkpoint_variant` | string | no | `pdb` | enum: `['pdb', 'pdb_amortization', 'scope']` | Which trained checkpoint to sample from. See the doc's "checkpoint_variant" section for what each one was trained on. `pdb` is FrameFlow's own default for unconditional sampling. |
| `seed` | integer | no | `123` | minimum: `0` | Random seed. FrameFlow's own default is 123; fixed here too so a repeated call is reproducible unless explicitly varied. |
