# run_la_proteina

**Category:** monomer_generation  
**Engine:** `la_proteina`  
**Environment:** `/home/jk661/.conda/envs/laproteina_env`  
**GPU required:** yes

> This file is generated from `src/protein_design_mcp/manifests/run_la_proteina.yaml`. Edit the manifest, then run `python scripts/generate_tool_docs.py`.

## Summary

Generate an unconditional, all-atom monomer with La-Proteina, a latent flow-matching model (backbone AND side-chain coordinates, unlike the backbone-only diffusion/flow tools in this category -- no separate sequence-design step is needed to get a full atomic model, though the designed sequence still comes from this model's own decoding, not from a dedicated inverse-folding pass). Only the "LD1/AE1" unconditional checkpoint pair (generation up to 500 residues, no triangular-update layers) is downloaded on this host; motif-scaffolding checkpoints (LD4-7) are not.

## What this is
La-Proteina's `proteinfoundation/generate.py --config_name
inference_ucond_notri`-equivalent unconditional sampling path: a latent
diffusion/flow model whose latents are decoded through a matching
autoencoder checkpoint into full atom37 coordinates (all atoms, not just
the backbone frame). This tool always uses the LD1 (`LD1_ucond_notri_512
.ckpt`) + AE1 (`AE1_ucond_512.ckpt`) pair -- the only checkpoints
downloaded on this host, verified LD1 trained up to 500 residues with no
triangular-update layers.

## When to use this instead of the alternatives
- `run_genie2`, `run_frameflow` and `run_genie3_scaffold` generate
  BACKBONE-ONLY monomers; this tool produces every heavy atom directly,
  at the cost of a materially different (latent, autoencoder-mediated)
  architecture. If all you need is a backbone to design a sequence for
  afterward with `run_mpnn`, the backbone-only tools are simpler and this
  tool's extra atoms would be thrown away.
- `run_multiflow` also co-designs a sequence, via a discrete diffusion
  head over amino acid TYPES on a backbone-only structure, and can still
  be refolded/scored independently; this tool's sequence comes from the
  SAME latent decode as its structure, which is a different kind of
  guarantee (self-consistent by construction, but not independently
  checkable the way a separate inverse-folding step is).
- For a target-conditioned binder, none of the monomer-generation tools
  apply -- use `run_genie3_binder` or `run_protpardelle` instead.

## What is NOT exposed, and why
- **Motif scaffolding** (La-Proteina's LD4-7 checkpoints, indexed/unindexed
  x all-atom/tip-atom) -- those checkpoints are not downloaded on this
  host. Only the LD1/AE1 unconditional pair is.
- **Autoguidance** (`ag_ratio`/`ag_ckpt_path`) -- La-Proteina's own
  `inference_base.yaml` comment states "ag not supported for now"; this
  tool fixes `ag_ratio=0.0`, `ag_ckpt_path=null` rather than exposing a
  knob the upstream engine itself says does not work.
- **Designability/novelty/FID metrics** -- La-Proteina's own generation
  config can compute these inline (several invoke ESMFold internally).
  This tool forces every one of them off: it exposes GENERATION only, like
  every other tool in this wave; score a result afterward with
  `run_esmfold2` + `run_ipsae`, or design a sequence
  for it with `run_mpnn` if you want an independent (non-self-decoded)
  one.

## Where La-Proteina actually writes, and how this tool contains it
`generate.py` is not an installed package (`sys.path.insert(0,
os.path.abspath("."))` at import time) and its Hydra config loader
resolves config files relative to its OWN file location, not any path
this tool controls -- so this tool's wrapper script writes a uniquely
named per-call config into the La-Proteina checkout's own `configs/`
directory, runs the engine with the checkout as cwd (required for the
import to work at all), then copies the result into this call's actual
scratch directory and deletes both the temporary config and the
checkout-side output copy. See the wrapper script's own docstring for the
full reasoning. This is the same class of "engine will not write where
told" containment problem as FrameFlow's cwd-relative output, solved
differently because the constraint here is Hydra's own config-path
resolution rather than a fixed output-dir default.

## What you must supply
`lengths` -- everything else has a documented default.

## What you get back
`backbones`: one entry per generated PDB, each `{"id", "length"}` (`length`
counted from the file's own CA atoms). `num_backbones`, and under `outputs`
the path to every generated PDB.

## Parameters

| Parameter | Type | Required | Default | Constraints | Description |
|---|---|---|---|---|---|
| `lengths` | array | yes | `—` | minItems: `1`<br>maxItems: `20` | Exact residue lengths to sample, e.g. [100, 200]. The downloaded LD1 checkpoint was trained for unconditional generation up to 500 residues; La-Proteina's own defaults sample [100, 200, 300, 400, 500]. Each length draws `num_samples` independent structures. |
| `num_samples` | integer | no | `2` | minimum: `1`<br>maximum: `100` | Number of samples to draw AT EACH length in `lengths` (not a total count). La-Proteina's own default is 100; this tool defaults much lower for a single interactive call. Total runtime is approximately len(lengths) * num_samples * per-sample cost. |
| `max_nsamples_per_batch` | integer | no | `2` | minimum: `1`<br>maximum: `50` | Maximum number of structures batched together in one forward pass. Raise for throughput if GPU memory allows; La-Proteina's own default is 10. Has no effect on the samples produced, only how they are grouped for computation. |
| `nsteps` | integer | no | `400` | minimum: `5`<br>maximum: `1000` | Number of flow-integration steps. La-Proteina's own default is 400. More steps is a finer-grained integration at roughly linear cost; this host's own install verification used 20 for a fast smoke test, which is far below the recommended value and only appropriate for confirming the path works, not for a real generation. |
| `self_cond` | boolean | no | `True` | — | Whether each integration step also conditions on the model's own prediction from the previous step. La-Proteina's own default is true. |
| `sc_scale_noise` | number | no | `0.1` | minimum: `0.0`<br>maximum: `2.0` | Noise scale applied during the model's stochastic-corrector sampling mode (`sampling_mode: sc`, La-Proteina's own default sampler). Higher values inject more stochasticity into each step (more diverse, more likely to drift from a clean trajectory); La-Proteina's own default is 0.1 for both the backbone and latent-variable components. |
| `sc_scale_score` | number | no | `1.0` | minimum: `0.0`<br>maximum: `5.0` | Scale applied to the model's own predicted score (denoising direction) during stochastic-corrector sampling. La-Proteina's own default is 1.0; lowering it weakens how strongly each step follows the model's prediction relative to the injected noise above. |
| `guidance_w` | number | no | `1.0` | minimum: `0.0`<br>maximum: `5.0` | Classifier-free guidance weight. La-Proteina's own convention: 1.0 means guidance is off (the main model alone), lower values move toward excluding the main model's unconditional prediction. Values other than 1.0 are not exercised on this host and are unverified; the upstream README does not tie an unconditional-sampling recommendation to any value besides the 1.0 default. |
| `seed` | integer | no | `5` | minimum: `0` | Random seed. La-Proteina's own `inference_base.yaml` default is 5. |
