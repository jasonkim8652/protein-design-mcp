# run_multiflow

**Category:** monomer_generation  
**Engine:** `multiflow`  
**Environment:** `/home/jk661/.conda/envs/multiflow_fixed`  
**GPU required:** yes

> This file is generated from `src/protein_design_mcp/manifests/run_multiflow.yaml`. Edit the manifest, then run `python scripts/generate_tool_docs.py`.

## Summary

Generate an unconditional monomer backbone AND a ProteinMPNN-codesigned sequence for it in one call with MultiFlow, a discrete + continuous flow model (structure via SE(3) flow-matching, sequence via a discrete diffusion head over amino acid identities, jointly). MultiFlow's own built-in ESMFold self-consistency refold now runs to completion on this host and its bb_rmsd/mean_plddt are included in the reply -- confirmed live (bb_rmsd=0.637, mean_plddt=81.96 for a real sample). See the doc for the environment fix that unlocked this.

## What this is
MultiFlow's `multiflow/experiments/inference_se3_flows.py -cn
inference_unconditional`: a backbone is sampled by SE(3) flow-matching
while a sequence is sampled JOINTLY by a discrete diffusion head over the
20 amino acid types, conditioned on the evolving structure at each step
(co-design, not a separate downstream step). ProteinMPNN also runs
afterward on the finished backbone to produce a second candidate sequence
for the SAME backbone -- both are returned (see "What you get back").

## The ESMFold self-consistency step -- now unlocked (wave-I)
Immediately after writing `sample.pdb`, MultiFlow's own script
unconditionally attempts to refold the codesigned sequence with ESMFold to
score self-consistency. This USED TO crash with `ModuleNotFoundError: No
module named 'deepspeed'` (the bundled `openfold` imports it) in the
original `multiflow` env, which never had it installed -- confirmed live,
and NOT fixed by `inference.also_fold_pmpnn_seq`, which only gates a
second, separate, optional fold. This tool now runs under
`multiflow_fixed`, a clone with `deepspeed` installed (see `engine.prefix`
above), where the refold genuinely completes -- confirmed live:
`sc_results.csv` written with real `bb_rmsd`/`mean_plddt`, and an actual
ESMFold-refolded PDB on disk. `self_consistency` in each sample's reply
entry (and the `self_consistency_summary` output) carries this score;
`null` only in the defensive fallback documented in the wrapper's own
docstring (this tool's env missing deepspeed again), which should not
occur in normal operation on this host. You can still independently
re-score a result with `run_esmfold2` + `run_ipsae` if you want a second
opinion or a different predictor's number.

## `aatypes_temp`/`aatypes_noise` -- the sequence-codesign knobs
These control the discrete diffusion head's own sampling, independent of
the structure knobs: `aatypes_temp` is the sampling temperature for amino
acid identity at each denoising step (lower is more conservative/greedy,
higher is more diverse); `aatypes_noise` scales how much re-masking noise
is injected during decoding. Both are MultiFlow's own defaults (0.1 and
20.0 respectively) unless changed.

## When to use this instead of the alternatives
- `run_genie2`, `run_frameflow` and `run_genie3_scaffold` generate
  backbones only; this tool additionally gives you a designed sequence AND
  an ESMFold self-consistency score for it, in one call.
- `run_la_proteina` also produces a full atomic model with sequence, via
  a materially different (latent, autoencoder-mediated) architecture
  rather than joint discrete/continuous flow-matching.
- Either sequence this tool returns can still be independently verified:
  fold it with `run_chai1` or `run_boltz` and score the result with
  `run_ipsae`.

## What is NOT exposed, and why
- **`also_fold_pmpnn_seq`** is fixed to `false` -- it only gates a SECOND,
  entirely optional fold (the ProteinMPNN-designed sequence refolded
  separately, distinct from the codesigned-sequence refold this tool
  already reports via `self_consistency`); leaving it on would spend GPU
  time on a score this tool does not surface.
- **`write_sample_trajectories`** is fixed to `false` -- intermediate
  diffusion states, not part of this tool's declared output.
- **The checkpoint** is fixed to `weights/last_gpu0.ckpt`. MultiFlow's own
  default (`weights/last.ckpt`) was saved pinned to `cuda:1` and crashes
  under a single-visible-device dispatch (`RuntimeError: Attempting to
  deserialize object on CUDA device 1 but torch.cuda.device_count() is
  1`) -- confirmed live (survey). `last_gpu0.ckpt` is the repo's own
  device-0-pinned copy of the same weights, provided for exactly this
  case.
- **`interpolant.trans.potential`/`rog.weight`/`rog.cutoff`** (a
  radius-of-gyration guidance potential in the shipped YAML) is not
  exposed: `grep` across every `.py` file in this checkout found no code
  path that reads `trans.potential` or the `rog` block at sampling time
  at all -- it appears to be dead configuration, not a working knob, and
  this tool does not invent an effect for a field it could not verify
  does anything.

## What you must supply
Nothing beyond the length range -- this is unconditional generation.

## What you get back
`samples`: one entry per generated backbone, each `{"id", "length",
"codesign_sequence", "self_consistency"}` (`codesign_sequence` is `null`
if that sample's FASTA was not found, which should not happen on a
genuinely successful run but is handled rather than assumed;
`self_consistency` is `{"bb_rmsd", "mean_plddt"}` from MultiFlow's own
ESMFold refold, or `null` only in the defensive fallback described above).
`num_samples`, and under `outputs` the paths to every `backbones` PDB,
`codesign_sequences` FASTA, and `self_consistency_summary` (the same
scores, keyed by sample, as a file).

## Parameters

| Parameter | Type | Required | Default | Constraints | Description |
|---|---|---|---|---|---|
| `min_length` | integer | yes | `—` | minimum: `20`<br>maximum: `384` | Shortest length to sample, inclusive. MultiFlow's own unconditional config samples lengths up to 256 by default; longer lengths are unverified but not blocked here. |
| `max_length` | integer | yes | `—` | minimum: `20`<br>maximum: `384` | Longest length to sample, inclusive. Set equal to min_length to sample one length only. |
| `length_step` | integer | no | `10` | minimum: `1`<br>maximum: `384` | Gap between sampled lengths, from min_length to max_length. Ignored when min_length equals max_length. |
| `samples_per_length` | integer | no | `1` | minimum: `1`<br>maximum: `200` | Number of samples to draw AT EACH sampled length. MultiFlow's own default is 100; this tool defaults much lower since it multiplies with the number of lengths in [min_length, max_length]. |
| `num_timesteps` | integer | no | `500` | minimum: `1`<br>maximum: `1000` | Number of flow-integration steps for the structure. MultiFlow's own default is 500 (higher than FrameFlow's 100 -- a different tuned default for this model). More steps is finer-grained integration at roughly linear cost. |
| `do_sde` | boolean | no | `False` | — | Whether to integrate the structure flow with an SDE (stochastic) rather than the default ODE (deterministic) solver. MultiFlow's own default is false (ODE). SDE integration injects noise at every step, trading determinism for (potentially) more diverse samples. |
| `min_t` | number | no | `0.01` | minimum: `0.0001`<br>maximum: `0.5` | The minimum flow time the integrator runs to, stopping just short of t=0 for numerical stability. MultiFlow's own default is 0.01. |
| `self_condition` | boolean | no | `True` | — | Whether each structure-integration step also conditions on the model's own prediction from the previous step. MultiFlow's own default is true. |
| `trans_sample_temp` | number | no | `1.0` | minimum: `0.01`<br>maximum: `5.0` | Sampling temperature for the translation (CA position) component of the structure flow. MultiFlow's own default is 1.0; lower values sample closer to the model's mean prediction (less diverse), higher values inject more spread. |
| `aatypes_temp` | number | no | `0.1` | minimum: `0.01`<br>maximum: `5.0` | Sampling temperature for amino acid identity at each sequence-decoding step. MultiFlow's own default is 0.1 (fairly conservative/greedy); raising it samples more diverse sequences at the cost of lower per-step confidence. |
| `aatypes_noise` | number | no | `20.0` | minimum: `0.0`<br>maximum: `100.0` | Re-masking noise scale for the discrete sequence-diffusion head. MultiFlow's own default is 20.0. Higher values re-mask more aggressively during decoding, giving the model more chances to revise earlier choices at higher compute cost. |
| `aatypes_do_purity` | boolean | no | `True` | — | Whether sequence decoding uses "purity sampling" (committing to the most confident positions first each step) rather than a uniformly random decoding order. MultiFlow's own default is true. |
| `seed` | integer | no | `123` | minimum: `0` | Random seed. MultiFlow's own default is 123. |
