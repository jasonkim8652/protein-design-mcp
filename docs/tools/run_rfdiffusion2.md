# run_rfdiffusion2

**Category:** binder_generation  
**Engine:** `rfdiffusion2`  
**Environment:** `/home/jk661/.conda/envs/rfd2_src`  
**GPU required:** yes

> This file is generated from `src/protein_design_mcp/manifests/run_rfdiffusion2.yaml`. Edit the manifest, then run `python scripts/generate_tool_docs.py`.

## Summary

Generate de novo binder backbones against a target with RFdiffusion2, the open-source all-atom successor to RFdiffusion 1.1.0. Runs the official container image (converted from upstream's Apptainer .sif, since Apptainer itself cannot run on this host) as a sibling Docker container. Backbone only (pair with run_mpnn). CONFIRMED LIVE end to end on GPU 7. Cannot condition on hotspot residues -- the only checkpoints available on this host were never trained with hotspot conditioning; use run_rfdiffusion3_binder when hotspot-directed interface targeting matters more than RFdiffusion2's all-atom architecture.

## What this is
RFdiffusion2 (Krishna et al.; Baker lab / IPD), the all-atom successor to
RFdiffusion 1.1.0, run through the official container image rather than
a source workaround (see the manifest's `engine` comment and
`scripts/engines/rfdiffusion2.py`'s module docstring for the full
reasoning: Apptainer cannot run on this host at all -- a kernel policy
issue, confirmed, not a missing package -- so the same `.sif` was
converted to a Docker image instead; every live confirmation below was
run through that image).

## A sibling container, launched from inside this server
This is the only tool in this project whose wrapper shells out to
`docker run` rather than running its engine as a plain subprocess in a
mounted host conda environment. That has one real consequence for
deployment: wherever this wrapper itself executes needs a working
`docker` client with access to a Docker daemon that can see GPU 7 (the
Docker-outside-of-Docker pattern, if this MCP server is itself deployed
inside a container). That is an infrastructure prerequisite this wave's
file scope cannot provision -- it is stated here rather than silently
assumed. Every "CONFIRMED LIVE" claim in this doc was run directly on
the host, the same way the wrapper itself invokes `docker`; whether the
currently-deployed server container has the matching socket/CLI access
was not possible to verify from within this wave.

## When to use this instead of the alternatives -- three generations, one lineage
- `run_rfdiffusion_binder` (RFdiffusion 1.1.0): the legacy baseline, a
  different (space-separated) contig grammar.
- **This tool**: all-atom architecture (ligand/nucleic-acid awareness
  exists upstream but is not exposed here yet -- see "Not exposed"), a
  third contig grammar (underscore-separated chains, comma-separated
  sub-ranges within a chain -- see "`contig`" below), and CANNOT use
  hotspot conditioning on this host (see "Hotspots are not available").
- `run_rfdiffusion3_binder`: RosettaCommons' current production model, a
  completely different JSON conditioning schema, with WORKING hotspot
  conditioning. Prefer it when directing the binder to a specific
  interface patch matters more than RFdiffusion2's all-atom lineage.

## `contig` -- RFdiffusion2's own grammar (neither of the other two generations')
Underscores separate CHAINS; commas separate sub-ranges WITHIN one chain
(CONFIRMED LIVE by reading `rf_diffusion/contigs.py::get_sampled_mask`,
which literally does `contigs[0].split('_')` then, per chain,
`.split(',')`). Each sub-range is either a target segment
`<Chain><start>-<end>` (e.g. `A1-150`) pulled from `target_pdb`, or a
diffused-length range `<min>-<max>` / fixed `<n>-<n>`. Worked example --
a 5-residue target motif (chain A, residues 1-5) followed by a 10-residue
binder chain: `contig: "A1-5_10-10"`. Do not reuse a contig string
written for `run_rfdiffusion_binder` (space-separated, `/0` breaks) or
`run_rfdiffusion3_binder` (comma-separated, no chain-break token at all)
here without rewriting it.
`contigmap.has_termini` (a required RFdiffusion2 flag whenever more than
one chain is described) is derived automatically from the number of
underscore-separated chain segments in `contig` -- not a separate
parameter, since it is a mechanical consequence of `contig` and exposing
it separately would only invite the two going out of sync.

## Hotspots are not available on this host -- confirmed, not a policy choice
RFdiffusion2's PPI-specialised checkpoint (`aa_ppi.yaml`'s own default,
trained with `FindHotspotsTrainingTransform`) is **not present on this
host** -- only the two general-purpose checkpoints below are. CONFIRMED
LIVE: passing `ppi.hotspot_res` through either of those checkpoints
raises `AssertionError: Model not set up for hotspots` from
`rf_diffusion/ppi.py` (the checkpoint's OWN saved training-transform list
is what is checked, not anything this tool's invocation controls) --
this is why this tool's schema has no hotspot parameter at all, rather
than one that always fails. Centering instead uses
`transforms.configs.CenterPostTransform.center_type=is_not_diffused`
(center on the target's own center of mass) -- CONFIRMED LIVE to work
without hotspots and without requiring an `ORI` HETATM record in
`target_pdb`, which this tool has no way to supply.

## `ckpt_variant`
`"140"` (`RFD_140.pt`, this tool's default, matching the upstream `aa`
config's own default) or `"173"` (`RFD_173.pt`, a later training
checkpoint) -- the two general-purpose all-atom checkpoints present on
this host (CONFIRMED, initial engine survey). Neither supports hotspot
conditioning (see above).

## Important caveats
- **Backbone only.** Follow with `run_mpnn`.
- The first call after this image starts a fresh container builds a
  one-time SO(3) rotation-schedule cache (a few minutes, CONFIRMED LIVE:
  ~2.5 of this tool's ~2.6-minute total smoke-test runtime) -- expect the
  first call in any batch to be much slower than the rest.
- Ligand conditioning, nucleic-acid conditioning, partial diffusion, and
  symmetry are real RFdiffusion2 capabilities not exposed here -- none
  were exercised in this tool's verification.

## What you get back
`num_structures`, and under `outputs` the paths to every `structures`
PDB and its paired `metadata_trb`.

## Parameters

| Parameter | Type | Required | Default | Constraints | Description |
|---|---|---|---|---|---|
| `target_pdb` | string | yes | `—` | pattern: `\.pdb$` | Path to the target structure (PDB format). Crop it around the intended interface before diffusing against a large target. |
| `contig` | string | yes | `—` | pattern: `^([A-Za-z]\d+(-\d+)?\|\d+(-\d+)?)(,([A-Za-z]\d+(-\d+)?\|\d+(-\d+)?))*(_([A-Za-z]\d+(-\d+)?\|\d+(-\d+)?)(,([A-Za-z]\d+(-\d+)?\|\d+(-\d+)?))*)*$` | RFdiffusion2's own contig grammar (see the doc's "contig" section for the full explanation) -- underscores separate chains, commas separate sub-ranges within one chain: a target segment like "A1-150" pulled from target_pdb, or a diffused-length range like "10-10". Do not wrap it in outer brackets or quotes -- this tool adds those itself. NOT the same grammar as run_rfdiffusion_binder's or run_rfdiffusion3_binder's contig. |
| `num_designs` | integer | no | `10` | minimum: `1`<br>maximum: `1000` | Number of independent backbones to sample. Cost scales linearly. Drop it to 1 for a fast sanity check on a new contig before committing to a larger batch. |
| `diffusion_steps` | integer | no | `100` | minimum: `1`<br>maximum: `200` | Number of denoising timesteps (diffuser.T). 100 is RFdiffusion2's own base-config default (unlike RFdiffusion 1.1.0's 50). 15 was the lowest value exercised in this tool's own verification (a fast sanity-check run); no assertion enforcing a hard floor the way RFdiffusion 1.1.0 does was observed here, but values well below 15 have not been tried. |
| `noise_scale_ca` | number | no | `1.0` | minimum: `0.0`<br>maximum: `2.0` | Translational noise scale for the Cα denoiser. Same meaning as run_rfdiffusion_binder's parameter of the same name. |
| `noise_scale_frame` | number | no | `1.0` | minimum: `0.0`<br>maximum: `2.0` | Rotational noise scale for the frame (orientation) denoiser. Same meaning as run_rfdiffusion_binder's parameter of the same name. |
| `ckpt_variant` | string | no | `140` | enum: `['140', '173']` | Which general-purpose all-atom checkpoint to use -- see the doc's "ckpt_variant" section for why only these two (of RFdiffusion2's several trained checkpoints) are available on this host. "140" (RFD_140.pt) matches the upstream aa config's own default. |
