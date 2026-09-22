# run_rfdiffusion2

**Category:** binder_generation  
**Engine:** `rfdiffusion2`  
**Environment:** `/home/jk661/.conda/envs/rfd2_fixed`  
**GPU required:** yes

> This file is generated from `src/protein_design_mcp/manifests/run_rfdiffusion2.yaml`. Edit the manifest, then run `python scripts/generate_tool_docs.py`.

## Summary

Generate de novo binder backbones against a target with RFdiffusion2, the open-source all-atom successor to RFdiffusion 1.1.0. Dispatches through its conda prefix by default, like every other engine here -- CONFIRMED LIVE end to end through ServerApp.call_tool on GPU 7, 2026-09-22, in the `rfd2_fixed` environment (a clone of `rfd2_src` with its missing dependencies -- pydantic, a working scipy/numpy pairing, and `fire` -- closed; `rfd2_src` itself was left untouched). `backend: "docker"` runs the official container image instead, also CONFIRMED LIVE working end to end, but requires this server's deployment to have explicitly granted Docker socket access -- read the doc's "backend" section before enabling it. Cannot condition on hotspot residues either way -- the only checkpoints available on this host were never trained with hotspot conditioning; use run_rfdiffusion3_binder when that matters more than this architecture.

## What this is
RFdiffusion2 (Krishna et al.; Baker lab / IPD), the all-atom successor to
RFdiffusion 1.1.0.

## `backend` -- read this before choosing "docker"
This tool can run two ways, chosen by the `backend` parameter:
- `"conda"` (the default, and this tool's default `engine:` dispatch):
  runs directly in this host's `rfd2_fixed` conda environment, the exact
  same mechanism every other GPU tool in this server uses -- no extra
  host capability beyond what every other tool already needs. **NOT the
  upstream-supported invocation path** (RFdiffusion2's own README only
  documents the Apptainer route) but the one that does not require
  handing this server root-equivalent access to the host. **CONFIRMED
  LIVE to complete a generation on this host**, 2026-09-22, through
  `ServerApp.call_tool` end to end on GPU 7 (`contig="A1-5_10-10"`,
  `num_designs=1`, `diffusion_steps=15`, `ckpt_variant="140"` -- a real
  `design_0-atomized-bb-False.pdb`/`.trb` pair was produced).
  `rfd2_fixed` is a clone of `rfd2_src` (left untouched) with three gaps
  closed that a bare `import rf_diffusion` does not exercise but a real
  run does: `pydantic` (missing outright -- `dgl`'s own import chain
  needs it), a broken scipy/numpy pairing (an earlier, unrelated `pip
  install` had left a stray newer numpy partially overwriting the
  environment's numpy while leaving its own dist-info stale, which then
  made scipy's compiled extensions fail to import against the numpy
  actually on disk -- fixed by purging and cleanly reinstalling
  `numpy==1.26.4`), and `fire` (missing outright -- imported directly by
  RFdiffusion2's own inference entrypoint script). None of this touched
  `rfd2_src` or its torch/dgl/cuda pins.
- `"docker"`: launches the OFFICIAL upstream container image
  (`rfdiffusion2-sif:converted`, converted via `unsquashfs` + `docker
  import` from RFdiffusion2's own Apptainer `.sif` -- Apptainer itself
  cannot run on this host, a kernel `apparmor` policy, confirmed, not a
  misconfiguration) as a SIBLING Docker container over the host's Docker
  socket. CONFIRMED LIVE working end to end on GPU 7, including through
  this server's own real dispatch path. **This requires wherever this
  server's dispatcher actually runs to have `/var/run/docker.sock`
  bind-mounted in and the `docker` CLI installed** (the
  "Docker-outside-of-Docker" pattern) -- granting that is granting
  root-equivalent control of the host to anything that reaches it, which
  is every tool in this server plus anyone who reaches the HTTP
  transport (unauthenticated as of this writing). That is NOT part of
  this server's default container recipe and should be enabled only by
  an operator who has explicitly decided to accept that trade-off for a
  specific host-side deployment, not assumed. If your deployment has not
  made that choice, `backend: "docker"` will fail to find a working
  `docker` client/socket -- that failure is the capability boundary
  working as intended, not a bug.

## When to use this instead of the alternatives -- three generations, one lineage
- `run_rfdiffusion_binder` (RFdiffusion 1.1.0): the legacy baseline, a
  different (space-separated) contig grammar, CONFIRMED working end to
  end through its own (plain conda-prefix) dispatch.
- **This tool**: all-atom architecture (ligand/nucleic-acid awareness
  exists upstream but is not exposed here yet -- see "Not exposed"), a
  third contig grammar (underscore-separated chains, comma-separated
  sub-ranges within a chain -- see "`contig`" below), CANNOT use hotspot
  conditioning on this host (see "Hotspots are not available"), and its
  default dispatch mode is CONFIRMED to complete a run here (see
  "`backend`" above).
- `run_rfdiffusion3_binder`: RosettaCommons' current production model, a
  completely different JSON conditioning schema, with WORKING hotspot
  conditioning, CONFIRMED working end to end through its own plain
  conda-prefix dispatch (no Docker, no capability trade-off). Prefer it
  for binder-design work right now unless RFdiffusion2's all-atom
  lineage specifically matters and your deployment has opted into
  `backend: "docker"`.

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
LIVE (through `backend: "docker"`, the only mode that has completed a
run to reach this code path): passing `ppi.hotspot_res` through either
of those checkpoints raises `AssertionError: Model not set up for
hotspots` from `rf_diffusion/ppi.py` (the checkpoint's OWN saved
training-transform list is what is checked, not anything this tool's
invocation controls) -- this is why this tool's schema has no hotspot
parameter at all, rather than one that always fails. Centering instead
uses `transforms.configs.CenterPostTransform.center_type=is_not_diffused`
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
- Under `backend: "docker"`, the first call after the image starts a
  fresh container builds a one-time SO(3) rotation-schedule cache (a few
  minutes, CONFIRMED LIVE: ~2.5 of this tool's ~2.6-minute total
  smoke-test runtime) -- expect the first call in any batch to be much
  slower than the rest.
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
| `backend` | string | no | `conda` | enum: `['conda', 'docker']` | "conda" (default) runs in this host's rfd2_fixed conda environment -- no extra capability needed, CONFIRMED LIVE to complete a generation on this host, 2026-09-22, through ServerApp.call_tool end to end on GPU 7 -- see the doc's "backend" section for the exact case and what rfd2_fixed closes relative to rfd2_src. "docker" runs the official upstream container image as a sibling container, also CONFIRMED LIVE working end to end, but requires this server's own deployment to have explicitly granted Docker socket access -- a root-equivalent capability, not part of the default container recipe. Read the doc's "backend" section in full before setting this to "docker". |
| `num_designs` | integer | no | `10` | minimum: `1`<br>maximum: `1000` | Number of independent backbones to sample. Cost scales linearly. Drop it to 1 for a fast sanity check on a new contig before committing to a larger batch. |
| `diffusion_steps` | integer | no | `100` | minimum: `1`<br>maximum: `200` | Number of denoising timesteps (diffuser.T). 100 is RFdiffusion2's own base-config default (unlike RFdiffusion 1.1.0's 50). 15 was the lowest value exercised in this tool's own verification (a fast sanity-check run); no assertion enforcing a hard floor the way RFdiffusion 1.1.0 does was observed here, but values well below 15 have not been tried. |
| `noise_scale_ca` | number | no | `1.0` | minimum: `0.0`<br>maximum: `2.0` | Translational noise scale for the Cα denoiser. Same meaning as run_rfdiffusion_binder's parameter of the same name. |
| `noise_scale_frame` | number | no | `1.0` | minimum: `0.0`<br>maximum: `2.0` | Rotational noise scale for the frame (orientation) denoiser. Same meaning as run_rfdiffusion_binder's parameter of the same name. |
| `ckpt_variant` | string | no | `140` | enum: `['140', '173']` | Which general-purpose all-atom checkpoint to use -- see the doc's "ckpt_variant" section for why only these two (of RFdiffusion2's several trained checkpoints) are available on this host. "140" (RFD_140.pt) matches the upstream aa config's own default. |
