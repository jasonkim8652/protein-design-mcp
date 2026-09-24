# run_rfdiffusion_binder

**Category:** binder_generation  
**Engine:** `rfdiffusion`  
**Environment:** `/home/jk661/.conda/envs/SE3nv`  
**GPU required:** yes

> This file is generated from `src/protein_design_mcp/manifests/run_rfdiffusion_binder.yaml`. Edit the manifest, then run `python scripts/generate_tool_docs.py`.

## Summary

Generate de novo binder backbones against a target with RFdiffusion 1.1.0 -- the original RFdiffusion, and the legacy baseline of this lineage. Backbone only, no sequence (pair with run_mpnn). Conditions on the target through an explicit contig string plus optional hotspot residues. Superseded in this server by run_rfdiffusion2 (open-source SOTA, all-atom/ligand-aware) and run_rfdiffusion3_binder (RosettaCommons' current production binder model); use this one for reproducing published RFdiffusion 1.1.0 results, or when the newer generations' extra conditioning machinery is not needed and its simpler, best-understood contig grammar is preferred.

## What this is
RFdiffusion 1.1.0 (Watson et al. 2023; Baker lab / IPD), the original
RFdiffusion, run from the actual checkout at
the RFdiffusion checkout the deployment mounts (the package's own pip
metadata points at a now-nonexistent path -- see the manifest's engine
comment; this tool works around that with an explicit PYTHONPATH, and
ALSO works around a live nvrtc/JIT compilation crash on this host's
GPUs -- state both when reporting results from this tool, since either
workaround failing silently would look like a hang, not an error).
It is the legacy baseline of a three-generation lineage this server also
exposes as `run_rfdiffusion2` and `run_rfdiffusion3_binder` -- see "When
to use this instead" below for what actually differs between them.

## What it is for
Diffusing a novel protein backbone that contacts a fixed target structure
at residues you choose. It produces coordinates only -- no sequence -- so
a real design campaign always follows this with `run_mpnn` (sequence) and
a structure predictor (to check the sequence actually refolds to this
backbone).

## When to use this instead of the alternatives -- three generations, one lineage
This server exposes three RFdiffusion generations because their contig
and conditioning grammars differ enough that picking the wrong one for a
task silently gets worse results, not an error:
- **This tool (RFdiffusion 1.1.0)**: backbone-only diffusion over a
  residue-level frame representation. Conditions on the target with a
  single flat contig string (`contigs`) plus `hotspot_res`. No native
  ligand, nucleic-acid or fine-grained atom-level conditioning. Choose
  this to reproduce published RFdiffusion 1.1.0 results, or when you want
  the simplest, longest-battle-tested contig grammar in this lineage.
- `run_rfdiffusion2`: the all-atom successor (Krishna et al.), aware of
  ligands and nucleic acids and conditioned through the same
  `contigmap.contigs` / `ppi.hotspot_res` grammar as this tool (so
  contigs written for this tool mostly transfer). Its documented
  invocation path needs Apptainer, which is **not installed on this
  host**; this server runs it through a config-loading workaround
  instead -- see that tool's own doc for why that matters, and note its
  live-generation status there before treating it as a drop-in upgrade.
- `run_rfdiffusion3_binder`: RosettaCommons' current production model
  (`rc-foundry`'s `rfd3`), a from-scratch rebuild with a completely
  different, pydantic-validated JSON conditioning schema (`contig` +
  `select_hotspots`, comma-separated with `/0` chain breaks, no spaces --
  NOT the same grammar as this tool's `contigs`/`hotspot_res`, despite
  looking similar). No Apptainer dependency, confirmed working end to end
  live on this host. Prefer this one for new binder-design work unless
  you specifically need RFdiffusion 1.1.0 or RFdiffusion2 behaviour.

## `contig` -- RFdiffusion 1.1.0's own grammar, not RFdiffusion3's
Space-separated tokens, each one of:
- a target segment: `<Chain><start>-<end>` (e.g. `B1-100`), optionally
  followed immediately (no space) by `/0` to mark a chain break after it
  (e.g. `B1-100/0`) -- `/0` can also stand alone as its own token.
- a diffused-length range: `<min>-<max>` (e.g. `100-150`), or a single
  integer for a fixed length (e.g. `100-100`).
Worked example -- a 100-residue binder against residues 1-100 of target
chain B: `contig: "B1-100/0 100-100"`. Target segments always come first
in this tool's convention (unlike `run_rfdiffusion3_binder`, where the
diffused region conventionally comes first). Providing the whole,
uncropped target makes diffusion slow (RFdiffusion scales ~O(N^2) in
total residue count); crop the target around the interface and rely on
`hotspot_res` to pin the binding site instead of relying on target size
alone.

## `hotspot_res` -- format enforced, not just documented
Each entry must be exactly `<ChainID><ResidueNumber>` (e.g. `A30`) --
matching the target's chain ID and residue numbering in `target_pdb`. The
earlier form of this parameter in this codebase accepted any string
(`"45"`, `"A:45"`, ...) and failed deep inside the subprocess with no
useful message; this schema rejects a malformed entry before dispatch.
RFdiffusion's own training only ever showed the model 0-20% of a
binding site's true contacts as hotspots, so it expects to have to make
more contacts than you list -- 3-6 hotspots is the upstream-recommended
range. An empty list is accepted (no hotspot conditioning at all, a
legitimate but not recommended choice -- the binder can then attach
anywhere the contig geometry allows).

## Partial diffusion (`partial_t`, `provide_seq`)
Set `partial_t` to noise an EXISTING structure (rather than starting from
pure noise) and let RFdiffusion denoise it back out -- useful for
diversifying a binder you already have while mostly preserving its fold.
`contig` must then describe a region the exact same length as the input
chain being diversified. `provide_seq` additionally fixes the sequence
identity of chosen residue ranges within the diffused region while their
geometry still moves (e.g. keeping a helical peptide's sequence while
letting its backbone relax) -- it requires `partial_t` to be set, and
silently needs a different checkpoint, which this tool selects
automatically (matching the upstream script's own behaviour).

## `ckpt_variant`
`"auto"` (default) matches RFdiffusion's own automatic selection:
`Complex_base_ckpt.pt` for a binder design with a target, or the
InpaintSeq checkpoint automatically when `provide_seq` is set. `"beta"`
forces `Complex_beta_ckpt.pt`, trained for greater topological diversity
(fewer all-helical binders) at the cost of being far less
experimentally validated -- try it, per upstream guidance, at your own
risk, and never combined with `provide_seq` (no beta+InpaintSeq
checkpoint exists).

## `diffusion_steps` has a hard floor of 15
CONFIRMED LIVE: `diffuser.T` below 15 raises
`AssertionError: With discrete time and T < 15, the schedule is badly
approximated` before any GPU work starts. 50 is RFdiffusion's own
default and the value its released checkpoints were trained/benchmarked
with; lower values trade quality for speed and are appropriate for a
fast sanity check, not a design you intend to keep.

## Important caveats
- **Backbone only.** The output sequence is a placeholder (poly-glycine
  over the diffused region) -- always follow with `run_mpnn`.
- **Two live-confirmed host-specific workarounds are baked into this
  tool's wrapper** (stale editable-install path; nvrtc/JIT GPU-fusion
  crash on this host's GPUs) -- state both when reporting results, since
  either silently failing would look like the run hanging rather than
  erroring.
- Output chain IDs are RFdiffusion's own convention (binder always A,
  target(s) follow), not necessarily `target_pdb`'s original chain IDs.
- Symmetric oligomer design (`inference.symmetry`) and scaffold-guided
  secondary-structure control are not exposed here -- neither was
  exercised in this tool's install/verification and this server does not
  invent flags it has not verified.

## What you get back
`num_structures`, `checkpoint_used` (the actual weights file loaded, read
back from the engine's own log line so `ckpt_variant="auto"`'s choice is
never a guess), `contig_used` (the contig string RFdiffusion itself
echoed back), and under `outputs` the paths to every `structures` PDB and
its paired `metadata_trb`.

## Parameters

| Parameter | Type | Required | Default | Constraints | Description |
|---|---|---|---|---|---|
| `target_pdb` | string | yes | `—` | pattern: `\.pdb$` | Path to the target structure (PDB format only -- RFdiffusion 1.1.0's own parser does not accept mmCIF). Crop it around the intended interface before diffusing against a large target; see the doc's note on O(N^2) runtime scaling. WHERE THIS COMES FROM -- The target you want a binder against -- your own structure file, or a path a previous step returned. |
| `contig` | string | yes | `—` | pattern: `^([A-Za-z]\d+(-\d+)?(/\d+)?\|/\d+\|\d+(-\d+)?)(\s+([A-Za-z]\d+(-\d+)?(/\d+)?\|/\d+\|\d+(-\d+)?))*$` | RFdiffusion 1.1.0's own contig grammar (see the doc's "contig" section for the full explanation and worked example). Space-separated tokens: a target segment like "B1-100", optionally suffixed with "/0" for a chain break ("B1-100/0"), a bare "/0", or a diffused-length range like "100-150" (or a fixed "100-100"). Do not wrap it in outer brackets -- this tool adds those itself. |
| `hotspot_res` | array | no | `[]` | — | Target residues the binder must contact, each EXACTLY "<ChainID><ResidueNumber>" (e.g. "A30") -- no ranges, no colons, no other separator. 3-6 entries is the upstream-recommended range (the model expects to make more contacts than you list -- see the doc). An empty list runs without hotspot conditioning (a real but not recommended choice; the binder can attach anywhere the contig geometry allows). |
| `num_designs` | integer | no | `10` | minimum: `1`<br>maximum: `1000` | Number of independent backbones to sample. Cost scales linearly. 10 is RFdiffusion's own default; drop it to 1 for a fast sanity check on a new contig/hotspot combination before committing to a larger batch. |
| `diffusion_steps` | integer | no | `50` | minimum: `15`<br>maximum: `200` | Number of denoising timesteps (`diffuser.T`). CONFIRMED LIVE floor of 15 (the engine itself asserts on anything lower -- see the doc). 50 is RFdiffusion's own default and what its released checkpoints were trained with; use the minimum only for a structural sanity check, not a design you intend to keep. |
| `partial_t` | integer | no | `—` | minimum: `1`<br>maximum: `199` | Enables partial diffusion: noise an existing structure for this many timesteps (out of diffusion_steps) instead of starting from pure noise, then denoise it back out -- diversifies a known structure while mostly preserving its fold. Requires contig to describe a region the SAME length as the corresponding input chain (see the doc). Omit (the default) for ordinary from-scratch generation. |
| `provide_seq` | array | no | `—` | — | Zero-indexed inclusive residue ranges within the diffused region whose SEQUENCE stays fixed while its geometry is still diffused (e.g. "100-119" fixes 20 residues' identities). Requires partial_t to be set; this tool raises a clear error otherwise rather than letting the engine fail deep in the subprocess. Omit (the default) to diffuse sequence and structure together as normal. |
| `noise_scale_ca` | number | no | `1.0` | minimum: `0.0`<br>maximum: `2.0` | Translational noise scale for the Cα denoiser. 1.0 (RFdiffusion's own default) matches training; lowering it (e.g. toward 0) reduces diversity and can improve designability of individual samples at the cost of sampling a narrower region of structure space. |
| `noise_scale_frame` | number | no | `1.0` | minimum: `0.0`<br>maximum: `2.0` | Rotational noise scale for the frame (orientation) denoiser. 1.0 (RFdiffusion's own default) matches training; same diversity/ designability trade-off as noise_scale_ca, applied to orientation rather than position. |
| `deterministic` | boolean | no | `False` | — | When true, seeds every design's sampling from its own design index (design 0 always samples identically to a prior design-0 run, etc.), making num_designs=1 runs exactly reproducible. False (RFdiffusion's own default) draws a fresh random seed each call. |
| `ckpt_variant` | string | no | `auto` | enum: `['auto', 'beta']` | "auto" matches RFdiffusion's own automatic checkpoint selection for a target+contig binder design (Complex_base_ckpt.pt, or the InpaintSeq checkpoint automatically when provide_seq is set). "beta" forces the Complex_beta_ckpt.pt model, which produces a wider range of non-helical topologies but is far less experimentally validated -- see the doc's "ckpt_variant" section. Not combinable with provide_seq. |
