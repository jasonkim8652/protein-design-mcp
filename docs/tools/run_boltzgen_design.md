# run_boltzgen_design

**Category:** binder_generation  
**Engine:** `boltzgen`  
**Environment:** `/home/jk661/miniforge3/envs/boltzgen`  
**GPU required:** yes

> This file is generated from `src/protein_design_mcp/manifests/run_boltzgen_design.yaml`. Edit the manifest, then run `python scripts/generate_tool_docs.py`.

## Summary

Generate a binder against a target with BoltzGen's all-atom diffusion model, MSA-free (BoltzGen never builds or accepts an alignment -- the target comes straight from a PDB/CIF you supply inside design_spec). This is the `design` step of BoltzGen's own pipeline, run in isolation: it samples backbone coordinates AND a sequence together, but the sequence it proposes is usually lower quality than what BoltzGen's own inverse-folding head produces afterward -- chain this tool's output into `run_boltzgen_inverse_fold` for the sequence BoltzGen's own default pipeline would actually keep.

## What this is
BoltzGen's `design` pipeline step (`boltzgen.task.predict.predict.Predict`
over `design.yaml`), run in isolation via
`boltzgen run <design_spec> --steps design`. An all-atom diffusion model
jointly samples 3D coordinates and a sequence for every residue
design_spec marks as designed, conditioned on the target structure(s) and
any fixed/templated residues design_spec also supplies. BoltzGen refolds
and scores internally with its OWN weights elsewhere in its pipeline (the
`folding`/`design_folding`/`analysis` steps) -- it does not call ESMFold,
and this step alone computes no confidence metric at all.

## What it is for
The first step of a binder-design campaign: propose backbone shape and an
initial sequence for a chain that binds a target you supply directly as a
structure file. MSA-free by construction -- there is no alignment
parameter here because BoltzGen's diffusion model never takes one.

## `--protocol` has no effect on this step, verified empirically
BoltzGen's `--protocol` preset (`protein-anything`, `peptide-anything`,
...) changes settings on the `analysis`, `filtering`, and (for
peptide/nanobody/antibody protocols) `inverse_folding` steps only --
confirmed live by running `boltzgen configure` with `--steps design` under
all six protocols and diffing the resolved `design.yaml`: byte-identical
in every case except the `--output` path itself. This tool therefore does
not expose `protocol` -- a parameter with no observable effect would be
actively misleading to document as a choice. Chain composition, target
identity, and which residues are designed come entirely from design_spec
itself; nothing about "what kind of binder this is" is inferred from a
protocol name.

## When to use this instead of the alternatives
- `run_rfdiffusion_binder` (not yet implemented) and
  `run_rfdiffusion2` (not yet implemented) and
  `run_rfdiffusion3_binder` (not yet implemented) and
  `run_genie3_binder` (not yet implemented) and
  `run_protpardelle` (not yet implemented)
  are the direct siblings -- backbone-only binder generators
  that need a separate sequence-design step afterward (`run_mpnn`) rather
  than BoltzGen's own joint all-atom sampling.
- This tool's own generated sequence is usually a starting point, not the
  keeper -- BoltzGen's own default pipeline immediately re-derives it with
  its inverse-folding head. Feed this tool's output `.cif` into
  `run_boltzgen_inverse_fold`, marking the designed
  chain for redesign, to get the sequence BoltzGen's own pipeline would
  actually keep.
- After that, refold the result with a structure predictor (e.g.
  `run_chai1` (not yet implemented), `run_boltz` (not yet implemented))
  and score the interface (`run_ipsae`) before trusting any candidate --
  this tool computes no confidence metric of its own.

## What you must supply
`design_spec`: a BoltzGen design specification YAML. At minimum, an
`entities` list with a `protein:` block for the chain to design (a fixed
sequence, or a residue-count range like `15..20` for BoltzGen to sample
the length too) and a `file:` block pointing at the target structure (PDB
or mmCIF), with `include:`/`chain:` selectors naming which chain(s) of
that file to use. Chain composition -- which chain is the target, which is
designed, whether more than one chain of each is present -- is entirely
design_spec's own decision; nothing here is inferred from a bare
structure's shape. See BoltzGen's own `example/vanilla_protein/` for the
full grammar (multi-entity mixes, binding-site hints, motifs,
`design:`/`not_design:` residue masks on a `file:` entity).

## Important caveats
- This tool runs no scoring of any kind -- not even an internal one. A
  design with a low design_ptm/design_iptm after refolding is expected and
  normal; that is what `run_boltzgen_inverse_fold`
  and `run_boltzgen_filter` are for.
- `--reuse` (BoltzGen's own flag to skip regenerating designs already on
  disk) is not exposed here: every call gets a fresh, empty scratch
  directory, so there is never anything to reuse.
- `num_designs` defaults to 10 here, not BoltzGen's own CLI default of
  10,000 -- that default is sized for an unattended production campaign,
  not a single interactive call. Raising it raises this call's own
  runtime roughly linearly (see `timeout_s`, fixed at 3600s for this
  tool -- a large `num_designs` will need more time than that budget
  gives it; the run is killed at the timeout with its scratch directory
  preserved for diagnosis, not silently truncated).

## What you get back
`designs`: one entry per generated `.cif`, each `{"id", "chains": [{
"chain_id", "sequence", "length"}, ...]}` for every chain in that file
(target chain(s) included, unchanged across designs -- a genuine
confirmation of what stayed fixed, not a guess). `num_designs`, and under
`outputs` the paths to every generated `.cif`.

## Parameters

| Parameter | Type | Required | Default | Constraints | Description |
|---|---|---|---|---|---|
| `design_spec` | string | yes | `—` | pattern: `\.(yaml\|yml)$` | BoltzGen design specification YAML: entities (designed chain(s), target structure(s), any binding-site/motif hints), entirely in BoltzGen's own spec grammar. Chain composition is defined here, never inferred by this tool. |
| `num_designs` | integer | no | `10` | minimum: `1`<br>maximum: `10000` | Total number of designs to sample. BoltzGen's own CLI default is 10,000 (sized for an unattended campaign that gets filtered down afterward); this tool defaults to 10, sized for one interactive call -- raise it for a real campaign, but runtime grows roughly linearly and this tool's timeout_s is a fixed 3600s (see the doc's caveats). |
| `diffusion_batch_size` | integer | no | `—` | minimum: `1`<br>maximum: `1000` | Number of diffusion samples generated per trunk run (a batching/speed knob, not a quality one). Left unset, BoltzGen picks 1 if num_designs < 100 else 10 -- its own default heuristic, kept as this tool's default too. All designs generated in the same batch share the same sampled length when design_spec gives a residue-count range, so a large batch size relative to num_designs undersamples the length range; leave this unset unless you specifically need to control that. |
| `design_checkpoints` | array | no | `['huggingface:boltzgen/boltzgen-1:boltzgen1_diverse.ckpt', 'huggingface:boltzgen/boltzgen-1:boltzgen1_adherence.ckpt']` | minItems: `1` | Which trained checkpoint(s) to sample from; with more than one, each gets an equal fraction of num_designs. BoltzGen's own defaults are both official checkpoints -- "diverse" favours broader exploration of backbone shapes, "adherence" favours matching design_spec's structural hints more tightly. Pass a single entry (either one, or a local path to a fine-tuned checkpoint) to sample from only that one. |
| `step_scale` | number | no | `—` | minimum: `0.1`<br>maximum: `5.0` | Fixed diffusion step size, overriding BoltzGen's own default four-phase schedule (which alternates 1.8/2.0). Left unset, the schedule is used, which is BoltzGen's own recommended default. Set this only if you have a specific reason to fix it -- there is no single value that dominates the schedule across the board. |
| `noise_scale` | number | no | `—` | minimum: `0.0`<br>maximum: `2.0` | Fixed diffusion noise scale, overriding BoltzGen's own default four-phase schedule (which alternates 0.95/0.88). Left unset, the schedule is used, which is BoltzGen's own recommended default. |
| `use_kernels` | string | no | `auto` | enum: `['auto', 'true', 'false']` | Whether to use BoltzGen's fused CUDA kernels. "auto" (BoltzGen's own default) enables them when the GPU's compute capability is >= 8.0 -- confirmed live on this host's GPUs (capability 8.9), kernels are used. "false" is slower but a useful fallback if a kernel-related error ever needs ruling out. |
| `moldir` | string | no | `huggingface:boltzgen/inference-data:mols.zip` | — | Path or huggingface:repo:file reference for BoltzGen's canonical molecule/CCD library, needed to resolve residue and ligand chemistry. BoltzGen's own default, already cached on this host. |
| `num_workers` | integer | no | `1` | minimum: `0`<br>maximum: `32` | DataLoader worker process count. Raise for a large num_designs if data loading (not GPU compute) is the bottleneck; 1 is a safe default for a single interactive call. |
