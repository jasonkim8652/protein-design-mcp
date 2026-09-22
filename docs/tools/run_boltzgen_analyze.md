# run_boltzgen_analyze

**Category:** run_analysis  
**Engine:** `boltzgen`  
**Environment:** `/home/jk661/miniforge3/envs/boltzgen`  
**GPU required:** no

> This file is generated from `src/protein_design_mcp/manifests/run_boltzgen_analyze.yaml`. Edit the manifest, then run `python scripts/generate_tool_docs.py`.

## Summary

Compute CPU metrics over a design and aggregate them into the ranking table `run_boltzgen_filter` reads -- BoltzGen's own `analysis` pipeline step, run in isolation. This step runs no model: no GPU use, no `boltz` import, pure geometry/dataframe computation (ΔSASA, backbone RMSD, non-covalent contacts, hydrophobic patches, liability scoring) over structures `run_boltzgen_fold`/`run_boltzgen_design_fold` already produced. Feed it the WHOLE outputs of run_boltzgen_design (or run_boltzgen_inverse_fold) and run_boltzgen_fold -- every file, not just the structures.

## What this is
BoltzGen's `analysis` pipeline step (`boltzgen.task.analyze.analyze.Analyze`),
run in isolation via `boltzgen run <design_spec> --steps analysis`. It
aggregates metrics from the Folding, Designfolding, and (if requested)
Affinity predictions into a single per-design CSV, computing several more
metrics itself on CPU along the way: ΔSASA (buried surface area),
backbone/all-atom refolding RMSD, non-covalent contact counts, largest
hydrophobic patch, and sequence liability scores (deamidation, oxidation,
protease sites). This is the step that produces
`design_iptm`/`min_interaction_pae`/`delta_sasa_refolded`/`bb_rmsd` and
every other column `run_boltzgen_filter` ranks or thresholds on -- without
it, filter has nothing to read.

**This step runs no model.** No GPU use, no `boltz` import -- CPU-only
dataframe and structure-geometry computation over numbers
`run_boltzgen_fold`/`run_boltzgen_design_fold` already produced elsewhere.

## `--protocol` is not exposed, for the same reason it is not exposed on
`run_boltzgen_filter`: every setting a protocol preset would otherwise
control here (`affinity_metrics`, `largest_hydrophobic`,
`largest_hydrophobic_refolded`, `designfolding_metrics`,
`use_design_mask_for_target`) is instead its own explicit parameter
below, always passed via `--config analysis`, which -- as with
`run_boltzgen_filter` -- always wins over whatever a protocol preset
would otherwise set (BoltzGen's own CLI applies protocol presets AFTER
its computed top-level values in the same argv concatenation).

## When to use this instead of the alternatives
- This is the ONLY tool in this server that produces
  `aggregate_metrics_*.csv` -- `run_boltzgen_filter` cannot run without
  having called this first (or an equivalent full `boltzgen run`, done
  outside this server).
- `run_ipsae`/`run_prodigy` score ONE structure at a time from a raw
  predictor output; this tool aggregates a whole batch's worth of
  BoltzGen-specific metrics (ΔSASA, liability scores, refolding RMSD)
  that those tools do not compute at all.
- `run_proteina_complexa_analyze` is the equivalent step for a DIFFERENT
  engine (Proteina-Complexa) -- do not mix a Proteina-Complexa run's
  outputs into this tool, the file formats are not compatible.

## What you must supply
- `design_spec`: the SAME design specification YAML the designs came
  from. Required only because `boltzgen run` validates every design spec
  it is given before running any step at all; its content does not
  change this step's own numbers.
- `generated_files`: the WHOLE `outputs.generated_designs` (from
  `run_boltzgen_design`) or `outputs.inverse_folded_designs` (from
  `run_boltzgen_inverse_fold`) list.
- `refold_structures` + `refold_metrics`: `run_boltzgen_fold`'s OWN
  `refolded_structures` and `refold_metrics` outputs, passed through
  unfiltered.
- `design_refold_structures` + `design_refold_metrics` (optional):
  `run_boltzgen_design_fold`'s own outputs, if you also ran that tool --
  required together with `designfolding_metrics: true` to get
  `designfolding-*` columns; leave both unset otherwise.

## What you get back
`num_designs_analyzed`, and under `outputs` the paths to
`aggregate_metrics_csv` (the ranking table), `per_target_metrics_csv`,
and `sequence_cache` -- pass `aggregate_metrics_csv` and `sequence_cache`
straight through as `run_boltzgen_filter`'s `metrics_files`. This tool
does not itself rank or select anything -- that is `run_boltzgen_filter`'s
job, over the table this tool produces.

## Parameters

| Parameter | Type | Required | Default | Constraints | Description |
|---|---|---|---|---|---|
| `design_spec` | string | yes | `—` | pattern: `\.(yaml\|yml)$` | The design specification YAML the designs came from. Its content has no effect on this step's own numbers -- required only because `boltzgen run` validates every design spec it is given before running any step, `--steps analysis` included. |
| `generated_files` | array | yes | `—` | minItems: `2` | The WHOLE `outputs.generated_designs` (from run_boltzgen_design) or `outputs.inverse_folded_designs` (from run_boltzgen_inverse_fold) list -- every `.cif` AND `.npz` path that call returned, unfiltered. |
| `refold_structures` | array | yes | `—` | minItems: `1` | `run_boltzgen_fold`'s own `refolded_structures` output, passed through unfiltered. |
| `refold_metrics` | array | yes | `—` | minItems: `1` | `run_boltzgen_fold`'s own `refold_metrics` output, passed through unfiltered -- required to compute `design_iptm`, `min_interaction_pae`, and every other confidence-based column. |
| `design_refold_structures` | array | no | `—` | minItems: `1` | `run_boltzgen_design_fold`'s own `refolded_structures` output, if you also ran that tool. Supply together with `design_refold_metrics` and set `designfolding_metrics: true` to get `designfolding-*` columns; leave unset otherwise. |
| `design_refold_metrics` | array | no | `—` | minItems: `1` | `run_boltzgen_design_fold`'s own `refold_metrics` output, if you also ran that tool. Supply together with `design_refold_structures`. |
| `affinity_metrics` | boolean | no | `False` | — | Compute affinity-related columns. Requires an `affinity` step's output (small-molecule-binder campaigns) which no tool in this server produces -- BoltzGen's affinity head is protein-ligand only and excluded here the same way Boltz-2's is. Leave false. |
| `backbone_fold_metrics` | boolean | no | `True` | — | Compute backbone-only refolding RMSD metrics (`bb_rmsd*`, `bb_designability_rmsd_*`). true is BoltzGen's own default for a normal (inverse-folded) run. |
| `allatom_fold_metrics` | boolean | no | `True` | — | Compute all-atom refolding RMSD metrics (`rmsd`, `rmsd_design`), in addition to (or instead of, if the design was not inverse-folded) the backbone-only ones. true is BoltzGen's own default. |
| `noncovalents_original` | boolean | no | `True` | — | Count non-covalent interface contacts (hydrogen bonds, salt bridges, hydrophobic contacts -- `plip_hbonds`, `plip_saltbridge`, `plip_hydrophobic`) on the ORIGINAL (pre-refold) structure. true is BoltzGen's own default. |
| `noncovalents_refolded` | boolean | no | `True` | — | Same as `noncovalents_original` but on the REFOLDED structure (`plip_hbonds_refolded` etc. -- the columns `run_boltzgen_filter` ranks on by default). true is BoltzGen's own default. |
| `delta_sasa_original` | boolean | no | `True` | — | Compute ΔSASA (buried surface area: binder present vs absent) on the ORIGINAL (pre-refold) structure (`delta_sasa_original`). true is BoltzGen's own default. |
| `delta_sasa_refolded` | boolean | no | `True` | — | Same as `delta_sasa_original` but on the REFOLDED structure (`delta_sasa_refolded` -- the column `run_boltzgen_filter` ranks on by default). true is BoltzGen's own default. |
| `largest_hydrophobic` | boolean | no | `False` | — | Compute the largest contiguous surface hydrophobic patch on the ORIGINAL (pre-refold) structure. false is BoltzGen's own `protein-anything` default (its peptide/nanobody/antibody protocol presets also default this false). |
| `largest_hydrophobic_refolded` | boolean | no | `True` | — | Same as `largest_hydrophobic` but on the REFOLDED structure. true is BoltzGen's `protein-anything` default; its peptide/nanobody/antibody protocol presets default this false instead (a hydrophobic-patch penalty matters less for a short peptide/CDR loop than a bulk protein binder) -- set false yourself if the designs are peptide/nanobody/antibody-typed. |
| `run_clustering` | boolean | no | `False` | — | Compute foldseek-based structural clustering/diversity metrics. Requires a local foldseek binary and database, not confirmed present in every deployment of this server -- false (BoltzGen's own default) unless you have confirmed foldseek is installed; true without it fails with a clear "binary not found" error rather than a silent wrong answer. |
| `liability_analysis` | boolean | no | `True` | — | Score each designed sequence for developability liabilities (deamidation, oxidation, protease cleavage sites, N-terminal cyclization). true is BoltzGen's own default. |
| `liability_modality` | string | no | `peptide` | enum: `['peptide', 'antibody']` | Which liability-scoring rules apply (deamidation/oxidation/protease- site heuristics differ between a short peptide and an antibody CDR). "peptide" is BoltzGen's own default for this step and the right choice for a bulk protein binder too (there is no third "protein" option in the underlying engine). |
| `liability_peptide_type` | string | no | `linear` | enum: `['linear', 'cyclic']` | Whether liability scoring treats the designed chain as linear or head-to-tail cyclic. Only matters when `liability_modality` is "peptide". "linear" is BoltzGen's own default. |
| `designfolding_metrics` | boolean | no | `False` | — | Compute the `designfolding-*`-prefixed columns (the design-alone self-consistency comparison). Requires `design_refold_structures` and `design_refold_metrics` to also be supplied (from `run_boltzgen_design_fold`) -- true without them raises a FileNotFoundError from the underlying engine, not a clean skip. |
| `use_design_mask_for_target` | boolean | no | `False` | — | Use the design mask (rather than the chain-design mask) to decide what counts as "target" for target-relative metrics -- BoltzGen's own `protein-redesign` protocol default, for redesigning/optimizing an existing protein where every chain may carry designed residues. false (BoltzGen's default otherwise) is correct for a normal binder-design campaign where the target is a separate, entirely fixed chain. |
| `compute_lddts` | boolean | no | `False` | — | Compute per-residue lDDT scores against the refolded structure. Adds noticeable CPU time (BoltzGen's own docs: ~5-15s per design). false is BoltzGen's own default when run through its CLI (its Analyze class's own Python default is true, but the CLI's analysis.yaml config overrides it to false). |
| `num_processes` | integer | no | `32` | minimum: `1`<br>maximum: `128` | Number of worker processes for the CPU metric computation (`ProcessPoolExecutor`). 32 is BoltzGen's own default; lower it on a host with fewer cores available, or if you are running several of this server's tools concurrently and want to leave CPU headroom. |
| `moldir` | string | no | `huggingface:boltzgen/inference-data:mols.zip` | — | Path or huggingface:repo:file reference for BoltzGen's canonical molecule/CCD library, needed to resolve residue and ligand chemistry. BoltzGen's own default, already cached on this host. |
| `num_workers` | integer | no | `4` | minimum: `0`<br>maximum: `32` | DataLoader worker process count for reading designs back in (a different pool from `num_processes`, which does the metric computation itself). 4 is BoltzGen's own default for this step. |
