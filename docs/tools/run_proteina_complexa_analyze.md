# run_proteina_complexa_analyze

**Category:** run_analysis  
**Engine:** `proteinfoundation`  
**Environment:** `/home/jk661/.conda/envs/proteina_complexa`  
**GPU required:** no

> This file is generated from `src/protein_design_mcp/manifests/run_proteina_complexa_analyze.yaml`. Edit the manifest, then run `python scripts/generate_tool_docs.py`.

## Summary

Compute structural (Foldseek) and sequence (MMseqs2) diversity over a set of designs -- Proteina-Complexa's `analyze` pipeline step's own unique value, since it runs no neural model at all. Because Proteina-Complexa's `evaluate` step (which would normally write the CSV `analyze` reads) is deliberately excluded from this server, this tool takes its designs as structured parameters and materialises that CSV itself. Diversity needs only each design's structure and sequence; refolding metrics (designability, codesignability, interface hydrogen bonds) are optional extras that unlock additional analysis when you already have them from elsewhere.

## How this tool works
Proteina-Complexa's `evaluate` step (excluded from this server by
design -- it bundles refolding, interface analysis and force-field
metrics that are all exposed individually elsewhere, and hides a
"binder is the last chain" convention) is what would normally write
`binder_results_{config}_{job}.csv` for the real `analyze` step to read.
With `evaluate` gone, nothing does -- so this tool's adapter builds that
CSV itself from `structure_paths`/`sequences` and whichever optional
metric arrays you supply, then runs the real `complexa analyze` over it.
You are never running a hand-rolled reimplementation of the analysis --
every number in the response comes from Proteina-Complexa's own
`compute_foldseek_diversity`/`compute_mmseqs_diversity`
(`result_analysis/compute_diversity.py`) and threshold/pass-rate code
(`analyze.py`), just fed a CSV this adapter assembled instead of one
`evaluate` wrote.

## Verified: diversity needs only structures, not refolding metrics
Read live before building this tool (2026-09-22), tracing
`compute_foldseek_diversity`/`compute_mmseqs_diversity` end to end:
both take a DataFrame, `groupby_cols` and a few numeric knobs, and touch
exactly TWO things from the DataFrame -- `pdb_path` (grouped into lists
of structure paths) and whatever `groupby_cols` names. Neither function
reads a sequence column at all: `diversity_sequence_mmseqs` extracts each
sequence itself, from the structure file, via `extract_seq_from_pdb`.
`analyze.py`'s own driver confirms the same thing one layer up:
`merge_monomer_into_binder` (which pulls in designability/codesignability
columns from a separate monomer-results file when one exists) explicitly
returns the ORIGINAL, unmodified DataFrame when no such file is found --
logged, not raised -- and the diversity call over the FULL design set
(`metric_suffix="all_generated"`) runs on that same unmodified `df`
inside its own `try/except`, independent of the success-threshold/
pass-rate code that DOES need the refolding columns. A materialised CSV
carrying only `pdb_path` plus grouping columns is therefore not a
degraded input to this pipeline -- it is precisely what the diversity
computation was always going to use. `structure_paths` is this tool's
only genuinely required input; every metric below is optional and adds
analysis on top rather than being needed for diversity to run at all.

This finding is from reading Proteina-Complexa's own source, not from a
completed live run: `complexa analyze` itself could not be executed on
this host (see "Not verified live" below), so this is exactly what
WAVE-COMMON asks be said plainly rather than implied by a passing test.

## What you must supply
`structure_paths`: one structure file (PDB or mmCIF) per design, at least
1. `sequences`: one sequence string per design, in the SAME order --
matched to `structure_paths` by index, and must be the same length as
it. Sequences are carried through for the materialised CSV's own record
(and are available in `combined_results_csv`); the diversity computation
itself derives sequence straight from each structure file regardless
(see above), so a mismatch between a supplied sequence and what is
actually in the structure does not affect the diversity numbers, only
the record.

## Optional refolding/interface metrics -- name the tool that produces each
Supply any of these as a PARALLEL array (same length and order as
`structure_paths`) to unlock the corresponding analysis; omit an array
entirely to skip just that analysis, with everything else unaffected.
Partial coverage (a value for only SOME designs) is not supported -- an
array, when supplied, must cover every design.
- `designability_scrmsd_ca`: CA backbone RMSD (Angstrom) after
  redesigning the sequence and refolding it, compared to the generated
  structure. Produce this by redesigning with `run_mpnn`, refolding the
  redesigned sequence with `run_esmfold2` (or another
  `structure_prediction` tool), and computing backbone RMSD between that
  refolded structure and the original.
- `codesignability_scrmsd_ca` / `codesignability_scrmsd_all_atom`: CA / all-atom RMSD
  after refolding the design's OWN generated sequence (no redesign) and
  comparing to the generated structure. Produce this by refolding this
  design's sequence directly with `run_esmfold2` (or another
  `structure_prediction` tool) and computing RMSD against the original.
- `interface_hbonds_tmol`: hydrogen-bond count across the binder-target
  interface, from Proteina's OWN tmol force field
  (`rewards/tmol_reward.py`). This is NOT reproducible with this
  server's generic scoring tools -- only a Proteina-Complexa
  `run_proteina_complexa_generate` call with the (not currently exposed,
  see that tool's doc) TMOL reward model enabled can supply it. Omit
  this if your designs did not come from such a run.

These map onto Proteina-Complexa's own internal column names as
`_res_scRMSD_ca_external`, `_res_co_scRMSD_ca_external`,
`_res_co_scRMSD_all_atom_external` and
`generated_n_interface_hbonds_tmol` respectively -- "external" is this
adapter's fixed stand-in for a folding-model name (Proteina-Complexa's
own convention keys these by which folding model produced them; since
the refold happened OUTSIDE this engine, "external" is used uniformly).
If you set `designability_thresholds`, `ca_codesignability_thresholds` or
`allatom_codesignability_thresholds` below, key your threshold dict on
`"external"` to match -- a threshold keyed on e.g. `"esmfold"` will
silently match nothing.

## Diversity parameters
`compute_foldseek_diversity`/`compute_mmseqs_diversity` toggle each
computation independently; `mmseqs_min_seq_id`/`mmseqs_coverage` tune
MMseqs2 clustering (verified wired to `aggregation.mmseqs_min_seq_id`/
`aggregation.mmseqs_coverage` in `analyze.py`). Foldseek's own
`min_seq_id`/`alignment_type` are NOT exposed: reading `analyze.py`'s
binder-analysis driver shows both are hardcoded literals
(`min_seq_id=0.0, alignment_type=1`) at the one call site this tool's
`result_type=protein_binder` path reaches, not read from `aggregation.*`
at all -- exposing them would be a knob with no observable effect, the
same "no knob with no effect" rule as `run_boltzgen_design`'s
`--protocol`.

## When to use this instead of the alternatives
- If you have not generated anything yet, this tool has nothing to
  analyze -- use `run_proteina_complexa_generate` first.
- `run_ipsae`/`run_prodigy` score ONE structure's interface confidence/
  energy; this tool measures diversity across a whole SET and optionally
  aggregates pass rates over thresholds you choose -- different
  questions.

## Not verified live
`complexa analyze` could not be executed on this host during this
tool's development: the `proteina` conda environment is missing several
of Proteina-Complexa's own declared dependencies (`lightning`,
`pydantic`, `typing_extensions`, `fsspec` among others -- `pip check`
run live, 2026-09-22, lists the full set), which blocks every
`complexa` step (`generate`/`filter`/`analyze` alike), not something
specific to this tool. See the wave report for the exact diagnostic.

## What you get back
`foldseek_diversity`/`mmseqs_diversity`: `{"score", "num_clusters",
"num_samples"}` for the full design set (Foldseek's diversity score is
cluster_count / sample_count; higher means less redundant). `num_designs`,
and under `outputs` the paths to both diversity CSVs (which also include
any success-threshold-filtered subsets, when the relevant optional
metrics were supplied) and the full combined results CSV.

## Parameters

| Parameter | Type | Required | Default | Constraints | Description |
|---|---|---|---|---|---|
| `structure_paths` | array | yes | `—` | minItems: `1` | One structure file (PDB or mmCIF) per design. Must be ABSOLUTE paths -- unlike a single format:path parameter, an array of paths is not auto-resolved against the server's working directory (see app._resolve_path_params), so a relative path here resolves against whatever the server process's own cwd happens to be, not against any directory meaningful to you. Paths returned by another tool's own outputs (e.g. run_proteina_complexa_generate's outputs.generated_structures) are already absolute and safe to pass straight through. Two entries sharing an exact basename overwrite each other during staging (last one wins) -- use distinctly-named files. |
| `sequences` | array | yes | `—` | minItems: `1` | One sequence per design, in the SAME order as structure_paths -- must be the same length as structure_paths. Carried through into combined_results_csv for the record; the diversity computation itself derives sequence directly from each structure file regardless (see the doc's "Verified" section), so this does not need to be re-derived precisely from the structure to be useful. |
| `designability_scrmsd_ca` | array | no | `—` | — | CA backbone RMSD (Angstrom) per design, in the SAME order as structure_paths, after redesigning the sequence with run_mpnn and refolding it with run_esmfold2 (or another structure_prediction tool). Omit to skip designability pass-rate analysis. When supplied, must cover every design (no partial coverage). |
| `codesignability_scrmsd_ca` | array | no | `—` | — | CA backbone RMSD (Angstrom) per design, in the SAME order as structure_paths, after refolding the design's OWN generated sequence (no redesign) with run_esmfold2 (or another structure_prediction tool) and comparing to the generated structure. When supplied, must cover every design. |
| `codesignability_scrmsd_all_atom` | array | no | `—` | — | All-atom RMSD (Angstrom) per design, same procedure as codesignability_scrmsd_ca but comparing every atom rather than just CA. When supplied, must cover every design. |
| `interface_hbonds_tmol` | array | no | `—` | — | Binder-target interface hydrogen-bond count per design, from Proteina's own tmol force field. NOT reproducible with this server's generic scoring tools -- only available from a run_proteina_complexa_generate call with the TMOL reward model enabled. Omit if unavailable. When supplied, must cover every design. |
| `result_type` | string | no | `protein_binder` | enum: `['protein_binder', 'ligand_binder']` | Whether these designs are protein-protein binders (default -- what run_proteina_complexa_generate produces today) or small-molecule ("ligand") binders. Selects which default success thresholds and column conventions analyze.py applies. |
| `compute_foldseek_diversity` | boolean | no | `True` | — | Whether to run Foldseek structural-diversity clustering at all. true is this engine's own config default. |
| `compute_mmseqs_diversity` | boolean | no | `True` | — | Whether to run MMseqs2 sequence-diversity clustering at all. true is this engine's own config default. |
| `mmseqs_min_seq_id` | number | no | `0.1` | minimum: `0.0`<br>maximum: `1.0` | MMseqs2 minimum sequence identity for two designs to cluster together. Higher values require closer matches to cluster, so diversity (cluster_count / sample_count) rises. 0.1 is this engine's own config default. |
| `mmseqs_coverage` | number | no | `0.7` | minimum: `0.0`<br>maximum: `1.0` | MMseqs2 minimum alignment coverage for two designs to cluster together. 0.7 is this engine's own config default. |
| `analysis_modes` | array | no | `['binder', 'monomer']` | minItems: `1` | Which analysis families to run. "binder" covers diversity, interface pass rates and success filtering; "monomer" covers designability/ codesignability pass rates over the optional scRMSD arrays above. ["binder", "monomer"] is this engine's own default for protein_binder/ligand_binder results. "monomer" analysis produces no extra numbers (but does not fail) if none of the optional scRMSD arrays were supplied. |
| `success_thresholds` | object | no | `—` | — | Custom interface success-threshold spec overriding this engine's own default ({"i_pAE": {"threshold": 7.0, "op": "<=", "scale": 31.0, "column_prefix": "complex"}, "pLDDT": {"threshold": 0.9, "op": ">=", "scale": 1.0, "column_prefix": "complex"}, "scRMSD": {"threshold": 1.5, "op": "<", "scale": 1.0, "column_prefix": "binder"}}) -- these thresholds only match columns this tool's fixed reward/refolding pipeline actually produces; since this tool does not currently accept raw i_pAE/pLDDT columns (only the scRMSD-style optional metrics above), setting this without also having matching data has no effect. See this checkout's own configs/analyze.yaml for the full nested shape. |
| `designability_thresholds` | object | no | `—` | — | Custom threshold spec for designability_scrmsd_ca pass-rate computation, as {"ca": {"external": {"threshold": <Angstrom>, "op": "<="}}} -- must key on "external" (see the doc's naming section), not a real folding-model name. Unset (this engine's own default) uses a 2.0 Angstrom threshold. Only has any effect when designability_scrmsd_ca was supplied. |
| `ca_codesignability_thresholds` | object | no | `—` | — | Same shape as designability_thresholds, for codesignability_scrmsd_ca. Only has any effect when codesignability_scrmsd_ca was supplied. |
| `allatom_codesignability_thresholds` | object | no | `—` | — | Same shape as designability_thresholds but keyed "all_atom", for codesignability_scrmsd_all_atom. Only has any effect when codesignability_scrmsd_all_atom was supplied. |
| `require_all_thresholds` | boolean | no | `False` | — | Whether a design must pass EVERY configured designability/ codesignability threshold (true, AND logic) or just one of them (false, OR logic, this engine's own default) to count as a pass. |
