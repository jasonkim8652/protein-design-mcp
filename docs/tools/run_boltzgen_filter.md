# run_boltzgen_filter

**Category:** run_analysis  
**Engine:** `boltzgen`  
**Environment:** `None`  
**GPU required:** no

> This file is generated from `src/protein_design_mcp/manifests/run_boltzgen_filter.yaml`. Edit the manifest, then run `python scripts/generate_tool_docs.py`.

## Summary

Re-rank a finished BoltzGen run with new thresholds, for free, without regenerating anything (BoltzGen's own `filtering` pipeline step). This tool runs no model at all -- it is pure dataframe filtering and ranking over the numeric columns BoltzGen's own predict/score steps already wrote to design_dir/aggregate_metrics_*.csv. Change a threshold and call again as many times as you like; each call costs seconds, not GPU time.

## What this is
BoltzGen's `filtering` pipeline step (`boltzgen.task.filter.filter.Filter`),
run in isolation via `boltzgen run <design_spec> --steps filtering`. It reads
`design_dir/aggregate_metrics_*.csv` (written earlier by BoltzGen's own
`analysis` step, over structures its `folding`/`design_folding` steps
produced) and does three things, in order: (1) apply hard pass/fail
thresholds, (2) rank the survivors by a composite of the metrics you
choose, weighted by how much each should matter, (3) run a lazy-greedy
diversity selection so the final set is not just the top-ranked cluster of
near-duplicate sequences.

**This step runs no model.** There is no `boltz` package import anywhere
in it and no GPU use -- it is dataframe arithmetic (pandas/numpy) over
numbers a GPU step already produced elsewhere. Do not describe it as
scoring, predicting, or running Boltz-2; it is ranking.

## What it is for
Getting the ranking right without paying for regeneration. Every threshold
and every ranking weight here can be swept -- run once with the defaults,
look at `results_overview_pdf`, loosen `refolding_rmsd_threshold` or
reweight `metrics_override`, and call again. Each call is CPU-only and
takes on the order of seconds to tens of seconds (verified live: 12.1s
over a completed 1-design analysis directory).

## What you must supply
- `design_spec`: the SAME design specification YAML the original run used.
  `boltzgen run` validates every design spec it is given before running
  any step, `--steps filtering` included -- this file's content has no
  effect on the filtering step's own numbers, but it must still parse.
- `design_dir`: the directory BoltzGen's own `analysis` step wrote into --
  it must contain `aggregate_metrics_*.csv` and `ca_coords_sequences.pkl.gz`,
  and (unless you disable the filters that need them) `refold_cif/` and
  `refold_design_cif/`. This is a read-only input to this tool; nothing is
  ever written back into it -- every output goes to this tool's own scratch
  directory instead (`outdir`/`--output` are always this run's workdir, not
  `design_dir`, and not the original run's `--output` directory either).

## Ranking direction -- read before setting metrics_override
Every metric this tool ranks by is defined so that **higher is better**
after this tool's own sign convention is applied. `min_interaction_pae`
and `min_design_to_target_pae` are themselves lower-is-better (PAE in
Angstroms), so BoltzGen negates them internally
(`neg_min_design_to_target_pae`, `neg_min_interaction_pae`) before ranking
-- use the `neg_` name in `metrics_override`, not the raw PAE column, or
you will be ranking backwards. `filter_rmsd`/`refolding_rmsd_threshold`
and the composition-fraction filters (`ALA_fraction` etc.) are genuinely
lower-is-better and are applied as upper-bound thresholds, not ranked.

## When to use this instead of the alternatives
- If you have not yet generated or inverse-folded any designs, this tool
  has nothing to rank -- use `run_boltzgen_design` and
  `run_boltzgen_inverse_fold` (not yet implemented), and BoltzGen's own
  `folding`/`design_folding`/`analysis` steps, which this server does not
  expose as separate tools, first.
- `run_ipsae` and `run_prodigy` score ONE structure at a time from a
  predictor's raw output; this tool ranks a whole finished BATCH using
  metrics BoltzGen's own analysis step already computed, and adds
  diversity selection on top -- they solve different problems.

## What you get back
`selected_designs` (one entry per row of the quality+diversity CSV: `id`,
`final_rank`, `designed_sequence`, `designed_chain_sequence`, `num_design`,
`design_to_target_iptm`, `design_ptm`, `min_design_to_target_pae`),
`num_selected`, `num_total_designs`, `num_passing_all_filters`, and under
`outputs` the paths to the full metrics CSVs, the selected structures, and
the summary PDF.

## Parameters

| Parameter | Type | Required | Default | Constraints | Description |
|---|---|---|---|---|---|
| `design_spec` | string | yes | `—` | pattern: `\.(yaml\|yml)$` | The design specification YAML the original run used. Its content does not change this step's numbers (protocol/entities only matter to the design/inverse_folding/folding/analysis steps, none of which run here) -- it is required because `boltzgen run` validates every design spec it is given before running any step at all, `--steps filtering` included. Pass the same file you used (or would have used) to produce design_dir. |
| `design_dir` | string | yes | `—` | — | Directory BoltzGen's own `analysis` step wrote into: must contain `aggregate_metrics_*.csv` and `ca_coords_sequences.pkl.gz`, and (for the default filter/metric settings below) `refold_cif/` and `refold_design_cif/`. Read-only -- nothing is ever written back here. |
| `budget` | integer | no | `30` | minimum: `1`<br>maximum: `100000` | How many designs land in the final quality+diversity selected set (`selected_designs`, `final_*_designs/`, `final_designs_metrics_*.csv`). BoltzGen's own default for a full pipeline run is also 30; lower it for a quick look, raise it to keep more candidates for downstream scoring. |
| `top_budget` | integer | no | `10` | minimum: `1`<br>maximum: `100000` | How many of the highest-QUALITY-ranked designs (before diversity selection) are copied to `intermediate_ranked_<top_budget>_designs/` and used as the "Top" comparison group in results_overview_pdf's tables/plots. Purely quality-ranked, unlike `budget`'s set which also optimizes for sequence diversity. |
| `alpha` | number | no | `0.001` | minimum: `0.0`<br>maximum: `1.0` | Trade-off for the final diversity selection: 0.0 = pick purely by quality rank (ties `budget`'s set to `top_budget`'s), 1.0 = pick purely to maximize sequence dissimilarity and ignore quality entirely. 0.001 is BoltzGen's own `protein-anything` default (almost pure quality); its own `peptide-anything` protocol preset instead defaults this to 0.01 (more diversity-weighted, since peptide campaigns usually want a varied set to synthesize) -- raise this toward 0.01-0.05 for a peptide/binder library rather than a single best candidate. |
| `refolding_rmsd_threshold` | number | no | `2.5` | minimum: `0.0`<br>maximum: `50.0` | Backbone RMSD (Angstrom) ceiling for the refolding-consistency filters (`filter_rmsd`, `filter_rmsd_design`, and -- when `filter_designfolding` is true -- `designfolding-filter_rmsd`): a design must refold within this RMSD of its generated backbone to pass. Lower is stricter. 2.5 is BoltzGen's `protein-anything` default; its `peptide-anything` protocol preset instead defaults this to 2 (peptides are small enough that 2.5A is comparatively loose). |
| `filter_biased` | boolean | no | `True` | — | Reject designs whose amino-acid composition is an outlier: more than 30% of any of ALA/GLY/GLU/LEU/VAL. These residues are cheap for a diffusion model to over-use and rarely make a good real sequence. true is BoltzGen's own default in every protocol. |
| `filter_cysteine` | boolean | no | `False` | — | Reject any design containing a cysteine BoltzGen itself designed (a pre-existing cysteine already in the input is not counted). false (allow cysteines) is BoltzGen's `protein-anything` default; its `peptide-anything`, `nanobody-anything` and `antibody-anything` protocol presets all default this to true instead, since an unpaired free cysteine is a liability for those modalities (aggregation, disulfide scrambling) more than for a bulk protein binder. Set true if design_dir's designs were peptide/nanobody/ antibody-typed. |
| `filter_designfolding` | boolean | no | `True` | — | Also require the design refolded ALONE (target absent) to match its generated shape (`designfolding-filter_rmsd`), on top of the with-target refolding check every run already applies. true is BoltzGen's default when its `design_folding` step ran (its `protein-anything`/`protein-small_molecule` protocols run it by default; `peptide-anything`/`nanobody-anything`/`antibody-anything` do not). Set this to false if design_dir has no `refold_design_cif/` directory -- true against missing data raises a KeyError from the underlying engine, not a clean filtered-out result. |
| `from_inverse_folded` | boolean | no | `True` | — | True if design_dir's designs went through BoltzGen's own inverse-folding head (this server's `run_boltzgen_inverse_fold`, or the `inverse_folding` step of a full `boltzgen run`) -- the normal case, and the only one this server's own tools produce. Ranks and filters on backbone-only RMSD (`bb_rmsd*`) and the refolded ΔSASA column. Set false only if design_dir's designs skipped inverse folding entirely, which switches ranking to all-atom RMSD and the ΔSASA computed on the original (non-refolded) structure instead. |
| `filter_bindingsite` | boolean | no | `False` | — | Reject any design with no residue near a binding-site residue named in design_spec (`bindsite_under_8rmsd > 0`). false is BoltzGen's own default; set true only if design_spec actually names binding-site residues to check against -- against a spec with none, this filter rejects every design. |
| `filter_target_aligned` | boolean | no | `False` | — | Require the TARGET chain's backbone (not the design's) to still align within 2.5A after refolding -- catches designs that distort the target itself, not just their own shape. false is BoltzGen's own default in every protocol. |
| `modality` | string | no | `peptide` | enum: `['peptide', 'antibody']` | Which liability-scoring and sequence-visualization rules apply (deamidation/oxidation/protease-site heuristics differ between a short peptide and an antibody CDR). Only affects `num_liability_plots`/`plot_seq_logos` output, never the pass/fail filters or the ranking itself. "peptide" is BoltzGen's own default and the right choice for a bulk protein binder too (there is no third "protein" option in the underlying engine). |
| `peptide_type` | string | no | `linear` | enum: `['linear', 'cyclic']` | Whether liability scoring treats the designed chain as a linear or head-to-tail cyclic peptide. Only matters when modality is "peptide" and num_liability_plots > 0. "linear" is BoltzGen's own default. |
| `metrics_override` | object | no | `—` | — | Per-metric ranking weight, as {"<metric column>": <inverse-importance weight> \| null}. A LARGER weight DOWN-WEIGHTS that metric's rank (rank is divided by the weight); set a metric to null to drop it from the ranking entirely; name a metric not in the default set to add it. The default ranking set (weight in parens) is design_to_target_iptm (1), design_ptm (1), neg_min_design_to_target_pae (1), plip_hbonds_refolded (2), plip_saltbridge_refolded (2), delta_sasa_refolded (2) -- all of these are already oriented so higher is better (see "Ranking direction" above; use the `neg_` column for anything that is natively lower-is-better). Example: {"design_ptm": 2, "plip_hbonds_refolded": null} halves design_ptm's influence and drops hydrogen-bond count from the ranking entirely. |
| `additional_filters` | array | no | `—` | — | Extra hard pass/fail thresholds beyond the built-in ones, as a list of {"feature": "<numeric column in aggregate_metrics>", "threshold": <number>, "lower_is_better": <bool>}. A design fails if lower_is_better is true and its value exceeds threshold, or lower_is_better is false and its value is below threshold. Example: {"feature": "design_ptm", "threshold": 0.7, "lower_is_better": false} keeps only designs with design_ptm >= 0.7. Empty (no extra filters) by default. |
| `size_buckets` | array | no | `—` | — | Optional cap on how many of the diversity-selected `budget` designs may fall in each length range, as a list of {"min": <int>, "max": <int>, "num_designs": <int>} (a design of length L lands in the first bucket where min <= L < max). Use this to force the selected set to span several lengths instead of clustering at whichever length scored best. Empty (no size constraint) by default. |
| `random_state` | integer | no | `0` | minimum: `0` | Random seed for the lazy-greedy diversity selection. Fixed by default so repeated calls with identical inputs select the identical diversity set. |
| `num_liability_plots` | integer | no | `0` | minimum: `0`<br>maximum: `1000` | How many of the top-quality designs get a per-residue developability liability heat-map (deamidation, oxidation, protease sites, ...) in results_overview_pdf. 0 (skip -- these plots are the slowest part of report generation) is BoltzGen's own default. |
| `plot_seq_logos` | boolean | no | `False` | — | Whether results_overview_pdf includes sequence-logo and amino-acid- composition plots for the All/Top/Diverse sets. false is BoltzGen's own default; these add noticeable PDF-generation time on a large batch. |
| `use_affinity` | boolean | no | `False` | — | Switch the ranking metric set to one built around `affinity_probability_binary1` instead of the default interface metrics -- only meaningful if design_dir's designs went through BoltzGen's own `affinity` step (small-molecule-binder campaigns, BoltzGen's `protein-small_molecule` protocol). false is BoltzGen's default for every other protocol; true against a design_dir with no affinity column ranks on a column that does not exist. |
