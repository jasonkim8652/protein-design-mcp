# run_proteina_complexa_filter

**Category:** run_analysis  
**Engine:** `proteinfoundation`  
**Environment:** `/home/jk661/.conda/envs/proteina_complexa`  
**GPU required:** no

> This file is generated from `src/protein_design_mcp/manifests/run_proteina_complexa_filter.yaml`. Edit the manifest, then run `python scripts/generate_tool_docs.py`.

## Summary

Re-rank a run_proteina_complexa_generate result with a new reward threshold and/or top-N cutoff, for free, without regenerating anything (Proteina-Complexa's own `filter` pipeline step). This tool runs no model -- it is pandas dedup/threshold/sort over the rewards_csv a run_proteina_complexa_generate call already wrote. Change a threshold and call again as many times as you like; each call costs seconds.

## What this is
Proteina-Complexa's `filter` pipeline step (`complexa filter`, wrapping
`proteinfoundation.filter`), run in isolation against ONE input: the
`rewards_csv` a prior run_proteina_complexa_generate call returned under
its own `outputs`. `filter.py` itself does three things, in order: drop
rows with a missing `total_reward`, optionally deduplicate by exact
sequence (`dedup_sequence`), then sort by `total_reward` descending and
keep the top `filter_samples_limit` rows (after an optional
`reward_threshold` floor). No model runs and no GPU is used -- verified
by reading `filter.py`: its only `torch` use is a CUDA-availability
assertion inside a `setup()` helper this tool's adapter deliberately
bypasses by supplying `root_path` directly (see below), so the assertion
never runs.

## What you must supply
`rewards_csv`: the exact file a run_proteina_complexa_generate call
returned under `outputs.rewards_csv` (or any CSV in the same shape -- a
`total_reward` column plus one row per candidate). Any other CSV works
as long as it has that column; you do not need to have generated it with
this server's own generate tool.

## Sample directories are not reconstructed, and delete_non_top_n_samples
is therefore not exposed
`filter.py`'s own logic can additionally DELETE or MOVE the per-sample
PDB subdirectories under its `root_path` that did not make the cut. This
tool never stages the generated structures themselves (only the rewards
CSV) -- so `root_path` for this tool never contains any sample
directories at all, and `delete_non_top_n_samples` would have nothing to
act on regardless of its value. It is therefore not exposed, the same
"no knob with no effect" rule as `run_boltzgen_design`'s `--protocol`.
The CSV outputs (`top_samples_csv`, `all_rewards_csv`) this tool returns
are computed purely from the DataFrame and are unaffected by this.

## When to use this instead of the alternatives
- `run_boltzgen_filter` is the direct sibling in this category: it
  performs the same "re-rank a finished batch for free" job but for
  BoltzGen's own `aggregate_metrics_*.csv`, with quality+diversity
  lazy-greedy selection rather than this tool's simple
  threshold-then-top-N. Use whichever tool matches the engine that
  generated your batch -- they read incompatible CSV shapes.
- `run_proteina_complexa_analyze` is a different, later step (structural/
  sequence diversity over a design SET) -- run this tool first if you
  have not yet cut your batch down to a manageable size, since analyze's
  diversity clustering cost scales with how many designs you feed it.
- `run_ipsae`/`run_prodigy` score ONE structure at a time from a
  predictor's own output; this tool ranks a whole batch by the reward
  Proteina-Complexa's OWN generation-time reward model already computed
  -- different signals, and this tool is far cheaper since it runs no
  model at all.

## What you get back
`selected_designs`: one entry per row of `top_samples_csv` (`pdb_path`,
`total_reward`, and every other rewards_csv column preserved).
`num_selected`, `num_total_designs` (rows in `all_rewards_csv`, i.e.
after NaN-drop and optional dedup, before the top-N cut), and under
`outputs` the paths to both CSVs.

## Parameters

| Parameter | Type | Required | Default | Constraints | Description |
|---|---|---|---|---|---|
| `rewards_csv` | string | yes | `—` | pattern: `\.csv$` | The rewards CSV a prior run_proteina_complexa_generate call returned under outputs.rewards_csv (or any CSV with a total_reward column and one row per candidate). |
| `filter_samples_limit` | integer | no | `1000` | minimum: `1`<br>maximum: `100000` | Maximum designs kept after filtering, taken as the top this-many by total_reward (after the optional reward_threshold floor and optional dedup_sequence). 1000 is this engine's own config default. |
| `dedup_sequence` | boolean | no | `True` | — | Drop rows whose sequence (the CSV's aatype column) exactly matches an earlier row before ranking. true is this engine's own config default. |
| `reward_threshold` | number | no | `—` | — | Drop rows with total_reward below this value before the top-N cut. Unset (this engine's own default) keeps every row regardless of reward. Since this tool costs seconds per call, sweep this rather than guessing a single value. |
