# run_alphafold2_multimer

**Category:** structure_prediction  
**Engine:** `colabfold`  
**Environment:** `/home/jk661/.conda/envs/colabfold`  
**GPU required:** yes

> This file is generated from `src/protein_design_mcp/manifests/run_alphafold2_multimer.yaml`. Edit the manifest, then run `python scripts/generate_tool_docs.py`.

## Summary

Co-fold one or more protein chains with AlphaFold2-Multimer via ColabFold (`colabfold_batch`, multimer_v3 weights). Only single_sequence (MSA-free) or a caller-supplied a3m are permitted here -- ColabFold's own mmseqs2_* modes query a public remote server, and the sequences this server predicts are usually novel designs that must not leave the machine; this restriction is enforced in the wrapper, not only in this doc.

## What this is
AlphaFold2-Multimer, run through ColabFold's `colabfold_batch` CLI with
the `alphafold2_multimer_v3` model (all 5 parameter sets cached on this
host). Given a set of protein chains it predicts one joint structure and
reports AlphaFold's own confidence (pLDDT, pTM, ipTM, PAE).

## `msa` -- remote search is never permitted, enforced in code
ColabFold's `--msa-mode` has four values:
`mmseqs2_uniref_env`/`mmseqs2_uniref_env_envpair`/`mmseqs2_uniref`
(all three query the public `api.colabfold.com` MSA server) and
`single_sequence` (no search at all). Only the latter is reachable
through this tool. `msa: null` runs `colabfold_batch --msa-mode
single_sequence` (deliberately MSA-free). A path instead makes the
wrapper pass that a3m FILE directly as `colabfold_batch`'s input in
place of a FASTA -- confirmed from `colabfold_batch --help`: "Using an
A3M file as input overwrites this option" -- so `--msa-mode` is never
even passed in that case. There is no third option and no parameter that
can select one of the `mmseqs2_*` modes: this tool has no `msa_mode`
parameter at all, so a caller cannot ask this tool to reach the remote
server no matter what value it sends.

## What alignment `msa` must be
For a single chain in `sequences`, a plain monomer a3m (e.g.
run_mmseqs_search's `unpaired_a3m` for that sequence). For MULTIPLE
chains, ColabFold expects its own PAIRED multimer a3m shape (per-species
row pairing across chains, its own header convention) -- a caller
supplying an arbitrary per-chain a3m for a multi-chain `sequences` list
must already have it in that exact shape; `run_mmseqs_search`'s plain
per-chain `unpaired_a3m` is NOT pre-paired and is not a drop-in here.
`run_colabfold_search` (not yet implemented) is the intended producer of
a correctly paired multimer a3m for this tool.

## `sequences` -- chain composition is explicit, never inferred
A one-entry list predicts a monomer (with the multimer_v3 model, which
works fine on a single chain); several entries co-fold a complex, joined
into one ColabFold FASTA record with `:` between chains -- exactly
ColabFold's own multimer input convention, read from `colabfold_batch
--help` and its README.

## When to use this instead of the alternatives
- `run_promera` and `run_rf3` also co-fold with an explicit `msa`, using
  different model families (Promera's own iCS/ipSAE heads; RF3's
  AF3-shaped confidence). This tool is the one to reach for when you
  specifically want AlphaFold2-Multimer's own weights.
- `model_type` is fixed to `alphafold2_multimer_v3` and is not exposed as
  a parameter: this tool's identity IS multimer prediction (its name
  says so), and ColabFold's monomer/`_ptm` model variants would overlap
  with `run_esmfold2` and any future dedicated monomer AF2 tool rather
  than add a real choice here.
- Once you have a structure, `run_ipsae` on `best_model_pdb` +
  `pae_json` gives the field-standard interface confidence; this tool's
  own `iptm` is ColabFold's built-in number for the same prediction, not
  an independent check.

## What you must supply
`sequences` and `msa` (`null` or an a3m path, never omitted).

## What you get back
ColabFold's own top-ranked-model scores JSON content (typically
`plddt`, `ptm`, `iptm`, `max_pae` -- ColabFold's own field set, passed
through as-is). Under `outputs`, paths to `best_model_pdb`, `scores_json`
and `pae_json`.

## Important caveats
- `--templates` (PDB template search) is never enabled: ColabFold's own
  `--help` states it "can result in the MSA server being queried with
  A3M input" even when `--msa-mode single_sequence` is set -- the exact
  leak this tool exists to prevent. Not exposed, hardcoded off.
- `--amber`/`--num-relax` (ColabFold's own OpenMM relaxation step) are
  not exposed: `run_openmm_minimize` is this server's dedicated tool for
  that step, and exposing a second path to it here would duplicate a
  step this server already registers on its own.
- `--data`/`--host-url` (weights directory / MSA server URL) are not
  exposed: both are deployment paths, not modeling choices, and
  `--host-url` in particular is exactly the kind of parameter that could
  otherwise redirect the remote-server restriction above to a
  caller-chosen endpoint.
- ColabFold's own subsampling (`--max-seq`/`--max-extra-seq`/`--max-msa`)
  and `--recycle-early-stop-tolerance` are not exposed: each only takes
  effect on an actual multi-row alignment, which this tool does not
  build itself, and none carries a documented default in `--help` this
  tool could state as "a choice with a reason" instead of a bare `None`.

## Parameters

| Parameter | Type | Required | Default | Constraints | Description |
|---|---|---|---|---|---|
| `sequences` | array | yes | `—` | minItems: `1` | One or more protein chains' amino acid sequences, in the order they should be joined (`:`-separated) into one ColabFold multimer query. A single entry predicts a monomer; several co-fold a complex -- chain composition is never inferred. |
| `msa` | — | yes | `—` | pattern: `\.a3m$` | null runs `sequences` MSA-free, deliberately, via ColabFold's own `--msa-mode single_sequence` (the only search-free mode). A path to an a3m file instead is passed directly as ColabFold's input in place of a FASTA, which ColabFold uses as the alignment as-is ("Using an A3M file as input overwrites [--msa-mode]" -- ColabFold's own --help). See the doc for what shape that a3m must be for a multi-chain `sequences` list. There is no third option: ColabFold's remote-search modes (mmseqs2_uniref_env and friends) are never reachable through this tool. |
| `num_recycle` | integer | no | `3` | minimum: `0`<br>maximum: `20` | Number of prediction recycles. Higher can improve quality at roughly linear cost; 3 matches `model.num_recycle`'s own default in this build's AlphaFold2 model config (`alphafold/model/config.py`). ColabFold's own CLI default is unset (None, meaning "use the model config's own value") -- 3 is that resolved value, not a value invented for this tool. |
| `num_models` | integer | no | `5` | enum: `[1, 2, 3, 4, 5]` | How many of the 5 cached multimer_v3 model parameter sets to run and rank between. Fewer is faster and lower quality; 5 (ColabFold's own CLI default) runs every cached model. |
| `num_seeds` | integer | no | `1` | minimum: `1`<br>maximum: `10` | How many random seeds to try per model (iterates from `random_seed` to `random_seed + num_seeds`), each producing its own prediction. 1 is ColabFold's own CLI default; raise it to check prediction stability across seeds, at proportional cost. |
| `random_seed` | integer | no | `0` | minimum: `0` | Starting seed for `num_seeds`. 0 is ColabFold's own CLI default. |
| `num_ensemble` | integer | no | `1` | minimum: `1`<br>maximum: `8` | Number of times the trunk is run per recycle with different random MSA cluster-center choices, then averaged -- can improve quality at roughly linear cost. 1 is ColabFold's own CLI default (AlphaFold's original CASP14 setting used up to 8 for the monomer pipeline). |
| `pair_mode` | string | no | `unpaired_paired` | enum: `['unpaired', 'paired', 'unpaired_paired']` | How multimer MSA rows are combined for cross-chain co-evolutionary signal: unpaired only, paired only, or both. unpaired_paired is ColabFold's own CLI default and the one to use unless you have a specific reason not to; it only has an effect when `msa` supplies an alignment for more than one chain. |
| `pair_strategy` | string | no | `greedy` | enum: `['complete', 'greedy']` | How MSA rows are paired across chains: complete requires the same species in every chain's alignment; greedy pairs whenever the same species appears in at least two. greedy (ColabFold's own CLI default) typically produces more paired rows and better predictions; complete can help when the alignments are already large and well-covered. Only matters when `msa` supplies a multi-chain alignment. |
| `use_dropout` | boolean | no | `False` | — | Activate dropout during inference to sample from the model's own uncertainty -- produces different predictions run to run, useful for (carefully) sampling alternate conformations. false (ColabFold's own CLI default) gives the deterministic, best-estimate prediction. |
| `rank` | string | no | `auto` | enum: `['auto', 'plddt', 'ptm', 'iptm', 'multimer']` | Which metric selects `best_model_pdb`/`scores_json` (rank_001) among `num_models` predictions. auto (ColabFold's own CLI default) picks pLDDT for a monomer and a multimer-appropriate metric for a complex; pin it to one metric explicitly if you need a specific ranking criterion across a batch of calls. |
| `stop_at_score` | number | no | `100.0` | minimum: `0.0`<br>maximum: `100.0` | Stop computing further models once the ranking score (pLDDT for a monomer, pTM for a multimer) exceeds this threshold -- speeds up an easy query by running fewer of `num_models`. 100.0 (ColabFold's own CLI default) effectively disables early stopping, since no prediction exceeds it. |
