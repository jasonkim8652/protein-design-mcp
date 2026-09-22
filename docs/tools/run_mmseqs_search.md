# run_mmseqs_search

**Category:** msa  
**Engine:** `mmseqs`  
**Environment:** `mmseqs`  
**GPU required:** yes

> This file is generated from `src/protein_design_mcp/manifests/run_mmseqs_search.yaml`. Edit the manifest, then run `python scripts/generate_tool_docs.py`.

## Summary

Search the local MMseqs2 databases and build the alignment AlphaFold 3 and every other structure-prediction tool here needs, so a folding tool never builds its own. Emits an unpaired a3m (valid for every co-folding tool's plain `msa` input), a paired a3m (AlphaFold 3's cross-chain pairing input only), and a templates a3m. Every search knob MMseqs2 exposes -- which databases, sensitivity, e-value, coverage, identity, iterations, max hits -- is a parameter here, not a fixed default.

## What this is
A wrapper around the standalone `mmseqs` binary (`/usr/local/bin/mmseqs`)
against the local, pre-padded (`mmseqs makepaddedseqdb`) databases under
`/opt/alphafold3_data/mmseqs_db/mmseqs`. Runs `search` -> `result2msa` ->
`unpackdb` once per selected database (this is the exact sequence the
user's own working AlphaFold 3 + MMseqs2-GPU fork uses --
`~/projects/af3-mmseqs-gpu/src/alphafold3/data/tools/mmseqs.py` -- not a
reimplementation from the MMseqs2 manual).

## What it is for
This is the ONLY alignment-producing tool in this server (alongside
`run_colabfold_search` (not yet implemented), which searches a *different*
sequence universe -- UniRef30/envDB, not these databases -- and whose a3m
is NOT interchangeable with this one). No structure-prediction tool here
builds its own alignment: hiding that step would make model-to-model
comparison meaningless and would silently send novel design sequences to
whichever MSA source a folding tool defaulted to. Call this first, then
pass its output into the folding tool's `msa` parameter.

## Which output goes where
- `unpaired_a3m`: every planned consumer that takes one plain a3m --
  RoseTTAFold3, Promera, and the Chai-1/Boltz-2/Protenix/OpenFold3 family
  (none implemented in this server yet). Also AlphaFold 3's
  `unpairedMsa`.
- `paired_a3m`: AlphaFold 3's `pairedMsa` only. Meaningless on its own;
  AlphaFold 3's featurisation pairs rows across chains sharing UniProt
  taxonomy. For a single chain that will be folded as (or as part of) a
  prediction you intend to pair, call this once per chain and pass each
  chain's own `paired_a3m` -- pairing happens downstream, not here.
- `templates_a3m`: template-aware predictors. Raw mmseqs headers -- see
  the output's own description for the AlphaFold-3-parser caveat.

## Single-chain queries and the `pair` parameter
This tool always takes one sequence per call, so "pairing" here can only
ever mean "search UniProt so a LATER cross-chain step has this chain's
candidates" -- it never performs the pairing itself. Set `pair: false` for
a chain you already know is going into a monomer prediction, to skip the
~78GB UniProt search; leave it `true` (the default) whenever this chain
might end up in a multimer, including when you are not yet sure. Either
way `paired_a3m` is always written (see that output's own description).

## Important caveats
- MMseqs2 against these local databases, and `run_colabfold_search`
  (not yet implemented) against UniRef30/envDB via `colabfold_search`,
  search different sequence universes. A consumer built against one and
  handed the other's a3m usually returns WORSE results, not an error --
  state which producer a folding tool's `msa` parameter expects before
  mixing them.
- `sequence` is validated against the standard 20 amino acids plus the
  common ambiguity codes (X, B, Z, J, U, O), uppercase only, BEFORE
  mmseqs ever sees it. This was checked live: mmseqs' own `createdb`
  accepts digits, lowercase, spaces and `*` without complaint and encodes
  them anyway, which is worse than a clean rejection -- it would search
  silently on a malformed query rather than fail loudly.
- A search that finds nothing is a normal, valid result, not an error:
  confirmed live that `result2msa`/`unpackdb` against a padded database
  with zero passing hits still emit a single query-only a3m record. Check
  `unpaired_hit_count` / `paired_hit_count` / `template_hit_count` in the
  reply rather than inferring "no hits" from file size or content shape.

## What you must supply
`sequence` -- one protein chain's amino acid sequence.

## What you get back
`query_length`, `unpaired_hit_count`, `paired_hit_count`,
`template_hit_count`, `unpaired_databases_searched`, `pair_searched`,
`templates_searched`, `used_gpu`, `elapsed_seconds`, and under `outputs`
the paths to `unpaired_a3m`, `paired_a3m`, `templates_a3m` and
`search_summary` (this same data, as a file).

## Parameters

| Parameter | Type | Required | Default | Constraints | Description |
|---|---|---|---|---|---|
| `sequence` | string | yes | `—` | pattern: `^[ACDEFGHIKLMNPQRSTVWYXBZJUO]+$` | One protein chain's amino acid sequence, uppercase single-letter codes only (the standard 20 plus the ambiguity codes X/B/Z/J/U/O). Anything else -- digits, lowercase, whitespace, `*`, a `>` that would corrupt the FASTA record this tool writes internally -- is rejected here rather than silently accepted by mmseqs (verified live: mmseqs' own createdb does not reject them itself). |
| `unpaired_databases` | array | no | `['uniref90', 'mgnify', 'small_bfd']` | minItems: `1` | Which unpaired protein databases to search and merge into `unpaired_a3m`. uniref90: UniRef90 clusters, the primary broad-coverage database. mgnify: MGnify environmental metagenomic proteins, adds diversity a UniProt-derived database misses. small_bfd: a reduced BFD (Big Fantastic Database) subset, deep coverage for hard/orphan targets. Dropping a database speeds up the call and narrows the alignment; the default searches all three, matching the reference pipeline this tool is built from. |
| `pair` | boolean | no | `True` | — | Whether to search UniProt (no deduplication) for `paired_a3m`. True (the default) is the safe choice when you are not certain this chain will end up folded alone -- it costs one extra ~78GB search. Set to false only when you already know this chain is going into a monomer prediction, where a paired alignment is meaningless (see the tool doc for what "paired" means at this tool's single-sequence granularity). |
| `search_templates` | boolean | no | `True` | — | Whether to search pdb_seqres for `templates_a3m`. False skips the search and writes a query-only placeholder instead, for a caller whose folding tool ignores templates entirely. |
| `sensitivity` | number | no | `7.5` | minimum: `1.0`<br>maximum: `7.5` | MMseqs2 `-s`. Higher finds more remote homologues and costs more time; 1.0 is fast and shallow, 7.5 is the deep default this tool inherits from the reference pipeline and is recommended for remote homology detection. Applies to every search this call makes (unpaired, paired, templates). |
| `e_value` | number | no | `0.0001` | minimum: `0.0` | MMseqs2 `-e` for the unpaired and paired searches: only hits with an E-value below this are kept. Lower is more stringent (fewer, more confident hits); higher admits more distant, noisier matches. The reference pipeline's default (1e-4) is noticeably stricter than mmseqs' own built-in default (1e-3). Template search uses the separate `template_e_value` instead, since broad template discovery wants a much looser threshold than a confident MSA does. |
| `max_sequences` | integer | no | `5000` | minimum: `1`<br>maximum: `1000000` | MMseqs2 `--max-seqs`: the maximum number of hits the prefilter is allowed to pass through per database, before the e-value/coverage/ identity thresholds are even applied -- set it too low and a real hit can be dropped before those thresholds get a chance to keep it. 5000 matches the reference pipeline. Raising it finds more hits at the cost of a slower align step; lowering it speeds up a call where you only need a handful of confident sequences. |
| `coverage` | number | no | `0.0` | minimum: `0.0`<br>maximum: `1.0` | MMseqs2 `-c`: keep only hits covering at least this fraction of residues, as `coverage_mode` defines "covered". 0.0 (mmseqs' own default, kept here) applies no coverage filter at all, which is why the reference pipeline never had to set it explicitly. Raise it (e.g. 0.5-0.8) to exclude short, partial-domain hits from the alignment. |
| `coverage_mode` | integer | no | `0` | enum: `[0, 1, 2, 3, 4, 5]` | MMseqs2 `--cov-mode`, which residues `coverage` measures against: 0 query AND target coverage, 1 target coverage only, 2 query coverage only, 3 target must be at least `coverage` fraction of query length, 4 query must be at least `coverage` fraction of target length, 5 the shorter sequence must cover `coverage` of the longer. 0 is mmseqs' own default and only matters once `coverage` is raised above 0.0. |
| `min_seq_id` | number | no | `0.0` | minimum: `0.0`<br>maximum: `1.0` | MMseqs2 `--min-seq-id`: keep only hits at or above this fractional sequence identity to the query. 0.0 (mmseqs' own default, and the reference pipeline's, since it never set this) applies no identity filter. Raise it to force a shallower, higher-identity alignment -- useful when a distant/noisy hit is hurting a downstream predictor more than helping it. |
| `num_iterations` | integer | no | `1` | minimum: `1`<br>maximum: `10` | MMseqs2 `--num-iterations`: how many rounds of profile-based search to run (PSI-BLAST-style). 1 (mmseqs' own default, and the reference pipeline's) is a single-pass sequence search. Raising it builds a profile from round N's hits and searches again, finding more remote homologues at a real cost in time -- each extra round is close to another full search. |
| `threads` | integer | no | `8` | minimum: `1`<br>maximum: `64` | MMseqs2 `--threads`: CPU threads for the parts of the search that are not GPU-accelerated (index loading, alignment, result2msa, unpackdb). 8 matches the reference pipeline. Raising it speeds up those stages on a host with spare cores; it does not change the alignment's content. |
| `use_gpu` | boolean | no | `True` | — | Whether to pass MMseqs2 `--gpu 1` for the search stage. True (the default, matching the reference pipeline) is the only practical choice against these particular databases: they are GPU-padded (`makepaddedseqdb`), and a CPU-mode search against even the smallest of them was confirmed live to still be running after 2+ minutes and over 80GB resident memory, against 43 seconds on GPU. Exposed anyway because it is a genuine mmseqs knob and this tool's policy is never to hide one behind a default that "usually works" -- but leaving it false here is not expected to finish in practice. |
| `template_e_value` | number | no | `100.0` | minimum: `0.0` | E-value threshold for the templates search only (separate from `e_value`, which governs the unpaired/paired searches). 100.0 matches the reference pipeline's choice to mirror hmmsearch's own broad, permissive template discovery -- structural templates are useful even from a distant hit, unlike MSA rows used for co-evolutionary signal. |
| `max_template_hits` | integer | no | `1000` | minimum: `1`<br>maximum: `100000` | MMseqs2 `--max-seqs` for the templates search only (separate from `max_sequences`). 1000 matches the reference pipeline. |
