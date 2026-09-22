# run_colabfold_search

**Category:** msa  
**Engine:** `colabfold`  
**Environment:** `/home/jk661/.conda/envs/colabfold`  
**GPU required:** no

> This file is generated from `src/protein_design_mcp/manifests/run_colabfold_search.yaml`. Edit the manifest, then run `python scripts/generate_tool_docs.py`.

## Summary

Search ColabFold's own local databases (UniRef30, envDB) with `colabfold_search`, the second MSA producer this server exposes -- MMseqs2 against a DIFFERENT sequence universe than run_mmseqs_search (which searches AlphaFold 3's own database set). Local search only: this tool never reaches ColabFold's remote MSA server under any parameter value, by construction -- see the doc's "Local search only" section. NOT VERIFIED LIVE on this host -- read "Verification status" before relying on it.

## What this is
A wrapper around the standalone `colabfold_search` CLI (env `colabfold`),
which runs MMseqs2 against ColabFold's own local database set --
`uniref30_2302_db` (UniRef30) and, by default, `colabfold_envdb_202108_db`
(envDB, metagenomic diversity) -- and writes one merged, deduplicated a3m
per query.

## Verification status -- read this before using this tool
`colabfold_search` itself IS installed and confirmed present in the
`colabfold` environment (`--help` inspected live, 2026-09-22; its full
flag set is what this tool's schema below is built from). What is NOT
present anywhere on this host (confirmed by an exhaustive search across
every real mounted filesystem, 2026-09-22) is the local database set
`colabfold_search` needs to search against -- `uniref30_2302_db`,
`colabfold_envdb_202108_db`, and the rest of ColabFold's own
`setup_databases.sh` output. `/home/jk661/.cache/colabfold_dbs` (this
tool's fixed `dbbase`) is currently an EMPTY placeholder directory this
wave created only so the manifest's mount-existence check (every
`engine.mounts` entry must exist on this host, checked at manifest-load
time) would pass at all -- it holds no database files. Calling this tool
right now will fail loudly with an mmseqs "database not found"-shaped
error, not silently return an empty or wrong alignment. An operator needs
to run ColabFold's own `setup_databases.sh` (or equivalent) against that
directory before this tool can complete a real search. This is the same
class of finding as `run_alphafold3`'s missing Docker image and
`run_rosetta_interface`'s incomplete wheel -- an asset gap this wave
could document but not fix.

The `results/query.a3m` output filename WAS verified against
`colabfold/mmseqs/search.py`'s own source, read directly out of this
host's `colabfold` environment (not from memory or the upstream docs) --
see that output's own description for the exact trace.

## Local search only -- enforced in the wrapper, not just stated here
`colabfold_search`'s own CLI (unlike the higher-level `colabfold_batch`
tool and its underlying remote-MSA helper function in `colabfold.batch`,
which default to `https://api.colabfold.com`) has NO remote-server flag
at all -- confirmed
live, 2026-09-22, its full `--help` output has no `--host-url` or
equivalent. This wrapper only ever shells out to the `colabfold_search`
binary directly (never imports `colabfold.batch` or any function that
defaults to the remote API), and `dbbase` is a fixed constant this
tool's schema never exposes as a parameter -- there is no argument value
a caller can pass that reaches a network address. The sequences handled
here are frequently novel designs, and this tool's whole point is to keep
them on the machine.

## When to use this instead of the alternatives
`run_mmseqs_search` is the default choice -- it is the alignment every
other tool in this server was built and verified against. Prefer THIS
tool only when a specific downstream consumer documents that it wants
ColabFold's own alignment shape instead.

MMseqs2 against AlphaFold 3's database set (`run_mmseqs_search`) and
MMseqs2 against ColabFold's own database set (this tool) search DIFFERENT
sequence universes and apply different filtering defaults -- their a3m
outputs are NOT interchangeable. A consumer built for one handed the
other's alignment usually returns WORSE results, not an error. State
which producer a co-folding tool's `msa` parameter expects before mixing
them; when unstated, prefer `run_mmseqs_search`'s output, since that is
the alignment this server's other tools were built and verified against.

## Single-sequence queries -- pairing never activates, so it is not exposed
Exactly like `run_mmseqs_search`, this tool takes ONE sequence per call.
Read live out of `colabfold/mmseqs/search.py`: `colabfold_search`'s
cross-chain pairing step only runs when the input FASTA is a
"`:`-joined" MULTI-chain record (`is_complex is True`) -- for a
single-sequence query it is skipped unconditionally, regardless of
`--pair-mode`/`--pairing_strategy`/`--use-env-pairing`. Exposing those as
parameters here would be knobs with no observable effect -- the same rule
`run_proteina_complexa_analyze`'s doc already applies to Foldseek's
hardcoded thresholds. This tool always runs with pairing effectively off.

## Not exposed
- `--pair-mode`, `--pairing_strategy`, `--use-env-pairing`, `--db4`: see
  "Single-sequence queries" above -- no observable effect at this tool's
  granularity.
- `--use-templates`/`--db2` (structural templates): `run_mmseqs_search`
  already provides templates via its own, independently-verified
  `search_templates`/`templates_a3m` path. Exposing `colabfold_search`'s
  OWN template feature would need a real templates database present to
  verify its output naming/format (the `.m8` rename logic in
  `colabfold/mmseqs/search.py`) -- like the search databases themselves,
  that is not available on this host to confirm, so this tool does not
  guess at it.
- `--af3-json`/`--af3-msa-as-path`: this server's structure-prediction
  tools build their own AlphaFold 3 JSON from a plain a3m via their own
  schemas (see `run_alphafold3`) rather than consuming colabfold_search's
  own convenience JSON format -- consistent with `run_mmseqs_search` never
  emitting one either.
- `--unpack`: pinned to `1` (unpacked loose a3m files) -- this tool's
  declared `a3m` output and its adapter both depend on that shape; `0`
  would leave results as an MMseqs2 database this tool cannot read.

## What you must supply
`sequence` -- one protein chain's amino acid sequence.

## What you get back
`query_length` and, under `outputs`, the path to `a3m`.

## Parameters

| Parameter | Type | Required | Default | Constraints | Description |
|---|---|---|---|---|---|
| `sequence` | string | yes | `—` | pattern: `^[ACDEFGHIKLMNPQRSTVWYXBZJUO]+$` | One protein chain's amino acid sequence, uppercase single-letter codes only (the standard 20 plus the ambiguity codes X/B/Z/J/U/O) -- same character policy as run_mmseqs_search's `sequence`, enforced here for the same reason (colabfold_search's own FASTA parsing does not reject a malformed sequence itself). |
| `db1` | string | no | `uniref30_2302_db` | pattern: `^[A-Za-z0-9_.]+$` | Which UniRef database (colabfold_search's `--db1`) to search, by its filename prefix inside the fixed local database directory. uniref30_2302_db is colabfold_search's own default and the standard ColabFold UniRef30 release; change only if a different UniRef30/40 build is installed under a different name in that directory. |
| `use_env` | boolean | no | `True` | — | Whether to also search the environmental (metagenomic) database (`--use-env`, controls `--db3`). True is colabfold_search's own default -- envDB adds sequence diversity a UniProt-derived database like UniRef30 misses, at real extra search cost. |
| `db3` | string | no | `colabfold_envdb_202108_db` | pattern: `^[A-Za-z0-9_.]+$` | Which environmental database (colabfold_search's `--db3`) to search when use_env is true; ignored otherwise. colabfold_envdb_202108_db is colabfold_search's own default and the standard ColabFold envDB release. |
| `prefilter_mode` | integer | no | `0` | enum: `[0, 1, 2]` | MMseqs2 prefiltering algorithm colabfold_search uses: 0 k-mer (high-memory, colabfold_search's own default), 1 ungapped (high-CPU), 2 exhaustive (no prefilter at all -- very slow, only for a query short/important enough that missing a hit is unacceptable). |
| `sensitivity` | number | no | `None` | minimum: `1.0`<br>maximum: `9.0` | MMseqs2 `-s`. No default (null) is colabfold_search's OWN default -- when omitted, it derives an internal k-mer threshold corresponding to roughly sensitivity 8 (matching the ColabFold web server), rather than a fixed `-s` value. Set explicitly (e.g. lower for a faster, shallower search) to override that derived threshold. |
| `filter` | integer | no | `1` | enum: `[0, 1, 2]` | MMseqs2 result filtering: 0 none, 1 filter the unpaired MSA by expand_eval/align_eval/diff/qsc (colabfold_search's own default), 2 also filter the paired MSA (irrelevant here -- see "Not exposed", no pairing runs for a single-sequence query). |
| `expand_eval` | number | no | `inf` | minimum: `0.0` | E-value threshold for MMseqs2's `expandaln` step. Infinity (no filtering at this step) is colabfold_search's own default; lower it to prune expansion hits before the align step. |
| `align_eval` | integer | no | `10` | minimum: `0` | E-value threshold for MMseqs2's `align` step. Verified live in colabfold_search's own argparse (2026-09-22) that this is an INTEGER flag (unlike expand_eval/qsc, which are floats) -- 10 is colabfold_search's own default; lower it for a more stringent, more confident alignment. |
| `diff` | integer | no | `3000` | minimum: `1` | Minimum number of sequences filterresult keeps in each MSA block, even after applying qsc/align_eval. 3000 is colabfold_search's own default. Raising it keeps a deeper alignment at the cost of noisier distant hits; lowering it prunes toward only the highest-scoring rows. |
| `qsc` | number | no | `-20.0` | — | filterresult's minimum query-score threshold used to reduce output MSA diversity. -20.0 (colabfold_search's own default) is permissive -- effectively no filtering by this criterion. Raise it (toward 0) to keep only higher-scoring rows. |
| `max_accept` | integer | no | `1000000` | minimum: `1` | Maximum accepted alignments per query before MMseqs2's align step stops. 1000000 is colabfold_search's own default (effectively unbounded for any query this tool would see); lower it only to cap align-step cost on a query expected to have an unusually large number of hits. |
| `db_load_mode` | integer | no | `0` | enum: `[0, 1, 2, 3]` | How MMseqs2 loads the database into memory: 0 auto (colabfold_search's own default), 1 fread, 2 mmap, 3 mmap+touch (pre-fault pages). Only matters for repeated-call performance on this host's storage; does not change the alignment's content. |
| `threads` | integer | no | `64` | minimum: `1`<br>maximum: `128` | CPU threads for MMseqs2's non-GPU stages. 64 is colabfold_search's own default. Raising it speeds up those stages on a host with spare cores; it does not change the alignment's content. |
| `use_gpu` | boolean | no | `False` | — | Whether to pass colabfold_search `--gpu 1` for the search stage. False is colabfold_search's own default (unlike run_mmseqs_search's databases, ColabFold's own database set is not necessarily GPU-padded, so CPU mode is colabfold_search's traditional, long- supported default -- this tool has no live finding of its own about relative CPU/GPU speed here, since the databases are not present to test against; see "Verification status"). Set true only if this deployment's colabfold databases were built GPU-padded. |
| `gpu_server` | boolean | no | `False` | — | Whether to pass colabfold_search `--gpu-server 1`, which reuses a persistent GPU-resident MMseqs2 server process across calls instead of reloading the database each time. False is colabfold_search's own default; only useful when use_gpu is also true and many calls are expected in sequence. |
