# run_alphafold3

**Category:** structure_prediction  
**Engine:** `alphafold3`  
**Environment:** `/alphafold3_venv`  
**GPU required:** yes

> This file is generated from `src/protein_design_mcp/manifests/run_alphafold3.yaml`. Edit the manifest, then run `python scripts/generate_tool_docs.py`.

## Summary

Predict a multi-chain protein structure with AlphaFold 3, given an explicit alignment (or none) per chain. The heaviest tool in this server: dispatches through a mounted environment (EngineSpec.prefix) exactly like every other GPU engine here, extracted from AF3's own image rather than built by conda -- see the doc's "How this tool is dispatched" section for the one detail (a relocated mount) that differs from the rest. MSA is optional and NEVER built by this tool (AF3's own alignment-search step is always disabled) -- pass run_mmseqs_search's unpaired_a3m/paired_a3m outputs, or null/null to run MSA-free. Protein chains only in this version -- no RNA, DNA, ligands, covalent bonds, or templates; see "Not exposed".

## What this is
AlphaFold 3, the current state-of-the-art structure predictor, covering
the same "chains + optional per-chain MSA" job shape as
`run_boltz`/`run_chai1`/`run_protenix`/`run_openfold3`, but from a
genuinely different model. The engine itself -- its venv and its
inference entrypoint script -- was extracted from
`romerolabduke/alphafast:latest` (the RomeroLab MMseqs2-GPU fork) onto
this host with `docker create` + `docker cp` (no sibling container ever
runs; see "How this tool is dispatched" for why, and why that changed
from an earlier `docker run` design). Ground truth for the invocation
SHAPE (GPU pinning, bind-mount pattern, in-venv activation) is the user's
own checkout, `~/projects/af3-mmseqs-gpu` (the reference inference script
under its `benchmarks/` directory) -- but NOT for which entrypoint script
executes: see "Which entrypoint script executes" below for why this tool
uses the image's own baked-in copy instead of the one that script mounts
from the host repo. `docs/input.md` for the AlphaFold 3 JSON schema still
applies either way (shared, package-level code, not
entrypoint-script-specific).

## Which entrypoint script executes -- confirmed live, not assumed
The ground-truth benchmark script bind-mounts the host repo's own
inference entrypoint into the container. **This tool does NOT do that**
-- CONFIRMED LIVE, 2026-09-22, that doing so fails immediately:
`ModuleNotFoundError: No module named 'alphafold3.jax.attention'`. That
host script's top-level `from alphafold3.jax.attention import attention`
(added by a LATER point in the same RomeroLab fork's history than this
image was built from) has no matching module in the `alphafold3` package
this image actually bakes in (confirmed by listing the image's
`alphafold3/jax/` directory: only a `geometry` subpackage). Instead, this
tool runs the image's OWN baked-in inference entrypoint -- extracted
alongside the venv itself, now under
`/alphafold3_venv/app/alphafold/` (see "How this tool is
dispatched") -- directly: the same fork family, the same
data-pipeline-disable flag and recycle/sample/MSA-overlap/flash-
attention/bucket/conformer flags this tool's schema already covers below,
built against `tokamax`/`ModelRunner` instead, and guaranteed
self-consistent with the package actually installed in this venv. This is
a real, live-confirmed incompatibility between two snapshots of the same
upstream fork, not a bug in either script on its own.

This also changes the OUTPUT layout from what the official AlphaFold 3
docs describe (see each `outputs:` entry's own comment in this manifest):
called with `--json_path` (not `--input_dir`) and the data-pipeline flag
disabled -- this tool's own call shape -- this entrypoint writes
DIRECTLY into `output_dir`, with no extra `<job_name>/` nesting.

## How this tool is dispatched -- read this before deploying it
**This changed in task 16.** AF3 used to run from its own Docker image as
a SIBLING container: this tool's wrapper shelled out to `docker run
romerolabduke/alphafast:latest` directly, which required this server's
OWN container to have the `docker` CLI on PATH and the host's Docker
socket bind-mounted in (`-v /var/run/docker.sock:/var/run/docker.sock`).
That is a ROOT-EQUIVALENT capability -- bind-mounting the socket grants
anything that can reach it full control of the host -- and this server's
HTTP transport has no authentication (defect C9, still open), so Ruling 3
forbade it as the default containerised path: `run_alphafold3` was
HOST-SIDE ONLY (Ruling 4), `FileNotFoundError: 'docker'` in-container.

AF3 only needed a sibling container because upstream SHIPS it as a Docker
image, not because inference itself requires one. This server already
mounts ~20 host conda environments and dispatches into them via
`EngineSpec.prefix` (`micromamba run -p <prefix> <entry>`); AF3 is now
the twenty-first, exactly the same way. `/alphafold3_venv` (this image's
own venv, python 3.12) and `/alphafold3_venv/app/alphafold`
(its inference entrypoint script plus the `alphafold3` package source)
were extracted from `romerolabduke/alphafast:latest` with `docker create` + `docker cp`
-- no container needs to run for that, only to exist locally -- onto this
host, at `engine.prefix_host` (see this manifest's `engine:` block for
the full reasoning, and `EngineSpec.prefix_host`'s own docstring for why
a SECOND path is needed at all here and nowhere else). `docker.sock` is
never mounted, the `docker` CLI is never needed inside this server's own
container, and `scripts/container_run.py --check` (task 16) confirms
zero socket mounts in the derived recipe.

This tool's wrapper script (`scripts/engines/alphafold3.py`) now runs
UNDER the mounted venv's own python (via `micromamba run -p
/alphafold3_venv`) and invokes the entrypoint script under
`/alphafold3_venv/app/alphafold/` as a plain local subprocess -- no
docker involved anywhere in this call.
GPU selection needs NO handling here any more either: the GPU is pinned
at THIS server's own container boundary (design §2.1,
`--device=nvidia.com/gpu=7`), the same as every other GPU engine, instead
of needing its own `--gpus device=7` re-applied at a second, sibling
container boundary the way the old design did.

## MSA is optional and NEVER built by this tool
AlphaFold 3's own alignment-search flag is always passed as `false` --
AF3 never searches for its own alignment, matching this server's
universal MSA policy (see
`docs/TOOLS.md`). Every chain's `unpaired_msa`/`paired_msa` must be BOTH
`null` (run that chain MSA-free) or BOTH a path (use these alignments) --
never one of each. `run_mmseqs_search`'s own doc already documents this
exact pairing: pass its `unpaired_a3m` output as `unpaired_msa` and its
`paired_a3m` output as `paired_msa` (that output is always written, even
for a chain you do not intend to pair, specifically so it can satisfy
this "both set" requirement). `null`/`null` is translated to AlphaFold
3's OWN `""`/`""` encoding (its documented "equivalent to running
completely MSA-free" combination) -- NEVER to JSON `null`/`null`, which
is AlphaFold 3's OWN encoding for "build both MSAs automatically", i.e.
exactly the self-search behaviour disabling that flag exists to
prevent. `run_colabfold_search`'s a3m is a DIFFERENT sequence universe
(UniRef30/envDB, not AlphaFold 3's own database set) -- valid here too,
but do not mix producers within one call: pair each chain's
`unpaired_msa`/`paired_msa` from the SAME search tool.

## Chain composition is explicit and never inferred
List every chain you want folded together in this call -- the target and
the binder together for a complex prediction, the binder alone to score
it in isolation. Nothing is inferred from anything else you have called.

## When to use this instead of the alternatives
`run_boltz`, `run_chai1`, `run_protenix`, `run_openfold3` and `run_promera`
cover the same "chains + optional per-chain MSA" job shape, each from a
different model; comparing their outputs on the SAME `msa` input is
exactly what this server's MSA-tool-first design exists to make possible
-- do not change engines and alignment source in the same comparison.
This tool is the heaviest and slowest of the family -- a full AlphaFold 3
inference pass per call, dispatched exactly like every other GPU engine
here (see "How this tool is dispatched" above) but the most expensive one
to run -- prefer a lighter structure predictor for routine screening and
reserve this one for a candidate you already want AlphaFold 3's specific
prediction on.
`run_rf3` and `run_esmfold2` are the fastest members of this family
(RF3 takes an optional MSA; ESMFold2 takes none at all) and are the
better choice for a first-pass, high-throughput filter before spending
this tool's much larger cost on a shortlist.

## Not exposed
- RNA, DNA, ligands, covalent bonds (`bondedAtomPairs`), user-provided
  CCD, and structural templates -- this version handles protein chains
  only. AlphaFold 3's own JSON schema supports all of these; they are a
  real, deliberate scope boundary of this tool, not an oversight, given
  this wave's time budget -- stated plainly rather than silently dropped.
- AF3's own alignment-search flag and its GPU-device-index flag: pinned,
  never a parameter -- see "MSA is optional" above and design §2.1/§7
  (the GPU is pinned at the container boundary, never a per-engine field).
- `--jax_compilation_cache_dir`: an internal performance cache, pointed at
  a scratch subdirectory automatically, the same way this project already
  handles HF_HOME/TORCH_HOME for conda-mounted engines -- never a
  caller-facing knob.

## What you must supply
`chains`: one entry per chain in the assembly, each
`{"sequence": "<protein AA string>", "unpaired_msa": "<path>" | null,
"paired_msa": "<path>" | null, "copies": <int, default 1>}`. Both MSA
fields are required on every entry, with no default -- see "MSA is
optional" above.

## What you get back
`ranking_score`, `ptm`, `iptm`, `fraction_disordered`, `has_clash` (from
the top-ranked sample), `num_samples`, and under `outputs` the paths to
the top-ranked structure/confidence files, the full ranking_scores.csv,
and every individual seed/sample's own structure and confidences.

## Parameters

| Parameter | Type | Required | Default | Constraints | Description |
|---|---|---|---|---|---|
| `chains` | array | yes | `—` | minItems: `1`<br>maxItems: `20` | One entry per chain in the predicted assembly: {"sequence": "<protein amino-acid string, uppercase, standard 20 plus X/B/Z/J/U/O>", "unpaired_msa": "<path to an unpaired a3m for this exact sequence>" or null, "paired_msa": "<path to a paired a3m for this exact sequence>" or null, "copies": <positive integer, default 1, identical copies of this chain -- use for a homo-oligomer instead of repeating the entry>}. Chain composition (who's present, alone or in complex) is never inferred -- list exactly the chains you want folded together. Both MSA fields are required on every entry (no default); both must be null (MSA-free) or both a path (use these alignments) -- one null and one a path is rejected. |
| `seeds` | array | no | `[1]` | minItems: `1`<br>maxItems: `20` | Random seeds AlphaFold 3 samples with -- one full inference pass per seed, each producing num_diffusion_samples structures. [1] is this tool's own default (a single, reproducible seed); pass several to get a broader diverse set at roughly linear extra cost. |
| `num_recycles` | integer | no | `10` | minimum: `1`<br>maximum: `20` | Number of recycling passes through the trunk before diffusion. 10 is AlphaFold 3's own default (`run_alphafold.py`'s `--num_recycles`). More recycling can improve hard interfaces at roughly linear extra cost. |
| `num_diffusion_samples` | integer | no | `5` | minimum: `1`<br>maximum: `20` | Number of independent structures to sample PER seed. 5 is AlphaFold 3's own default. Total structures this call produces is len(seeds) * num_diffusion_samples. |
| `max_template_date` | string | no | `2021-09-30` | pattern: `^\d{4}-\d{2}-\d{2}$` | Format YYYY-MM-DD. AlphaFold 3's own default (the AlphaFold 3 paper's cutoff date) -- even with templates not exposed here (see "Not exposed"), this date still gates whether RDKit-conformer-generation fallback to CCD model coordinates is allowed for a chemical component, so `run_alphafold.py` still requires a value. |
| `resolve_msa_overlaps` | boolean | no | `True` | — | Whether to deduplicate a chain's unpaired MSA against its paired MSA. True is AlphaFold 3's own default. Set false only if unpaired_msa was built with deliberate, hand-crafted cross-chain pairing you do not want deduplication to disturb (AlphaFold 3's own docs' explicit recommendation for that specific expert workflow). |
| `flash_attention_implementation` | string | no | `triton` | enum: `['triton', 'cudnn', 'xla']` | Which flash-attention kernel to use. triton is AlphaFold 3's own default and the most thoroughly tested; cudnn is a comparable alternative; xla is portable (no flash attention) and the ONLY valid choice on a GPU with compute capability in [7.0, 8.0) per AlphaFold 3's own startup check (not relevant on this host's L40S, compute capability 8.9, but stated since this tool may run on other hardware). |
| `save_embeddings` | boolean | no | `False` | — | Whether to save the trunk's single/pair embeddings (num_tokens*384 + num_tokens^2*128 float16 values -- several GB for a large complex). False is AlphaFold 3's own default; only needed for downstream embedding-based analysis this server does not otherwise provide. |
| `save_distogram` | boolean | no | `False` | — | Whether to save the predicted distogram (num_tokens^2*64 float16 values -- also several GB for a large complex). False is AlphaFold 3's own default. |
| `buckets` | array | no | `[256, 512, 768, 1024, 1280, 1536, 2048, 2560, 3072, 3584, 4096, 4608, 5120]` | minItems: `1` | Strictly increasing token-count sizes AlphaFold 3 caches JIT compilations for; an input with more tokens than the largest bucket gets its own bucket for exactly that count. This is AlphaFold 3's own default bucket list (`run_alphafold.py`'s `--buckets`) -- a pure compilation-cache performance knob, exposed because this project's policy is never to hide a real knob behind a default, but it does not change a prediction's result. |
| `conformer_max_iterations` | integer | no | `None` | minimum: `0` | Optional override for RDKit's maximum conformer-search iterations. null (AlphaFold 3's own default) uses RDKit's own default parameters. Not relevant to a protein-only chain set (this tool does not expose ligands -- see "Not exposed"), but this flag is unconditional in `run_alphafold.py`'s own invocation, so it is exposed for completeness. |
