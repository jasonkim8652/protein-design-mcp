# run_alphafold3

**Category:** structure_prediction  
**Engine:** `alphafold3`  
**Environment:** `scoring`  
**GPU required:** yes

> This file is generated from `src/protein_design_mcp/manifests/run_alphafold3.yaml`. Edit the manifest, then run `python scripts/generate_tool_docs.py`.

## Summary

Predict a multi-chain protein structure with AlphaFold 3, given an explicit alignment (or none) per chain. The heaviest tool in this server: runs as a SIBLING Docker container from AF3's own image, not a process in a mounted conda environment -- see the doc's "How this tool is dispatched" section, since that changes what an operator must configure for this one tool to work at all. MSA is optional and NEVER built by this tool (AF3's own alignment-search step is always disabled) -- pass run_mmseqs_search's unpaired_a3m/paired_a3m outputs, or null/null to run MSA-free. Protein chains only in this version -- no RNA, DNA, ligands, covalent bonds, or templates; see "Not exposed".

## What this is
AlphaFold 3, run from the user's own checkout `~/projects/af3-mmseqs-gpu`
(ground truth for invocation: the reference inference script under its
`benchmarks/` directory for the `docker run` shape,
`docs/input.md`/`docs/output.md` for the JSON schema and output layout)
-- the current state-of-the-art structure
predictor, covering the same "chains + optional per-chain MSA" job shape
as `run_boltz`/`run_chai1`/`run_protenix`/`run_openfold3`, but from a
genuinely different model.

## How this tool is dispatched -- read this before deploying it
AF3 runs from its OWN Docker image (per the design spec, §4: "AF3 itself
keeps running as its own container from its own image"), completely
separate from every other engine in this server, which run as
subprocesses inside conda environments mounted into THIS server's own
container. This tool's wrapper script therefore shells out to `docker
run` directly, launching a SIBLING container next to (not nested inside)
this server's own container.

This is a DEPLOYMENT requirement, stated explicitly rather than left
implicit: for this tool to work, this server's OWN container needs (1)
the `docker` CLI on PATH, and (2) the host's Docker socket bind-mounted
in (e.g. `-v /var/run/docker.sock:/var/run/docker.sock`), so `docker run`
here reaches the HOST's daemon and starts a sibling, not a nested,
container -- neither of which any manifest or adapter can arrange; both
are the operator's own container-runtime configuration, exactly like the
read-only host mounts every other GPU engine already needs. See the wave
report for what was and was not possible to verify about this from
inside an actual deployed instance of this server's own container.

There is a SECOND, related deployment requirement: the dispatcher's
scratch workdir (see `dispatch.env.EnvDispatcher`'s `scratch_root`) is
bind-mounted into this tool's `docker run` call so AF3 can read the job
JSON and write its output where this tool's `outputs:` patterns expect
it. Docker resolves a bind-mount SOURCE path against the HOST's own
filesystem, not this server's own container's filesystem -- so
`scratch_root` must be a path that is ALSO valid, at the IDENTICAL path,
on the host running the Docker daemon (the same "mount environments at
their own host path" rule design §2.2 already applies to conda
environments, applied here to the scratch directory instead). This
wrapper does not, and cannot, detect or correct a mismatch here -- it
trusts the path it is given.

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
This tool is the heaviest and slowest of the family (a full sibling
Docker container per call, see "How this tool is dispatched" above) and,
unlike the others, is not a simple conda-mounted subprocess -- prefer a
lighter structure predictor for routine screening and reserve this one
for a candidate you already want AlphaFold 3's specific prediction on.
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
  a scratch subdirectory inside the sibling container automatically, the
  same way this project already handles HF_HOME/TORCH_HOME for
  conda-mounted engines -- never a caller-facing knob.

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
