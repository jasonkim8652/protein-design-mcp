# Atomistic tool refresh — design spec

Date: 2026-09-21
Branch: `dev`
Status: awaiting review

## 1. Problem

`protein-design-mcp` v1.0.0 exposes 19 tools built around a 2023-era stack
(RFdiffusion → ProteinMPNN → ESMFold/AF2) plus PyRosetta and Boltz. Five
problems motivate a rewrite of the tool surface.

**The tool list is out of date.** The atomistic protein-protein design field
moved substantially in 2025-2026. Target-conditioned all-atom generative models
(Proteina-Complexa, BoltzGen), open co-folding models with interface confidence
(ESMFold2, Chai-1, Protenix, OpenFold3, RF3), and calibrated binder/non-binder
discriminators (ipSAE, Promera iCS) did not exist when this server was written.

**Composite tools take control away from the caller.** `design_binder` runs
RFdiffusion → ProteinMPNN → ESMFold internally. The caller cannot inspect the
backbone before sequence design, cannot change the MPNN temperature based on
what the backbone looks like, and cannot substitute a different folding model.
The tool returns a ranked list and the caller must trust it.

This is not hypothetical. `design_binder` currently returns the *target* chain
as its own design: `tools/design_binder.py:113` and `tools/design_sequence.py:86`
do `sequence.split("/")[-1]` assuming ProteinMPNN emits `"TARGET/BINDER"`, but
RFdiffusion places the binder in chain A so the real output is
`"BINDER/TARGET"`. A caller that trusts `designs[0]` presents the target as its
own binder. A caller that had been handed the intermediate PDB would have caught
it.

**The composite gate is dead code.** `server.py:44` defines
`COMPOSITE_TOOL_NAMES` with the comment "Composite tools hidden in benchmark
mode". Nothing in the repository reads that variable. `list_tools()` filters
only on `GPU_ONLY_TOOLS`. The set is also wrong: it omits `design_fold` and
`validate_design`, and it wrongly includes `design_sequence`, which is a plain
ProteinMPNN wrapper.

**Schemas do not constrain what the caller sends.** `hotspot_residues` is
`array<string>` with the format `['A45', 'A46']` described in prose and no
`pattern`. `rosetta_interface_score`'s `chains` uses a bespoke `'A_B'` grammar
as a free string. A caller that sends `"45"` or `"A:45"` passes schema
validation and fails at runtime, deep inside a subprocess, with an error that
does not say what the correct format is.

**One tool is scientifically wrong.** `predict_affinity_boltz` accepts
`sequences: [chain_a, chain_b]` and returns "binding affinity". Boltz-2's
affinity head is protein-ligand only — its correction flag is
`--affinity_mw_correction`, a molecular-weight term. arXiv 2512.06592 (MLSB
2025) fine-tuned Boltz-2 for protein-protein affinity and found it
underperformed sequence-based baselines. The tool returns plausible numbers
that mean nothing for a protein-protein interface.

## 2. Goals

1. Replace the tool list with current atomistic methods, chosen on published
   capability rather than on what is already installed.
2. Make every tool's parameters controllable and hard to get wrong.
3. Make composite pipelines unreachable, not merely hidden.
4. Rename tools to `run_<engine>[_<step>]` and move all meaning into
   documentation, so a tool name cannot make a promise the tool does not keep.
5. Run every tool from one Docker image.

### Non-goals

- Training or fine-tuning any model.
- Hallucination/optimization pipelines (BindCraft, FreeBindCraft, ColabDesign,
  BoltzDesign1, mosaic, Odin-Multi, EasyNano, dl_binder_design). See §3.3.
- Backwards compatibility with the v1.0.0 tool names. This is a breaking change.

## 3. Tool selection

### 3.1 What counts as atomistic

A tool qualifies when it is **one model's own inference procedure**, exposed at
the granularity its authors exposed it. It does not qualify when the MCP server
chains several independent models together and hides the seams.

This distinction is not about step count. Proteina-Complexa's binder search runs
generation, reward scoring and beam search — but its CLI already decomposes into
`generate`, `filter`, `evaluate` and `analyze`, and the caller drives the
test-time compute budget. BoltzGen is the same shape: `boltzgen run` is a
seven-stage pipeline, but `--steps` runs each stage alone, and every stage uses
BoltzGen's own weights rather than third-party tools.

So the rule is: **expose the steps, block the orchestrator.** `complexa design`
and `boltzgen run` are not registered; their steps are.

**An author's CLI boundary is not evidence that a step is atomic.** Added
2026-09-22, after this spec used exactly that reasoning to admit a composite. The
paragraph above defended Proteina-Complexa on the grounds that "its CLI already
decomposes into generate, filter, evaluate and analyze". That is a statement about
the authors' workflow convenience, not about our rule. Opening `evaluate` showed it
bundles refolding, interface analysis and force-field metrics — three capabilities
this server already exposes individually — behind a single call, and carries the
hidden convention `binder is last chain` (`evaluate.py:6`). That is the same shape
of buried chain-order assumption that made `design_binder` return the target as its
own design (§1), which is the bug that motivated this whole rewrite.

So the test is applied to what a step *does*, not to where its authors drew a
subcommand: **open every pipeline step before admitting it.** A step qualifies only
if it is one model's own inference, or a pure transformation over that model's own
output that runs no other model.

### 3.2 The tool list (34 tools + 2 meta-tools)

> **Amended 2026-09-22.** The original list held 29 tools and was written before the
> engine version sweep. Two engines were missed for the same reason — both postdate it —
> and both were caught by the user, not by this document:
>
> - **Genie 3** does binder design and motif scaffolding, not only the unconditional
>   generation Genie 2 offers. Its README claims "unconditional generation, motif
>   scaffolding, and binder design", and `scripts/problem/binder_design/` exists in the
>   checkout. Genie was filed under §B alone, which was wrong — it belongs in §A too.
>   Added `run_genie3_binder` (§A) and `run_genie3_scaffold` (§B).
> - **RFdiffusion3** was released in December 2025, after this list was written. Added
>   `run_rfdiffusion3` (§A). It ships through RosettaCommons **foundry**
>   (`pip install rc-foundry`, BSD-3), whose `models/` holds `rf3`, `rfd3`, `rfd3na`
>   and `mpnn` — one install serves both `run_rfdiffusion3` and `run_rf3`.
>   **`rc-foundry` requires python 3.12**; querying it from a 3.10 interpreter returns
>   "No matching distribution found", which is indistinguishable from the package not
>   existing. That false negative nearly removed RFD3 from this plan.
>
> **Removed 2026-09-22: `run_proteina_complexa_evaluate`.** It is a composite by
> §3.1 — it bundles refolding, interface analysis and force-field metrics, all of
> which this server exposes individually, and it hides a `binder is last chain`
> convention. Replace it by composing `run_esm_score` (it loads
> `facebook/esm2_t33_650M_UR50D`, the same weights as `run_esm_score`, via
> `AutoModelForMaskedLM` — it is **not** ESMFold, despite a misleading comment at
> `evaluate.py:49`) with `run_ipsae` / `run_prodigy` / `run_rosetta_interface`, and
> `run_openmm_minimize` where relaxation is wanted. The caller then states the chain
> roles explicitly instead of inheriting an assumption.
>
> `run_proteina_complexa_filter` and `run_proteina_complexa_analyze` were checked
> against the same rule and kept: `filter` runs no model at all (it re-ranks the
> `rewards_*.csv` that `generate` wrote), and `analyze`'s foldseek/mmseqs diversity
> is a capability nothing else here provides.
>
> The lesson for whoever amends this next: a list assembled from a point-in-time sweep
> goes stale silently, and "engine X is unavailable" must be checked with the right
> interpreter and the right distribution channel before it is believed.

#### A. Target-conditioned binder generation

| Tool | Engine | License | Notes |
|---|---|---|---|
| `run_proteina_complexa_generate` | Proteina-Complexa 160M | Apache-2.0 code, NVIDIA Open Model License weights | Partially latent flow matching; sequence and all atoms generated jointly. ICLR 2026 oral. **This is where test-time compute is spent** — rewards are computed inline during generation (`generate.py:316`) and `utils/mcts_utils.py` does the search. Corrected 2026-09-22; this note previously credited `filter`. |
| `run_proteina_complexa_filter` | ″ | ″ | **Tabular re-ranking — runs no model.** Reads the `rewards_*.csv` that `generate` wrote, dedups sequences, applies `reward_threshold`, keeps top-N. Cheap, CPU, re-runnable with different thresholds without regenerating. |
| `run_proteina_complexa_analyze` | ″ | ″ | Aggregation plus **diversity**: `compute_foldseek_diversity` / `compute_mmseqs_diversity` over the run. Runs foldseek/mmseqs, not a neural model. `complexa analysis` = evaluate → analyze. |
| `run_boltzgen_design` | BoltzGen 0.3.2 | **MIT — code, weights, training data** | All-atom diffusion, target from PDB/CIF directly, MSA-free. Confirmed working on this machine's L40S. |
| `run_boltzgen_inverse_fold` | ″ | ″ | BoltzGen's own IF head (12.6 MB), not ProteinMPNN. |
| `run_boltzgen_filter` | ″ | ″ | **Runs no model, and is not Boltz-2.** Pure dataframe ranking over columns BoltzGen's own predict/score steps produced (`design_iptm`, `min_interaction_pae`, `bb_rmsd`, `delta_sasa_refolded`, `structure_confidence`). The installed distribution has no `boltz` package and imports none. Distinct from `run_boltz`, which runs Boltz-2 inference on a GPU. |
| `run_rfdiffusion_binder` | RFdiffusion | BSD-3 (weights status ambiguous — see §7) | Backbone only; legacy baseline for comparison. |
| `run_rfdiffusion2` | RFdiffusion2 | BSD-3 | All-atom motif/interface. |
| `run_rfdiffusion3_binder` | RFdiffusion3 (RFD3) | BSD-3 | Target-conditioned binder design (`contig` + `select_hotspots`). Via foundry: `foundry install rfd3`, entry `rfd3 design`. **Python 3.12 only.** |
| `run_genie3_binder` | Genie 3 | permissive | Target-conditioned binder design (`scripts/problem/binder_design/`). Added 2026-09-22 — Genie 3 is not merely a newer Genie 2. |
| `run_protpardelle` | Protpardelle-1c | CC-BY-4.0 | All-atom multichain with hotspot conditioning. |

#### B. Monomer / scaffold generation

| Tool | Engine | License |
|---|---|---|
| `run_rfdiffusion3_scaffold` | RFdiffusion3 (RFD3) | BSD-3 |
| `run_genie3_scaffold` | Genie 3 | permissive |
| `run_genie2` | Genie2 | permissive |
| `run_frameflow` | FrameFlow | permissive |
| `run_multiflow` | MultiFlow | permissive |
| `run_la_proteina` | La-Proteina | Apache-2.0 code, NVIDIA OML weights |

These generate monomers, not complexes. They are included so a caller can build
a scaffold first and condition on it, which is a distinct workflow from direct
binder generation.

`run_genie3_scaffold` is motif scaffolding — conditioning on a motif rather than on a
target — which is why it sits here and `run_genie3_binder` sits in §A. `run_genie2`
is kept alongside it because Genie 2's surface is exactly
`sample_unconditional.py` / `sample_scaffold.py`, it has working weights on this host,
and Genie 3's checkpoints are not downloaded yet (`assets/` is 6 MB, a demo gif).
Retire `run_genie2` once Genie 3's weights are in place and exercised.

#### C. Sequence design

| Tool | Engine | License | Notes |
|---|---|---|---|
| `run_mpnn` | **ProteinMPNN** / LigandMPNN / SolubleMPNN (`dauparas/LigandMPNN`) | **MIT, weights in-repo** | One codebase serves all three; selected by `model_type` (`protein` is the default and IS ProteinMPNN), not by three separate tools. **The one-tool decision stands, but discoverability does not follow from it**: the enum values never name their engine, so a caller searching for "ProteinMPNN" cannot find it. Each enum value must name its engine in the schema description and in `docs/tools/run_mpnn.md`. Tracked as M3. |

#### D. Co-folding / structure prediction

| Tool | Engine | License | MSA | Notes |
|---|---|---|---|---|
| `run_esmfold2` | ESMFold2 / -Fast | **MIT, ungated** | optional | ESMC-6B backbone. Reports `iptm`, `pair_chains_iptm`, `complex_iplddt`. |
| `run_chai1` | Chai-1 0.6.1 | Apache-2.0 code+weights | off by default | L40S is an explicitly supported SKU. Lowest integration risk. |
| `run_boltz` | Boltz-2 2.2.1 | MIT | server/precomputed | Confidence only. Affinity head **not exposed** (§1). |
| `run_protenix` | Protenix v1 | Apache-2.0 | optional | Richest confidence output: `chain_iptm`, `chain_pair_iptm`. **`pip install protenix` now resolves to 2.0.0, not the 0.5.5 this spec first recorded.** 2.0.0's default `model_name` is already `protenix_base_default_v1.0.0` (368.48M params; the proprietary v2 is 464M and opt-in only) — verified live, but **always pass `-n protenix_base_default_v1.0.0` explicitly** rather than relying on a silent default that a future release can change. |
| `run_openfold3` | OpenFold3 / OpenBind-0 | Apache-2.0 code+weights+data | server default | `num_diffusion_samples` capped at 5 (§5.3). |
| `run_promera` | Promera | MIT | optional | **Moved from §E and renamed 2026-09-22.** It was filed as scoring-only, and as duplicating `run_ipsae`. Both were wrong: its input is a directory of target *schemas* plus an `msa_dir`, so it co-folds. Its iCS and ipSAE are its own confidence in its own prediction — the same kind of output as Chai's `iptm`, not a second opinion on someone else's structure. `run_ipsae` remains the model-agnostic tool for scoring any predictor's PAE. Promera's `Design` task stays excluded (composite). |
| `run_rf3` | RoseTTAFold3 | BSD-3 | **none — bring your own a3m** | Output schema unstable (§7). |
| `run_alphafold3` | AF3 (RomeroLab MMseqs2-GPU fork) | weights non-commercial, no redistribution | required | Bring-your-own weights (§5.4). |
| `run_alphafold2_multimer` | AF2-Multimer / ColabFold | Apache-2.0 code, CC-BY-4.0 params | required | Still the reference ipTM discriminator for binder filtering. |

#### E. Scoring and analysis

| Tool | Engine | License | Notes |
|---|---|---|---|
| `run_ipsae` | ipSAE (Dunbrack lab) | permissive | Computes ipSAE from any model's PAE. Model-agnostic, CPU, seconds. |
| `run_rosetta_interface` | PyRosetta InterfaceAnalyzer | **redistribution prohibited** | `dG_separated`, `dSASA`, `sc`, hbonds. Installed at runtime (§5.4). |
| `run_prodigy` | PRODIGY | Apache-2.0 | CPU, milliseconds. Calibrated on natural complexes — see doc warning. |
| `run_openmm_minimize` | OpenMM 8.6 | MIT/LGPL | Relaxation before scoring. |
| `run_esm_score` | ESM2-650M / ESM-C 300M | MIT | Pseudo-likelihood as a developability proxy, not a binding predictor. |

#### G. Multiple sequence alignment

| Tool | Engine | Notes |
|---|---|---|
| `run_mmseqs_search` | MMseqs2 (`/usr/local/bin/mmseqs`) | Builds an a3m from a sequence against the local databases. |
| `run_colabfold_search` | `colabfold_search` | ColabFold's own search pipeline, for the consumers that expect its a3m. |

**Added 2026-09-22 — this was a hole, not an omission of convenience.** The list carried
33 tools of which three (`run_rf3`, `run_alphafold3`, `run_alphafold2_multimer`) either
require an MSA or, in RF3's case, explicitly say "bring your own a3m" — and **nothing in
the list could produce one**. A caller following the documentation reached a dead end.

Two tools rather than one because **their outputs are not interchangeable**. MMseqs2
against AlphaFold 3's database set and `colabfold_search` against UniRef30/envDB produce
different alignments over different sequence universes, and a consumer built for one may
silently accept the other and give worse results rather than failing. Each tool's
documentation must state which consumers its a3m is valid for, and each co-folding tool's
`msa` parameter must state which producer it expects.

Local search only, by default. ColabFold's remote MSA server would transmit the caller's
sequences to a third party, and on this server those sequences are frequently novel
designs. If a remote mode is ever added it must be opt-in, and its documentation must say
plainly that the sequence leaves the machine.

Assets already present: `/opt/alphafold3_data/mmseqs_db` (1.3 TB, protein and RNA) and
`/opt/alphafold3_data/fasta_databases` (395 GB).

#### F. Meta (2)

| Tool | Purpose |
|---|---|
| `describe_tool` | Returns the full document for a tool, or lists a category with selection guidance. See §4.3. |
| `get_job_status` | Polls a long-running job. Renamed from `get_design_status`; generation calls run for minutes to hours. |

### 3.2.1 Parameter exposure policy

Added 2026-09-22 at the user's direction: **the caller decides, and the schema says how.**

1. **Every knob the engine exposes is exposed here**, unless it selects between a step
   this server registers separately or it would let the caller escape the workdir. A
   hyperparameter that only changes the engine's behaviour is never hidden because a
   default "usually works" — Proteina-Complexa's sampling settings, diffusion step counts,
   temperatures, seeds, sample counts and beam widths all appear in the schema.
2. **MSA use is an explicit parameter on every co-folding tool**, never an implicit
   default. The parameter states whether an MSA is used, where it comes from, and which
   producer's a3m is expected (§G). A tool that can run MSA-free says so and says what it
   costs in accuracy.
3. **Chain composition is an explicit parameter.** Whether a prediction runs as a multimer
   with the target present, or as the binder alone, is the caller's decision and one of the
   most consequential it makes — a binder predicted alone and a binder predicted in complex
   are different experiments. No tool may infer this from the shape of its input.
4. **Every parameter's description states what it does, what changes when it moves, and
   what a sensible range is** — not merely its type. "Which trained variant to use" is not
   a description; it names no variant and implies no consequence. A caller that has only
   the schema must be able to choose correctly from it.
5. **Defaults are documented as choices, not as facts.** Where a default exists, its
   description says why that value and when to move off it.

The cost of getting this wrong is asymmetric: a hidden parameter cannot be discovered by a
caller, while an exposed one with a good default costs a line of documentation.

### 3.3 Explicitly excluded, and why

Recorded here so `describe_tool` can explain the exclusion when a caller asks
for one of these.

| Excluded | Reason |
|---|---|
| `complexa design`, `boltzgen run` | Orchestrators over steps this server exposes individually. |
| Promera `Design` task | Depends on an external LigandMPNN fork; composite. |
| BindCraft, FreeBindCraft, ColabDesign, BoltzDesign1, mosaic, Odin-Multi, EasyNano, dl_binder_design, RFantibody | Gradient-based hallucination through a folding model. Beyond the composite rule, these need hand-tuned learning rates and, per mosaic's own README, "often produce proteins that fail simple in-silico tests". Success depends on operator judgement the caller cannot exercise or diagnose. |
| Boltz-2 affinity head | Protein-ligand only (§1). |
| Protenix v2 | Weights proprietary; must pin `-n protenix_base_default_v1.0.0`. |
| SeedProteo | No public code or weights as of 2026-09-21. |
| Chai-2, AlphaProteo, Pearl | No public weights. |
| Proteina (original), Chroma weights | Non-commercial / academic-only licenses. |

## 4. Tool manifest architecture

### 4.1 The problem it solves

Today a tool's identity is spread across three places: the `Tool(...)` literal
in `server.py`'s 1181-line `TOOLS` list, a branch in `call_tool`'s 19-arm
if/elif chain, and nothing else. Adding a tool means editing two places;
documentation lives nowhere. `COMPOSITE_TOOL_NAMES` became dead code precisely
because it was a fourth place that nothing forced anyone to update.

### 4.2 One manifest per tool

`tools/<name>.yaml` is the single source of truth:

```yaml
name: run_proteina_complexa_generate
category: generation
engine:
  repo: proteina-complexa
  env: complexa
  entry: ["complexa", "generate"]
composite: false
requires:
  gpu: true
  weights: ckpts/complexa.ckpt

summary: >                       # becomes Tool.description, ~600 chars
  Generate binder candidates against a target protein with Proteina-Complexa.
  Produces sequence and all-atom coordinates jointly. Output is unvalidated:
  fold it with run_esmfold2 or run_chai1 and score the interface with
  run_rosetta_interface before trusting any candidate.

doc: |                           # full document, served by describe_tool
  ## What this is
  ...
  ## When to use this instead of the alternatives
  ...
  ## What you must supply
  ...
  ## What you get back
  ...

schema:
  target_pdb:
    type: string
    pattern: '\.(pdb|cif)$'
    description: Path to the target structure.
  hotspot_residues:
    type: array
    items: { type: string, pattern: '^[A-Za-z][0-9]+$' }
    minItems: 1
    description: "Target residues at the interface, chain letter then number: ['A45','A46']."
  binder_length: { type: integer, minimum: 20, maximum: 300 }
  num_samples:   { type: integer, minimum: 1, maximum: 100, default: 8 }
  seed:          { type: integer, default: 0 }
```

Four artifacts derive from it, so they cannot drift:

| Artifact | Built from |
|---|---|
| `Tool(name, description, inputSchema)` | `name`, `summary`, `schema` |
| `describe_tool` response | `doc` |
| `docs/tools/<name>.md` | `summary` + `doc` + `schema` (generated, committed) |
| Dispatch registry entry | `engine` |

Registration filters on `composite` and on `requires`. A composite tool is never
built into a `Tool`, and — because dispatch reads the same registry — calling its
name by hand fails too. This closes the gap where `GPU_ONLY_TOOLS` must be
checked twice, once in `list_tools` and again in `call_tool`.

### 4.3 Making the names safe

Requirement: tool names become mechanical (`run_<engine>_<step>`) and all
meaning moves to documentation. The risk is that a harness which does not
surface documentation leaves the model with a bare name.

Documentation is therefore delivered in three layers, each independently
sufficient for a different client:

| Layer | Carries | Available in |
|---|---|---|
| `Tool.description` (from `summary`) | what it is, its role, which tools come before and after | **every** MCP client |
| `describe_tool(name)` / `describe_tool(category=...)` | the full `doc` | any client — it is an ordinary tool |
| MCP Resource `doc://tools/<name>` | the full `doc` | clients that surface resources |

Layer 1 is the floor and must be self-sufficient. Layer 2 exists because many
clients never expose MCP resources to the model, which would otherwise make
layer 3 the only home for the real documentation.

`describe_tool(category="cofolding")` returns every tool in the category with its
selection criteria. This is the mitigation for having eight co-folding tools
whose inputs and outputs look alike: the caller asks once and gets a comparison
rather than guessing from eight similar descriptions.

Every `doc` must contain a **"When to use this instead of the alternatives"**
section naming specific sibling tools and the condition that selects each. For
example, `run_chai1` points to `run_esmfold2` for antibody-antigen, to
`run_protenix` when per-chain-pair ipTM is needed, and to `run_boltz` when a
small molecule is present.

## 5. Execution environment

### 5.1 One image, many environments

A single flat environment is impossible. Verified pins:

| Tool | Python | numpy | torch | other |
|---|---|---|---|---|
| Boltz-2 | >=3.10,<3.13 | **>=1.26,<2.0** | >=2.2 | gemmi==0.6.5 |
| Protenix | >=3.11 | **==2.4.1** | ==2.7.1 | gemmi==0.6.7 |
| Chai-1 | >=3.10 | >=1.21 | >=2.3.1,**<2.7** | gemmi~=0.7.5 |
| RF3 | **>=3.12,<3.13** | — | >=2.2,<3 | atomworks[ml]>=2.1.1 |
| ESMFold2 | **>=3.12** | — | **>=2.11,<2.12** | transformers<5.0 |
| BoltzGen | >=3.11 | **==2.0.2** | >=2.4.1 | numba==0.61.0 |
| Promera | >=3.12 | >=2.4.2 | **==2.9.0** | cuequivariance==0.8.0 |
| PyRosetta | — | **<2** | — | numpy 1.x ABI |

numpy alone is unsatisfiable (`<2.0`, `==2.0.2`, `==2.4.1`, `>=2.4.2`, `<2`).
gemmi has three mutually exclusive pins. torch has two (`<2.7` and `==2.9.0`).
Python 3.12 is the only version satisfying Boltz, RF3 and ESMFold2 together, and
that still leaves numpy unresolved.

So: **one Docker image containing one micromamba environment per engine**, with
the MCP server in a thin outer environment dispatching by subprocess.

This generalises a pattern already in the repository. `boltz_runner.py:175-189`
wraps its command in `conda run -n <env> --no-banner` when `conda_env` is set and
calls the binary directly otherwise. Today each runner reimplements this;
`rfdiffusion.py` and `proteinmpnn.py` have no such branch and are stuck in the
server's own environment.

### 5.2 Dispatch contract

One `EnvDispatcher` replaces the per-runner logic. Every tool call becomes:

1. Write inputs to a scratch directory shared across environments.
2. `micromamba run -n <env> <entry> <args>`.
3. Parse the declared output files.
4. Return JSON, with numpy scalars coerced (the bug fixed by `d35e50d` on the
   old checkout — that commit is not on `origin/main` and its fix must be
   reimplemented here).

Environments are defined declaratively in `pixi.toml` with a committed
`pixi.lock`, one `[environments]` entry per engine. Installing from a lockfile
skips channel resolution at build time and makes the image reproducible.

### 5.3 Hardware

Target: NVIDIA L40S 46 GB (Ada, sm_89), 8 available.

| Tool | Fits 46 GB | Constraint |
|---|---|---|
| Chai-1 | yes | L40S is a documented supported SKU |
| Protenix | yes, wide margin | ~7-9 GB at 600 tokens; ceiling ~1300-1600 tokens |
| OpenFold3 | yes | **cap `num_diffusion_samples` at 5** — issue #71 reports a 75.94 GiB allocation on a 48 GB card at 100 samples |
| BoltzGen | yes | confirmed working by the repository owner; budget >=64 GB host RAM |
| ESMFold2 | probably | bf16 weights ~13 GB; O(L²) activations. Estimated ceiling ~1000-1100 residues — **unverified, needs measurement** |
| RF3 | probably | no published VRAM figure anywhere |

Each manifest declares a `max_residues` guard and every runner converts CUDA OOM
into an error naming the offending length and the parameter to lower.

### 5.4 License-constrained assets

Three assets cannot be baked into a distributable image:

| Asset | Constraint | Handling |
|---|---|---|
| PyRosetta | redistribution prohibited; non-commercial license is automatic since the 2024 Rosetta repo transition | Image creates an empty `rosetta` env; entrypoint installs from the quarterly wheel index on first run under the operator's own license. `run_rosetta_interface` is unregistered when absent. |
| AlphaFold3 weights | non-commercial, explicit no-share clause | Bind-mount only. Not present on this machine today. `run_alphafold3` unregistered when the mount is missing. |
| FoldX | registration + signed agreement | Not included. |

`requires.license_gated: true` in a manifest drives this: the tool is registered
only when its asset is present, and `describe_tool` explains how to supply it.

## 6. Harness and model compatibility

The server uses the official `mcp` SDK 1.25.0 low-level `Server` API with
protocol `2025-11-25`. This is the portable baseline. Four changes:

**Transport.** `run_server()` wires only `stdio_server()`. The SDK also ships
`sse`, `streamable_http` and `websocket`. Since every tool needs a GPU, the
natural deployment is the server on the L40S host with clients connecting
remotely, which stdio cannot do without an SSH tunnel. Add
`--transport {stdio,http}`, defaulting to stdio.

**Server-side validation.** JSON Schema support varies by model: Gemini's
function calling accepts a restricted OpenAPI subset and does not enforce
`pattern`; OpenAI strict mode requires `additionalProperties: false` and every
property listed in `required`. Client-side validation is therefore a hint, not a
guarantee. Every manifest constraint is re-checked server-side, and a violation
returns an error that states the expected format and gives a correct example.

**Defaults.** Schema `default` is documentation; most models omit the key
entirely. The dispatcher applies manifest defaults before invoking the engine.

**Version pin.** `mcp>=0.1.0` in `pyproject.toml:38` is effectively unpinned.
Change to `mcp>=1.25,<2`.

### Tool count

29 tools is more than the ~20 that provider guidance has typically suggested.
The evidence that large tool counts degrade selection is real in direction but
not quantified for this setting, and the dominant factor reported across tool-use
benchmarks is **confusability between similar tools**, not raw count. Eight
co-folding tools with near-identical signatures are the actual risk, not the
number 29.

Three mitigations, in order of importance:

1. Every `doc` carries "When to use this instead of the alternatives", naming
   siblings and selection conditions (§4.3).
2. `describe_tool(category=...)` returns a comparison table for a whole
   category.
3. `PROFILE` environment variable, mirroring the existing `DEVICE` gate:
   `PROFILE=full` (default) registers everything; `PROFILE=core` registers a
   9-tool subset for interactive work.

## 7. Known unknowns

Carried forward deliberately; each needs verification during implementation.

| Item | Risk |
|---|---|
| RF3 output schema | `_confidences.csv` column names and `_summary_confidences.json` keys are undocumented and the maintainers call the format "currently finalizing". Must introspect a real run; pin the rc-foundry version. |
| ESMFold2 on 46 GB | The ~1000-residue ceiling is derived from a third-party H100 report, not measured. Also a claimed ~26 GiB fp32 materialisation during load that bf16 may not avoid on the `esm` path. |
| Protenix default checkpoint | Unverified whether `protenix pred` without `-n` pulls proprietary v2 weights in 2.0.0. Must hard-code v1 and never let a caller pass a model name. |
| RFdiffusion weights license | Code is BSD-3 but RosettaCommons/RFdiffusion#135 indicates the weights' status was historically ambiguous. Read the LICENSE before redistributing. |
| BoltzGen / Promera seeds | Neither exposes a `seed` parameter in its documented CLI. Per-request reproducibility may not be achievable; if not, the manifest must say so rather than accept a `seed` it silently ignores. |
| OpenFold3 CLI flags | README uses underscores, one readthedocs page renders hyphens. Verify with `--help` on first run. |

## 8. Migration

Breaking change; v1.0.0 tool names are not preserved.

| Removed | Replacement |
|---|---|
| `design_binder` | `run_proteina_complexa_generate` or `run_boltzgen_design`, then `run_mpnn`, then a co-folding tool |
| `design_fold` | `run_genie2` / `run_rfdiffusion_binder` → `run_mpnn` → co-folding |
| `optimize_sequence`, `validate_design`, `suggest_hotspots`, `rosetta_design` | removed; callers compose atomistic tools |
| `predict_affinity_boltz` | removed — see §1 |
| `generate_backbone` | `run_rfdiffusion_binder` |
| `design_sequence` | `run_mpnn` |
| `predict_structure`, `predict_complex`, `predict_structure_boltz` | `run_esmfold2` / `run_chai1` / `run_boltz` |
| `rosetta_interface_score` | `run_rosetta_interface` |
| `analyze_interface` | `run_rosetta_interface` (buried surface area, hbonds, shape complementarity) and `run_prodigy` (absolute ΔG estimate) |
| `score_stability` | `run_esm_score` |
| `energy_minimize` | `run_openmm_minimize` |
| `get_design_status` | renamed `get_job_status` |

Tests for removed tools (`test_design_binder.py`, `test_validate_design.py`,
`test_optimize.py`, `test_hotspots.py`) are deleted. Tests covering retained
pipelines are kept and re-pointed.

## 9. Build order

Each step is test-first per the repository's TDD workflow.

1. **Manifest loader and registry.** Schema for the manifest format, loader,
   `composite`/`requires` filtering, derived `Tool` construction, dispatch
   registry. No engines yet — proves a composite tool is unregistrable and
   uncallable.
2. **Server-side validation, defaults, `describe_tool`, transport flag, SDK pin.**
   Harness- and model-compatibility work from §6, independent of any engine.
3. **`EnvDispatcher`** with the scratch-directory contract and numpy-safe JSON
   serialisation. Verified against one already-working engine (`run_mpnn`).
4. **`pixi.toml` + `pixi.lock`**, one environment per engine, then the Dockerfile
   built from the lock.
5. **Engines, easiest first:** `run_mpnn` → `run_chai1` → `run_boltz` →
   `run_prodigy` / `run_ipsae` → `run_proteina_complexa_*` →
   `run_boltzgen_*` → `run_esmfold2` → `run_protenix` → `run_openfold3` →
   remainder. Each engine lands with its manifest, its generated document and
   an integration test.
6. **Generated `docs/tools/`** and a README rewrite.
