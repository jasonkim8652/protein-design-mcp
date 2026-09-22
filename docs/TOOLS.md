# protein-design-mcp — tools

**36 tools** (34 `run_*` + 2 meta). Classified by **function**, not by which engine a
tool came from.

Status: ✅ shipped · 🟢 ran on GPU 7 · 🟠 partial · 🟡 present, not yet run

MSA column: **free** = runs without an alignment · **opt** = caller's choice ·
**req** = cannot run without one

---

## 1. `binder_generation` — generate a binder against a target (7)

| Tool | Engine | Status |
|---|---|---|
| `run_proteina_complexa_generate` | Proteina-Complexa 160M | 🟡 |
| `run_boltzgen_design` | BoltzGen 0.3.2 | 🟢 |
| `run_rfdiffusion_binder` | RFdiffusion 1.1.0 | 🟡 |
| `run_rfdiffusion2` | RFdiffusion2 | 🟡 |
| `run_rfdiffusion3_binder` | RFdiffusion3 | 🟡 |
| `run_genie3_binder` | Genie 3 | 🟡 |
| `run_protpardelle` | Protpardelle-1c 1.3.2 | 🟢 |

## 2. `monomer_generation` — generate a monomer or scaffold (6)

| Tool | Engine | Status |
|---|---|---|
| `run_rfdiffusion3_scaffold` | RFdiffusion3 | 🟢 |
| `run_genie3_scaffold` | Genie 3 | 🟡 |
| `run_genie2` | Genie2 | 🟢 |
| `run_frameflow` | FrameFlow | 🟢 |
| `run_multiflow` | MultiFlow | 🟠 |
| `run_la_proteina` | La-Proteina | 🟡 |

## 3. `sequence_design` — design a sequence for a fixed backbone (2)

| Tool | Engine | Status |
|---|---|---|
| `run_mpnn` | ProteinMPNN / LigandMPNN / SolubleMPNN | ✅ |
| `run_boltzgen_inverse_fold` | BoltzGen's own IF head | 🟢 |

## 4. `structure_prediction` — predict a structure (9)

| Tool | Engine | MSA | Status |
|---|---|---|---|
| `run_esmfold2` | ESMFold2 | free | 🟢 |
| `run_chai1` | Chai-1 0.6.1 | opt | 🟢 |
| `run_boltz` | Boltz-2 2.2.1 | opt | 🟢 |
| `run_protenix` | Protenix v1 | opt | 🟢 |
| `run_openfold3` | OpenFold3 | opt | 🟢 |
| `run_promera` | Promera | opt | 🟢 |
| `run_rf3` | RoseTTAFold3 | **req** | 🟢 |
| `run_alphafold3` | AlphaFold 3 | **req** | 🟡 |
| `run_alphafold2_multimer` | AF2-Multimer / ColabFold | **req** | 🟢 |

## 5. `msa` — build an alignment (2)

| Tool | Engine | Status |
|---|---|---|
| `run_mmseqs_search` | MMseqs2 vs the 1.3 TB local DB | 🟡 |
| `run_colabfold_search` | `colabfold_search` vs UniRef30/envDB | 🟡 |

## 6. `scoring` — score an existing structure (4)

| Tool | Engine | Status |
|---|---|---|
| `run_ipsae` | ipSAE — works on any predictor's PAE | ✅ |
| `run_prodigy` | PRODIGY | ✅ |
| `run_rosetta_interface` | PyRosetta InterfaceAnalyzer | 🟡 |
| `run_esm_score` | ESM2-650M / ESM-C | 🟡 |

## 7. `run_analysis` — operate on a finished run's outputs (3)

| Tool | Engine | Runs a model | Status |
|---|---|---|---|
| `run_proteina_complexa_filter` | Proteina-Complexa | no | 🟡 |
| `run_boltzgen_filter` | BoltzGen | no | 🟢 |
| `run_proteina_complexa_analyze` | Proteina-Complexa | no (foldseek/mmseqs) | 🟡 |

## 8. `preparation` — modify a structure before scoring (1)

| Tool | Engine | Status |
|---|---|---|
| `run_openmm_minimize` | OpenMM 8.6 | ✅ |

## 9. `meta` (2)

| Tool | Status |
|---|---|
| `describe_tool` | ✅ |
| `get_job_status` | 🟡 |

---

## Parameter policy

Every engine knob is exposed. A parameter is hidden only if it selects a step this
server registers separately, or if it would let a caller escape the workdir.

### `msa` — always the caller's choice

Being able to build an alignment is not a reason to use one. Running MSA-free and
running with an MSA are **different experiments**, and comparing them is a legitimate
thing to want. So the parameter is tri-state on every tool that can accept one:

| value | meaning |
|---|---|
| `null` | run MSA-free, even though an alignment could have been supplied |
| `<path to a3m>` | use this alignment |
| absent | rejected — the choice must be stated, never inherited from a default |

Tools marked **req** reject `null` and say so in the error, naming what to run to get an
a3m. Tools marked **free** accept only `null`.

Each `msa` description names **which producer's a3m it expects**. MMseqs2 against
AlphaFold 3's databases and `colabfold_search` against UniRef30/envDB search different
sequence universes; a consumer built for one will often accept the other and quietly
return worse results rather than failing.

### `chains` — multimer or binder alone, never inferred

Whether a prediction runs with the target present or on the binder alone is the
caller's decision, and one of the most consequential it makes — a binder predicted
alone and the same binder predicted in complex are different experiments. No tool
infers this from the shape of its input.

### Every other knob

Sampling steps, diffusion timesteps, temperature, seed, sample count, beam width,
recycles, model variant — all exposed. Each description says what the parameter does,
what changes when it moves, and a sensible range. Defaults are documented as choices
with a reason, not stated as facts.

---

## Corrections made while building this

| Was | Now |
|---|---|
| `run_promera_score` in `scoring`, "duplicates `run_ipsae`" | **Wrong.** Its input is a target-schema directory plus `msa_dir` — it co-folds. Its iCS/ipSAE are its own confidence in its own prediction, like Chai's iptm. Moved to `structure_prediction`, renamed `run_promera`. |
| `run_openmm_minimize` category `scoring` | It relaxes a structure and scores nothing. Moved to `preparation`. |
| `run_boltzgen_inverse_fold` in generation | It is inverse folding. Moved to `sequence_design`, which looked like it held one tool. |
| Filters and analyze in generation | They run no model. Moved to `run_analysis`. |
| `run_rfdiffusion3` as one tool | Split into `_binder` / `_scaffold` — different inputs, and merging would hide binder design inside a JSON field. |
| `cofolding` | Renamed `structure_prediction`; half of them also fold monomers. |
| No MSA tool at all | Three tools required an alignment nothing could produce. Added category 5. |
| `run_proteina_complexa_evaluate` | **Removed** — composite over capabilities exposed here individually, and it hid a `binder is last chain` convention. |

## Excluded by design

BindCraft, FreeBindCraft, ColabDesign, BoltzDesign1, mosaic, Odin-Multi, EasyNano,
dl_binder_design, RFantibody — gradient-based hallucination through a folding model.
`complexa design`, `boltzgen run`, Promera's `Design` task — orchestrators over steps
exposed here individually. Boltz-2's affinity head — protein–ligand only. Protenix v2,
SeedProteo, Chai-2, AlphaProteo, Pearl, Proteina (original), Chroma — no public or
permissive weights.

Remote MSA servers are off by default: they transmit the caller's sequence to a third
party, and here those sequences are usually novel designs.
