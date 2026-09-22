# protein-design-mcp — full tool list

The list agreed in `docs/superpowers/specs/2026-09-21-atomistic-tool-refresh-design.md`
§3.2, **amended 2026-09-22**: the spec's 29 tools + 2 meta, plus two Genie 3 tools added
below (Genie 3 does binder design and motif scaffolding, which the spec did not cover) —
**31 tools + 2 meta**. Status checked against this host on 2026-09-22, including
a GPU survey that actually ran the engines on **GPU 7 only** (0 MiB before/during/after
on every other index).

Every tool is `run_<engine>`. Composite pipelines are deliberately unreachable —
verified across `resolve()`, `call_tool()` and `describe_tool()`.

**Legend**
✅ shipped as an MCP tool, live-verified · 🟢 engine executed successfully on GPU 7,
wrapper not written · 🟡 engine present, invocation confirmed but job not executed ·
🟠 partially working · ⬜ not on this host

---

## A. Target-conditioned binder generation (11)

| Tool | Engine | Status | Invocation / blocker |
|---|---|---|---|
| `run_proteina_complexa_generate` | Proteina-Complexa 160M | 🟡 | `complexa generate configs/search_binder_local_pipeline.yaml ++generation.task_name=<T>`. CLI + validate verified, checkpoints present; job not run (default search is not small) |
| `run_proteina_complexa_filter` | ″ | 🟡 | reward-model ranking — where test-time compute goes |
| `run_proteina_complexa_evaluate` | ″ | 🟡 | refolding metrics |
| `run_proteina_complexa_analyze` | ″ | 🟡 | aggregate analysis over a run |
| `run_rfdiffusion2` | RFdiffusion2 | 🟡 | Official path needs **apptainer, which is not installed** (`.sif` present, 13.6G). Workaround verified: `PYTHONPATH=<repo> python rf_diffusion/benchmark/pipeline.py --config-name=...` in env `rfd2_src` |
| `run_protpardelle` | Protpardelle-1c | 🟢 | `python -m protpardelle.sample <yaml> --num-mpnn-seqs 0` (env `pp1c`, editable → `~/projects/protpardelle-1c/src`) |
| `run_genie3_binder` | Genie 3 | 🟡 | `~/projects/genie3`, `scripts/problem/binder_design/`. Imports under the `genie2` env; **checkpoints not downloaded** (`assets/` is 6 MB, a gif) |
| `run_rfdiffusion_binder` | RFdiffusion | 🟡 | env `SE3nv` has `rfdiffusion 1.1.0`; not exercised in this survey |
| `run_boltzgen_design` | BoltzGen 0.3.2 | ⬜ | **not on this host** — MIT code+weights+data |
| `run_boltzgen_inverse_fold` | ″ | ⬜ | ″ — BoltzGen's own IF head, not ProteinMPNN |
| `run_boltzgen_filter` | ″ | ⬜ | ″ — CPU-only, re-rankable without regenerating |

## B. Monomer / scaffold generation (5)

| Tool | Engine | Status | Invocation / blocker |
|---|---|---|---|
| `run_genie3_scaffold` | Genie 3 | 🟡 | Motif scaffolding. Same checkout and env as `run_genie3_binder`; checkpoints still needed |
| `run_genie2` | Genie2 | 🟢 | `python genie/sample_unconditional.py --name base --epoch 40 --scale 0.6 --outdir DIR`. Unconditional only — `sample_scaffold.py`/`sample_unconditional.py` are its whole surface. Genie 3 supersedes it once weights land |
| `run_frameflow` | FrameFlow | 🟢 | `PYTHONPATH=<repo> python experiments/inference_se3_flows.py -cn inference_unconditional`. ⚠ writes to cwd-relative `./inference_outputs/` and **ignores `inference.output_dir`** |
| `run_multiflow` | MultiFlow | 🟠 | Core flow-matching + ProteinMPNN codesign **ran and wrote output**. Built-in ESMFold self-consistency scoring is blocked — `deepspeed` missing from the env |
| `run_la_proteina` | La-Proteina | 🟡 | `~/projects/la-proteina`; checkpoints not yet downloaded |

## C. Sequence design (1)

| Tool | Engine | Status | Notes |
|---|---|---|---|
| `run_mpnn` | **ProteinMPNN** / LigandMPNN / SolubleMPNN | ✅ | Live: `num_designs: 2`. Serves ProteinMPNN / LigandMPNN / SolubleMPNN via `model_type`. Survey also found a second checkout at `proteina-complexa/community_models/LigandMPNN` — the shipped tool uses the pip build, not this one |

## D. Co-folding / structure prediction (8)

| Tool | Engine | Status | Invocation / blocker |
|---|---|---|---|
| `run_boltz` | Boltz-2 2.2.1 | 🟢 | `boltz predict in.yaml --out_dir DIR`. ⚠ editable → **your fork** `~/projects/lightning-boltz-dev`. Affinity head not exposed |
| `run_esmfold2` | ESMFold2 | 🟢 | Python API `EsmFold2Model.from_pretrained("biohub/ESMFold2")`. ⚠ **package shadowing**: plain `import esm` picks up the wrong `fair-esm 2.0.0` from `~/.local` instead of the env's real `esm 3.4.0`. Requires `PYTHONNOUSERSITE=1` |
| `run_alphafold3` | AF3 (MMseqs2-GPU fork) | 🟡 | `~/projects/af3-mmseqs-gpu` + `/opt/alphafold3_data`. Own Docker recipe. ⚠ `models/` is **empty** despite the name; weights are at `weights/af3.bin` (1.1G). Non-commercial, no redistribution |
| `run_chai1` | Chai-1 0.6.1 | ⬜ | **not on host** — Apache-2.0 code+weights. L40S is a supported SKU; lowest integration risk of the missing folders |
| `run_protenix` | Protenix v1 | ⬜ | **not on host** — Apache-2.0. Pin v1; v2 weights proprietary |
| `run_openfold3` | OpenFold3 | ⬜ | **not on host** — Apache-2.0 code+weights+data |
| `run_rf3` | RoseTTAFold3 | ⬜ | **not on host** (env `rf` holds only `se3_transformer`). Output schema unstable |
| `run_alphafold2_multimer` | AF2-Multimer / ColabFold | ⬜ | not confirmed. Env `BindCraft` carries `jax` and may bundle ColabFold — needs checking. Still the reference ipTM discriminator |

## E. Scoring and analysis (6)

| Tool | Engine | Status | Notes |
|---|---|---|---|
| `run_prodigy` | PRODIGY | ✅ | Live: `binding_affinity_kcal_per_mol: -11.2`. CPU, milliseconds |
| `run_ipsae` | ipSAE | ✅ | Live: `ipsae: 0.056897`. Works from any predictor's PAE |
| `run_openmm_minimize` | OpenMM 8.6 | ✅ | Live: `energy_change_kj_mol: -1248779.7`. ⚠ pdbfixer installed but unused (G3) |
| `run_rosetta_interface` | PyRosetta InterfaceAnalyzer | 🟡 | `/opt/pyrosetta_wheels` — one wheel, 1.8G, **cp312 only** (constrains the env's python). Redistribution prohibited: mount at runtime |
| `run_esm_score` | ESM2-650M / ESM-C | 🟡 | env `esm_env` (`esm` + torch 2.4). Developability proxy, **not** a binding predictor |
| `run_promera_score` | Promera | ⬜ | **not on host** — MIT. iCS + ipSAE. Its Design task is composite and excluded |

## F. Meta (2)

| Tool | Status | Notes |
|---|---|---|
| `describe_tool` | ✅ | Returns a tool's document, or lists a category. Two defects open (M1, M2) |
| `get_job_status` | 🟡 | Logic preserved at `src/protein_design_mcp/job_status.py`, 6 tests passing, not wired as a tool. Its time estimator hardcodes the old pipeline's steps and must be replaced |

---

## Tally

| | Count |
|---|---|
| ✅ Shipped as MCP tools, live-verified | **4** + `describe_tool` |
| 🟢 Executed successfully on GPU 7 | **4** |
| 🟡 Present, invocation known, job not run | **11** |
| 🟠 Partially working | **1** |
| ⬜ Needs installing | **9** |
| | **31 + 2 meta** |

**22 of 31 are reachable with what is already on this box**, and 8 of those have now been
run or had their exact invocation confirmed. Of the 9 missing, 7 are permissively
licensed installs (BoltzGen MIT; Chai-1, Protenix, OpenFold3 Apache-2.0; Promera MIT) —
a fetch, not a blocker. RF3 and AF2-Multimer need a decision: RF3's output schema is
unstable, and AF2-Multimer may already be reachable through `BindCraft`'s ColabFold.

## Environment traps found while surveying

These cost real time to find and will cost more to rediscover:

1. **ESMFold2 package shadowing.** `import esm` silently resolves to the wrong
   `fair-esm 2.0.0` in `~/.local` instead of the env's `esm 3.4.0`. Needs
   `PYTHONNOUSERSITE=1`. User-site is shared by every python 3.12 env on this host.
2. **FrameFlow ignores its own output-dir setting** and writes to cwd-relative
   `./inference_outputs/`. Our dispatcher runs each engine in a scratch workdir as cwd,
   so this is contained — but only because of that.
3. **RFdiffusion2's official path needs apptainer**, which is not installed.
4. **MultiFlow's scoring step needs `deepspeed`**, which is missing.
5. **`/opt/alphafold3_data/models/` is empty** despite the name — weights are at
   `weights/af3.bin`.
6. **The PyRosetta wheel is cp312 only.**
7. **Half the engine envs are editable installs** whose source lives in `~/projects`.
   Mounting the env alone gives `ModuleNotFoundError` while python still runs.

## Substrate — proven 2026-09-22

- GPU pinned at the container boundary: `--device nvidia.com/gpu=7` → the container
  sees exactly one L40S. **Only GPU 7 is used.**
- Host conda envs mounted read-only at their *identical* paths (absolute shebangs).
- Editable installs need their source checkout mounted too.
- `micromamba run -p <prefix>` reaches a mounted env; `-n <name>` does not.

Design: `docs/superpowers/specs/2026-09-22-gpu-engine-substrate-design.md`

## Excluded by design

BindCraft, FreeBindCraft, ColabDesign, BoltzDesign1, mosaic, Odin-Multi, EasyNano,
dl_binder_design, RFantibody — gradient-based hallucination through a folding model;
success depends on operator judgement the caller cannot exercise. `complexa design`,
`boltzgen run`, Promera's Design task — orchestrators over steps exposed individually.
Boltz-2 affinity head — protein–ligand only. Protenix v2, SeedProteo, Chai-2,
AlphaProteo, Pearl, Proteina (original), Chroma — no public/permissive weights.

## Open defects

| Ref | Issue |
|---|---|
| **C9** | HTTP transport has no auth while every tool takes a caller-supplied path. Safe at the `127.0.0.1` default; **needs a decision before `--host 0.0.0.0`** |
| **G1** | One malformed manifest removed every tool. Isolation fixed; the reason a model receives is still "unknown tool" — fix round in progress |
| **G2** | `run.outputs` discarded when `parse_output` raises — costly for hour-long GPU jobs |
| **G3** | pdbfixer installed but unused, so `run_openmm_minimize` fails on any PDB with HETATM/waters |
| **G4** | `run_prodigy` omits `timeout_s` |
| **M1** | `describe_tool` returns `isError=False` while carrying an error |
| **M2** | 4 of 6 advertised `category` values error; `meta` permanently excludes `describe_tool` from its own listing |
| **M3** | **ProteinMPNN is effectively undiscoverable.** It is the default of `run_mpnn` (`model_type` enum `protein`/`soluble`/`ligand`, default `protein`), but the enum values never name their engine, `model_type`'s description is only "Which trained variant to use.", and `docs/tools/run_mpnn.md` says "ProteinMPNN" exactly once. On a protein–protein binder-design server the standard sequence-design step must be findable by name. Fix by naming the engine behind each enum value and in the doc — **not** by splitting into three tools; it is one codebase and the spec's one-tool argument still holds |
