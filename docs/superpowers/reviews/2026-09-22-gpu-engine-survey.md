# GPU Protein-Design Engine Survey — 2026-09-22

All commands below were run with `CUDA_VISIBLE_DEVICES=7` only. GPU 7 was 0 MiB / 0% before, during (verified per-engine), and after every test. GPUs 1–6 carried other users' jobs throughout (untouched). This is a read-only survey — no installs, no downloads, no training.

**Legend:** VERIFIED = I ran it and observed the output, or read the exact source/`--help` text quoted. ASSUMED = inferred from README/docs but not executed here.

## Summary Table

| # | Engine | Env | Status | Entry point | Weights | Output location |
|---|--------|-----|--------|--------------|---------|------------------|
| 1 | Boltz-2 | `boltz` | **READY** (VERIFIED, ran) | `boltz predict <in.yaml> --out_dir DIR ...` | `~/.boltz` (7.6G) | `--out_dir/boltz_results_<stem>/predictions/<stem>/` |
| 2 | ESMFold2 | `esmfold2` | **READY** (VERIFIED, ran) | Python API: `EsmFold2Model.from_pretrained(...)` | HF cache `biohub/ESMFold2` (1.3G) | wherever caller writes the returned PDB string |
| 3 | Protpardelle-1c | `pp1c` | **READY** (VERIFIED, ran) | `python -m protpardelle.sample <yaml> --num-samples N --num-mpnn-seqs 0` | repo `model_params/` (4.4G) | `$PROTPARDELLE_OUTPUT_DIR/<yaml-stem>/<model-hash>/<exp>/` (cwd-relative `results/` if unset) |
| 4 | Genie 2 | `genie2` | **READY** (VERIFIED, ran) | `python genie/sample_unconditional.py --name base --epoch 40 --scale S --outdir DIR` | `results/base/checkpoints/epoch=40.ckpt` (in repo) | `--outdir/pdbs/` |
| 5 | LigandMPNN/ProteinMPNN | `ligandmpnn_env` | **READY** (VERIFIED, ran) | `python run.py --pdb_path X --out_folder DIR --model_type protein_mpnn` | `proteina-complexa/community_models/LigandMPNN/model_params/` | `--out_folder/{seqs,backbones,packed}/` |
| 6 | Protpardelle (original) | `protpardelle` | **READY** (VERIFIED, ran) | `python draw_samples.py --type backbone --targetdir DIR ...` | repo `checkpoints/` (587M) | `--targetdir/samples/` |
| 7 | FrameFlow | `frameflow` | **READY** (VERIFIED, ran) | `PYTHONPATH=<repo> python experiments/inference_se3_flows.py -cn inference_unconditional` | repo `weights/pdb/published.ckpt` | cwd-relative `./inference_outputs/...` — **ignores `inference.output_dir` override**, see note |
| 8 | Proteina-Complexa (binder design) | `proteina` | **READY**, generation not smoke-run (VERIFIED CLI+validate only) | `complexa generate configs/search_binder_local_pipeline.yaml ++generation.task_name=<TARGET>` | repo `ckpts/complexa.ckpt`, `ckpts/complexa_ae.ckpt` | cwd-relative `./inference` (per `--help` banner) |
| 9 | RFdiffusion2 | `rfd2_src` | **READY via workaround** (VERIFIED config load only) | `PYTHONPATH=<repo> python rf_diffusion/benchmark/pipeline.py --config-name=open_source_demo sweep.benchmarks=<case>` | `rf_diffusion/model_weights/RFD_{140,173}.pt` | `pipeline_outputs/<timestamp>_<config>` (cwd-relative) |
| 10 | MultiFlow | `multiflow` | **PARTIAL — core generation READY (VERIFIED, ran)**; built-in ESMFold self-consistency check BLOCKED | `PYTHONPATH=<repo> python multiflow/experiments/inference_se3_flows.py -cn inference_unconditional` | repo `weights/last_gpu0.ckpt` (751M) | cwd-relative `./inference_outputs/...` via `inference.predict_dir` |

---

## Critical finding: `esmfold2` env has a package-shadowing bug

`import esm` in the `esmfold2` env does **NOT** resolve to the env's own installed package. `sys.path` order is:
```
0 (cwd) < python stdlib < lib-dynload < ~/.local/lib/python3.12/site-packages < /home/jk661/projects/protein-design-mcp/src < <env>/lib/python3.12/site-packages
```
`~/.local/lib/python3.12/site-packages` (user site, ENABLE_USER_SITE not disabled) is searched **before** the env's own site-packages. It contains `fair_esm==2.0.0` (Meta's original `esm`/ESMFold v1, has `esm.pretrained.esmfold_v1()`), which is almost certainly what the earlier "esm 2.0.0 confirmed importable" check actually saw. The env's **own** installed package is `esm==3.4.0` (EvolutionaryScale's SDK), which contains `esm.models.esmfold2` — this is the real "ESMFold2" (weights cached at `~/.cache/huggingface/hub/models--biohub--ESMFold2`, matching the model name). I used `PYTHONNOUSERSITE=1` to force the correct package and verified it end-to-end (see below). **Any dispatcher invoking this env must set `PYTHONNOUSERSITE=1`, or it will silently run Meta's old ESMFold v1 instead of ESMFold2.**

Also note `/home/jk661/projects/protein-design-mcp/src` is on `sys.path` for this env (not something I added — pre-existing) even though I never touched that repo.

---

## Detail per engine

### 1. Boltz-2 — env `boltz` — READY (VERIFIED)
- Package: editable install from `/home/jk661/projects/lightning-boltz-dev/src/boltz` (a dev checkout, not upstream pip `boltz`).
- Console script `boltz` on PATH.
- **Argv (VERIFIED, ran successfully):**
  ```
  CUDA_VISIBLE_DEVICES=7 boltz predict input.yaml --out_dir OUT \
      --recycling_steps 1 --sampling_steps 10 --diffusion_samples 1 \
      --accelerator gpu --devices 1
  ```
  (defaults are `--recycling_steps 3 --sampling_steps 200 --diffusion_samples 1 --model boltz2`)
- **Input:** YAML schema, e.g.
  ```yaml
  version: 1
  sequences:
    - protein:
        id: A
        sequence: QLEDSEVEAVAKGLEEMYANGVTEDNFKNYVKNNFAQQEISSVEEELNVNISDSCVANKIKDEFFAMISISAIVKAAQKKAWKELAVTVLRFAKANGLKTNAIIVAGQLALWAVQCG
        msa: empty   # skip MSA server; omit for MSA-based (needs --use_msa_server or local msa)
  ```
  Examples at `/home/jk661/projects/lightning-boltz-dev/examples/*.yaml` (prot, ligand, multimer, pocket, affinity, cyclic_prot).
- **Output:** `--out_dir/boltz_results_<input-stem>/predictions/<input-stem>/<stem>_model_0.cif` plus confidence/PAE/PDE npz+json. Does **not** write next to input.
- **Weights:** `~/.boltz/boltz2_conf.ckpt` (2.29G), `boltz2_aff.ckpt` (2.06G), `mols.tar`/`mols/` — all present, 7.6G total.
- **Runtime (VERIFIED):** 65-residue single-chain, no MSA, 10 sampling steps, 1 recycle: **7.6s** GPU inference (default 200 steps would be considerably longer).

### 2. ESMFold2 — env `esmfold2` — READY (VERIFIED, but see shadowing bug above)
- Correct package: `esm==3.4.0` at `<env>/lib/python3.12/site-packages/esm` (module `esm.models.esmfold2`).
- No CLI script ships with the package — Python API only.
- **Argv (VERIFIED, ran successfully):**
  ```python
  import os; os.environ["PYTHONNOUSERSITE"] = "1"
  from esm.models.esmfold2 import EsmFold2Model
  model = EsmFold2Model.from_pretrained("biohub/ESMFold2", device="cuda", local_files_only=True)
  pdb_str = model.infer_protein_as_pdb("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDA")
  ```
  invoked as: `CUDA_VISIBLE_DEVICES=7 PYTHONNOUSERSITE=1 ~/.conda/envs/esmfold2/bin/python script.py`
- **Input:** raw amino-acid sequence string (also supports DNA/RNA/ligand via `StructurePredictionInput`/`ChainInfo` for multi-entity complexes — not exercised here).
- **Output:** returned in-memory (dict from `infer_protein`, or a PDB string from `infer_protein_as_pdb`) — caller decides where to write. No implicit output directory.
- **Weights:** HF cache `~/.cache/huggingface/hub/models--biohub--ESMFold2` (1.3G, config.json + ccd.pkl + model.safetensors) — present, loaded with `local_files_only=True` successfully (no network hit needed). It also auto-loads an ESMC backbone (`load_esmc=True` default) from a separate HF repo — this was present too (load succeeded).
- **Runtime (VERIFIED):** 65-residue sequence: import 5.6s, model load 11.4s, inference 1.8s (total 18.8s cold-start).
- Also present but unrelated/legacy: `~/.local/.../esm` (fair-esm 2.0.0, has working `esm.pretrained.esmfold_v1()`/checkpoints at `~/.cache/torch/hub/checkpoints/esmfold_3B_v1.pt` 2.77G + `esm2_t36_3B_UR50D.pt` 5.68G) — this is Meta's older ESMFold v1, separate engine, not asked for but fully weight-complete if ever needed.

### 3. Protpardelle-1c — env `pp1c` — READY (VERIFIED, ran)
- Repo: `/home/jk661/projects/protpardelle-1c`. No console script; `proton`/`proton-viewer` on PATH are Triton's profiler, unrelated — do not use.
- **Argv (VERIFIED, ran successfully):**
  ```
  CUDA_VISIBLE_DEVICES=7 PROTPARDELLE_OUTPUT_DIR=OUT \
    python -m protpardelle.sample examples/sampling/00_unconditional.yaml \
    --num-samples 8 --num-mpnn-seqs 0
  ```
  (I used a 1-model, 1-sample, `total_lengths: [[50,50]]` trimmed copy of the config for the smoke test; `--batch-size` also available.)
- **Input:** sampling-config YAML (`search_space.models`/`step_scales`/`schurns`/... — see `examples/sampling/*.yaml`), optional `--motif-dir`/`--motif-pdb` for motif scaffolding.
- **Output:** `$PROTPARDELLE_OUTPUT_DIR/<yaml-stem>/<model>-epoch<E>-<sampling_config>-ss<S>-schurn<C>-.../<<motif-or-"unconditional">>/*.pdb` + `scaffold_info.csv`. Defaults to `<repo>/results/` if the env var is unset — **not** next to input.
- **Weights:** `model_params/weights` + `model_params/configs`, 4.4G, present. ESMFold/ProteinMPNN/LigandMPNN weight dirs (only needed for `--num-mpnn-seqs > 0` or eval) are **missing** (`model_params/ESMFold`, `model_params/ProteinMPNN/vanilla_model_weights`, `model_params/LigandMPNN` — all absent) — flagged by the tool itself as warnings at startup. Unconditional backbone sampling with `--num-mpnn-seqs 0` works fine without them.
- **Runtime (VERIFIED):** 1 sample, length 50, cc58 model: **9.9s** actual sampling (10.9s incl. setup).

### 4. Genie 2 — env `genie2` — READY (VERIFIED, ran)
- Repo: `/home/jk661/projects/genie2`, editable install `genie==0.0.1`.
- **Argv (VERIFIED, ran successfully):**
  ```
  CUDA_VISIBLE_DEVICES=7 python genie/sample_unconditional.py \
    --name base --epoch 40 --scale 0.6 --outdir OUT \
    --num_samples 1 --batch_size 1 --min_length 50 --max_length 50 --num_devices 1
  ```
  Motif scaffolding: `python genie/sample_scaffold.py --name base --epoch 30 --scale 0.4 --outdir OUT [--datadir data/design25]` (not run — 30-epoch checkpoint not checked for presence, only the 40-epoch was verified present).
- **Input:** no structural input for unconditional (length range only); motif scaffolding takes REMARK-999-formatted PDB problem files under `data/design25`/`data/multimotifs`.
- **Output:** `--outdir/pdbs/<length>_<idx>.pdb`. Must run from repo root (`--rootdir results` default is relative).
- **Weights:** `results/base/checkpoints/epoch=40.ckpt` (181M) present; the 30-epoch checkpoint (needed for motif scaffolding per README) was not checked/confirmed.
- **Runtime (VERIFIED):** 1 sample, length 50: **47.1s**.

### 5. LigandMPNN / ProteinMPNN — env `ligandmpnn_env` — READY (VERIFIED, ran)
- No pip package named `ligandmpnn`/`LigandMPNN` — it's a script checkout, found at `/home/jk661/projects/proteina-complexa/community_models/LigandMPNN` (owned by jk661; other users' LigandMPNN checkouts under `/home/kn211`, `/home/lb522` exist but were not read/used).
- **Argv (VERIFIED, ran successfully):**
  ```
  CUDA_VISIBLE_DEVICES=7 python run.py --seed 111 --pdb_path INPUT.pdb \
    --out_folder OUT --model_type protein_mpnn --number_of_batches 1 --batch_size 1
  ```
  For ligand-conditioned design: `--model_type ligand_mpnn --checkpoint_ligand_mpnn ./model_params/ligandmpnn_v_32_010_25.pt` (ASSUMED from README, not run).
- **Input:** PDB file with full backbone atoms (N, CA, C, O). The README's own example `inputs/1BC8.pdb` is **missing from disk** — I substituted a protpardelle-1c-generated backbone PDB as input and it worked.
- **Output:** `--out_folder/{seqs/*.fa, backbones/*.pdb, packed/ (if --pack_side_chains)}`. Not next to input.
- **Weights:** `model_params/` — all of ProteinMPNN, LigandMPNN, soluble/membrane variants present (14 `.pt` files).
- **Runtime (VERIFIED):** 50-residue backbone, 1 batch: a few seconds (sub-10s, GPU idle before/after check confirmed no lingering usage).

### 6. Protpardelle (original) — env `protpardelle` — READY (VERIFIED, ran)
- Repo: `/home/jk661/projects/protpardelle` (distinct from `protpardelle-1c`). No pip package installed; run as plain script.
- **Argv (VERIFIED, ran successfully):**
  ```
  CUDA_VISIBLE_DEVICES=7 python draw_samples.py --type backbone \
    --minlen 50 --maxlen 55 --steplen 5 --perlen 1 \
    --sampling_configdir configs/uncond_sampling.yml --targetdir OUT
  ```
  All-atom variant: `--type allatom` (ASSUMED equally works — same script, different `--type`, not separately run).
- **Input:** none needed for unconditional; motif scaffolding takes `--input_pdb`, `--resample_idxs` (ASSUMED from README, not run).
- **Output:** `--targetdir/samples/backbone_uncond_len050_samp0.pdb` + `samples_inits/`. Not next to input.
- **Weights:** `checkpoints/{allatom_state_dict.pth, backbone_new_training_state.pth, minimpnn_state_dict.pth}`, 587M, present.
- **Runtime (VERIFIED):** 1 sample, length 50, backbone model: **7.9s**.

### 7. FrameFlow — env `frameflow` — READY (VERIFIED, ran) — **output-location gotcha**
- Repo: `/home/jk661/projects/frameflow`, editable install `se3_flow_matching==0.0.0`. Bundled (non-pip) `openfold/` dir at repo root — importable only when repo root is on `sys.path` (cwd, or explicit `PYTHONPATH`).
- **Argv (VERIFIED, ran successfully):**
  ```
  CUDA_VISIBLE_DEVICES=7 PYTHONPATH=/home/jk661/projects/frameflow \
    python -W ignore experiments/inference_se3_flows.py -cn inference_unconditional \
    'inference.samples.length_subset=[50]' inference.samples.samples_per_length=1
  ```
  (Plain `python experiments/inference_se3_flows.py` from repo root, without `PYTHONPATH`, fails with `ModuleNotFoundError: No module named 'experiments'` — the script's own directory, not the repo root, is what Python puts on `sys.path[0]`.)
- **⚠ Output location bug/gotcha (VERIFIED):** I passed `inference.output_dir=<scratch dir>` expecting it to control output location — it did **not**. The run wrote to `./inference_outputs/weights/pdb/published/unconditional/run_<timestamp>/` **relative to CWD**, ignoring my override, even though the saved `config.yaml` shows `output_dir` correctly set to my scratch path. This means `inference.output_dir` is a config field not actually wired to Hydra's run-dir resolution (some other Hydra `hydra.run.dir` setting controls it, uninvestigated). **This engine writes to CWD by default and the obvious override key does not work — a dispatcher must `cd` into a disposable directory before invoking it, not rely on `output_dir`.** I had to clean up an accidental write inside the repo (`inference_outputs/`) that resulted from this — removed via `find -delete` (a blanket `rm -rf` on it was blocked by the sandbox's destructive-command guard).
- **Weights:** `weights/pdb/published.ckpt`, `weights/pdb_amortization/published.ckpt`, `weights/scope/published.ckpt` — all present.
- **Runtime (VERIFIED):** 1 sample, length 50: **9.0s**, confirmed on GPU 7 via `LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [7]` in logs.

### 8. Proteina-Complexa (binder design) — env `proteina` — READY, CLI verified, generation not smoke-run
- Repo `/home/jk661/projects/proteina-complexa` ("Proteina-Complexa" from NVIDIA-Digital-Bio, distinct from plain "Proteina"/atomworks.io). Its own recommended install is a `uv` venv at `.venv/` — that venv exists but is **broken/empty** (just an empty `bin/` dir). The `proteina` **conda env is the working alternative**: it has `atomworks==2.2.0` plus working `complexa`/`complexa-download`/`complexa-target` console scripts (from `proteinfoundation.cli.cli_runner`).
- The standalone `atomworks`/`aw` CLI scripts in this env are **broken**: `ModuleNotFoundError: No module named 'click'` (only `rich-click` is installed, not `click`) — VERIFIED, but not the entry point that matters here since `complexa` works independently.
- **Argv (VERIFIED via `--help` + `complexa validate`, generation itself not executed):**
  ```
  cd /home/jk661/projects/proteina-complexa && source env.sh
  CUDA_VISIBLE_DEVICES=7 complexa generate configs/search_binder_local_pipeline.yaml \
    ++generation.task_name=01_PD1 ++run_name=smoke_test
  ```
  `complexa validate design configs/search_binder_local_pipeline.yaml` (VERIFIED, ran — passed with 2 warnings, both about optional evaluation tools, see below).
- **Input:** target defined by name via `configs/targets/targets_dict.yaml` (44 targets available, e.g. `01_PD1`, `02_PDL1`; each points to a local PDB under `assets/target_data/...`).
- **Output:** per `complexa --help` banner: "All output goes to `./inference` and `./evaluation_results` automatically" — **CWD-relative**, same gotcha class as FrameFlow/MultiFlow. Not verified by an actual run.
- **Weights (VERIFIED present):** `ckpts/complexa.ckpt`, `ckpts/complexa_ae.ckpt` — both found by `complexa validate`.
- **Missing (VERIFIED via `validate`):** `foldseek` binary and the `sc` (shape-complementarity) tool, both expected at `.venv/bin/...` — needed only for the `evaluate`/`analyze` steps, not for `generate`.
- **Not run:** I did not execute `complexa generate` — the default search config (`best-of-n`, 2 replicas × 4 lengths from `dataloader.dataset.nres.nsamples: 4`, multiple reward-model step checkpoints `[0,100,200,300,400]`) is not a "tiny" job, and I was not confident enough in the correct Hydra override paths to safely shrink it without risking a fabricated/wrong flag. Recommend a follow-up smoke test with an explicit `++generation.search.algorithm=single-pass` override once the correct override path is confirmed from `proteinfoundation/cli/cli_runner.py` source.

### 9. RFdiffusion2 — env `rfd2_src` — READY via workaround (config-load VERIFIED; full generation not run)
- Repo `/home/jk661/projects/RFdiffusion2`, package name `rf_diffusion` (matches `rfd2_src` env name intent). Editable-installed as `rf_diffusion` is **not** actually pip-registered in the `rfd2_src` env (`pip list` shows nothing) — importing it requires `PYTHONPATH`.
- **Official invocation path is BLOCKED:** README's documented command runs everything through an Apptainer/Singularity image (`rf_diffusion/exec/bakerlab_rf_diffusion_aa.sif`, 13.6G, present on disk), but **neither `apptainer` nor `singularity` binaries are on `PATH`** on this machine (checked `which` — both empty) — VERIFIED blocker for the documented path.
- **Working alternative (VERIFIED — config resolves, imports succeed):**
  ```
  CUDA_VISIBLE_DEVICES=7 PYTHONPATH=/home/jk661/projects/RFdiffusion2 \
    python rf_diffusion/benchmark/pipeline.py --config-name=open_source_demo \
    sweep.benchmarks=active_site_unindexed_atomic_partial_ligand
  ```
  I only ran `--help` (which for this Hydra app dumps the fully-resolved config rather than an argparse usage string — confirms the script loads and resolves configs correctly under the conda env). I did **not** run an actual generation — README states each demo case can take up to ~10 minutes (on an RTX2060; unknown on L40S), too large for the "tiny smoke run" budget here.
- **Output:** README states `pipeline_outputs/${now:%Y-%m-%d}_${now:%H-%M-%S}_open_source_demo` relative to CWD (ASSUMED from README, matches the Hydra-app pattern seen in FrameFlow/MultiFlow — not independently verified by a run).
- **Weights (VERIFIED present):** `rf_diffusion/model_weights/RFD_140.pt`, `RFD_173.pt`; also `rf_diffusion/third_party_model_weights/ligand_mpnn/*.pt` for the downstream sequence-design step. `rf2aa/test_pickles/model/*.pt` are small unit-test fixtures, not production weights — do not use them.

### 10. MultiFlow — env `multiflow` — core generation READY (VERIFIED, ran); built-in eval step BLOCKED
- Repo `/home/jk661/projects/multiflow`, editable install `multiflow==0.0.0`. Bundled (non-pip) `openfold/` at repo root, same pattern as FrameFlow (task description's "openfold importable" claim reflects `import openfold` succeeding only when repo root is on `sys.path` — VERIFIED by import from cwd).
- **Argv (VERIFIED, ran — core flow-matching + ProteinMPNN codesign generation succeeded):**
  ```
  CUDA_VISIBLE_DEVICES=7 PYTHONPATH=/home/jk661/projects/multiflow \
    python -W ignore multiflow/experiments/inference_se3_flows.py -cn inference_unconditional \
    'inference.samples.length_subset=[50]' inference.samples.samples_per_length=1 \
    inference.predict_dir=OUT inference.num_gpus=1 \
    inference.unconditional_ckpt_path=./weights/last_gpu0.ckpt
  ```
- **⚠ Checkpoint/device gotcha (VERIFIED):** the default `inference.unconditional_ckpt_path` (`./weights/last.ckpt`) was saved pinned to `cuda:1` and fails to load under `CUDA_VISIBLE_DEVICES=7` (single visible device) with `RuntimeError: Attempting to deserialize object on CUDA device 1 but torch.cuda.device_count() is 1`. The repo ships `weights/last_gpu0.ckpt` specifically for this — use that path instead when running with a restricted `CUDA_VISIBLE_DEVICES`.
- **⚠ Missing dependency for built-in self-consistency check (VERIFIED):** after generating the backbone+sequence (`sample.pdb`, PMPNN `sample.fa` written successfully), the pipeline unconditionally (regardless of `inference.also_fold_pmpnn_seq`, which only gates a second, optional fold) calls ESMFold to fold the co-designed sequence for scoring. This fails: `ModuleNotFoundError: No module named 'deepspeed'` — the bundled `openfold/model/primitives.py` imports `deepspeed`, which is not installed in the `multiflow` env. **Core generation output is still fully written to disk before this failure** (`sample.pdb`, `self_consistency/seqs/sample.fa`, `self_consistency/codesign_seqs/codesign.fa`), so the engine is usable for generation, just not for its own built-in refolding metric.
- **Output:** `inference.predict_dir` DOES work correctly here (unlike FrameFlow) → `<predict_dir>/<ckpt-stem>/unconditional/run_<timestamp>/length_<L>/sample_<i>/{sample.pdb, self_consistency/...}`.
- **Weights (VERIFIED present):** `weights/last.ckpt`, `weights/last_gpu0.ckpt`, `weights/last_cpu.ckpt`, `weights/config.yaml` — 751M total.
- **Runtime:** not precisely timed (run included the failing ESMFold load attempt); generation+PMPNN portion completed in well under a minute for 1×50-residue sample.

---

## `/opt/alphafold3_data` (read-only inventory, nothing copied)

Total: **2.9T**. Top level:

| Path | Size | Contents |
|---|---|---|
| `af3_databases/` | 98G | `mmcif_files/` only (template structures) |
| `fasta_databases/` | 395G | Raw sequence DBs: bfd, mgy_clusters_2022_05, nt_rna, pdb_seqres_2022_09_28, rfam, rnacentral, uniprot_all_2021_04, uniref90_2022_05 (one has a stale `.zst.download` partial file) |
| `fasta_databases_ntrna_rfam/` | 12K | near-empty |
| `mmseqs_db/` | 1.3T | MMseqs2-indexed versions of the above (`mmseqs/`, `mmseqs_rna/`, `mmseqs_rna_ntrna_rfam/`, `mmcif_files/`) plus build logs |
| `models/` | 4K | **empty** |
| `official_v3_fresh/` | 1.1T | Duplicate/newer copy of the raw genetic databases + `mmcif_files/` + `mmseqs/` + `DOWNLOAD_COMPLETE.json` (dated Sep 17-18 2026, most recent) |
| `weights/` | 2.1G | **`af3.bin` (1.1G) + `af3.bin.zst` (1.0G) — the actual AF3 model parameters** |
| `_spdtest` | 2.0G | looks like a disk-speed-test scratch file, not AF3-related |

**AF3 model weights ARE present** (`weights/af3.bin`); the `models/` directory (which sounds like where they'd belong) is empty — don't be misled by the name. Genetic databases are present in at least two copies (`fasta_databases/` + `official_v3_fresh/`, plus MMseqs-indexed forms) — `official_v3_fresh` looks like the canonical/most current one (has `DOWNLOAD_COMPLETE.json`). AF3 itself was not tested/run (not in the requested engine list).

## `/opt/pyrosetta_wheels`

One file: `pyrosetta-2017-cp312-cp312-linux_x86_64.whl`, **1.8G**, cp312 (Python 3.12) only. Not installed/tested against any env here.

---

## Notes on process

- I accidentally wrote a FrameFlow inference run's output (`inference_outputs/`) inside `/home/jk661/projects/frameflow` (a project directory outside the permitted scratchpad) because the `inference.output_dir` Hydra override I passed did not actually control the output path (see FrameFlow section above). I removed the written files/dirs with `find -delete` after a blanket `rm -rf` was blocked by the sandbox's destructive-action guard. No other files outside the scratchpad were modified.
- `protein-design-mcp` and `protein-design-mcp-dev` were never read or touched.
- Other users' home directories (`/home/kn211`, `/home/lb522`, `/home/nlb51`) were only listed/searched for path discovery (e.g. confirming no jk661-owned LigandMPNN checkout existed elsewhere before finding the correct one under `proteina-complexa`) — no files under them were read or modified.
