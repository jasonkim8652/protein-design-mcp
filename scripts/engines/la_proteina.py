"""Wrapper for La-Proteina's unconditional generator (`run_la_proteina`).

La-Proteina's `proteinfoundation/generate.py` is not an installed package --
it does `sys.path.insert(0, os.path.abspath("."))` at import time, so it
MUST be run with the repo root as the process's own cwd, and its Hydra
config loader (`hydra.initialize(config_path, ...)`) resolves `config_path`
relative to `generate.py`'s own file location, not cwd -- `--config_subdir
X` becomes `../configs/X`, i.e. a subdirectory of THIS repo's own
`configs/` directory, not an arbitrary external path. There is no CLI
surface to point either of these at the dispatcher's scratch workdir, so
this wrapper does the following instead:

1. Writes a per-call config pair into a UNIQUELY NAMED subdirectory of the
   repo's own `configs/` (named after this run's scratch-workdir token, so
   two concurrent calls -- there are, per WAVE-COMMON, other agents active
   on this host -- can never collide), copying `inference_base.yaml`
   alongside it so the `defaults:` chain still resolves without touching
   the repo's shared top-level configs at all.
2. Invokes `generate.py` as a subprocess with `cwd` explicitly set to the
   repo root (required for its own import mechanism), not this wrapper's
   own cwd (the dispatcher's scratch workdir).
3. Copies the run's output (written by La-Proteina, again by construction,
   to a cwd-relative `./inference/<config_name>/...` -- confirmed live) back
   into the ACTUAL scratch workdir this wrapper itself was invoked in, which
   is what this tool's manifest `outputs:` glob is relative to.
4. Removes both the temporary config subdirectory and the repo-side
   `inference/<config_name>/` output directory it created, in a `finally`
   block, so nothing lingers in the shared checkout regardless of success
   or failure.

Every parameter this tool exposes maps directly onto one of `generate.py`'s
own config fields (`generation.dataset.nlens_cfg.nres_lens`,
`generation.args.nsteps`, `generation.model.{bb_ca,local_latents}
.simulation_step_params.{sc_scale_noise,sc_scale_score}`, etc.) -- read from
`configs/inference_base.yaml` and `configs/generation/uncond_codes*.yaml`.
Every metric-computation flag (designability, novelty, FID, ...) is forced
off: this tool exposes GENERATION only, matching every other generative tool
in this wave, and several of those flags would invoke ESMFold or other
external scoring engines that belong to separate registered tools, not this
one.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

_REPO_ROOT = Path("/home/jk661/projects/la-proteina")
_CKPT_DIR = _REPO_ROOT / "checkpoints_laproteina"
_CKPT_NAME = "LD1_ucond_notri_512.ckpt"
_AE_CKPT_PATH = _CKPT_DIR / "AE1_ucond_512.ckpt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", required=True, help="JSON list of ints")
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--max-nsamples-per-batch", type=int, required=True)
    parser.add_argument("--nsteps", type=int, required=True)
    parser.add_argument("--self-cond", choices=["true", "false"], required=True)
    parser.add_argument("--sc-scale-noise", type=float, required=True)
    parser.add_argument("--sc-scale-score", type=float, required=True)
    parser.add_argument("--guidance-w", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    lengths = json.loads(args.lengths)
    self_cond = args.self_cond == "true"

    call_workdir = Path.cwd()  # the dispatcher's scratch workdir
    run_token = call_workdir.name  # unique per call, e.g. "pdmcp-<hex12>"
    config_name = f"mcp_{run_token}"
    config_subdir_name = f"_mcp_{run_token}"

    config_subdir = _REPO_ROOT / "configs" / config_subdir_name
    generation_subdir = config_subdir / "generation"
    repo_output_dir = _REPO_ROOT / "inference" / config_name

    step_params = {
        "sc_scale_noise": args.sc_scale_noise,
        "sc_scale_score": args.sc_scale_score,
    }

    inference_yaml = {
        "defaults": ["inference_base", {"generation": "gen"}, "_self_"],
        "run_name_": config_name,
        "ckpt_name": _CKPT_NAME,
        "ckpt_path": str(_CKPT_DIR),
        "autoencoder_ckpt_path": str(_AE_CKPT_PATH),
        "seed": args.seed,
        "generation": {
            "n_recycle": 0,
            "args": {
                "nsteps": args.nsteps,
                "self_cond": self_cond,
                "guidance_w": args.guidance_w,
                "ag_ratio": 0.0,
                "ag_ckpt_path": None,
                "save_trajectory_every": 0,
                "fold_cond": False,
            },
            "model": {
                "bb_ca": {"simulation_step_params": dict(step_params)},
                "local_latents": {"simulation_step_params": dict(step_params)},
            },
        },
    }

    generation_yaml = {
        "args": {"fold_cond": False},
        "dataset": {
            "nlens_cfg": {
                "nres_lens": lengths,
                "min_len": None,
                "max_len": None,
                "step_len": None,
            },
            "cath_codes": None,
            "nsamples": args.num_samples,
            "max_nsamples_per_batch": args.max_nsamples_per_batch,
            "empirical_distribution_cfg": {
                "len_cath_code_path": None,
                "cath_code_level": None,
                "bucket_min_len": 50,
                "bucket_max_len": 274,
                "bucket_step_size": 25,
            },
        },
        "metric": {
            "compute_designability": False,
            "designability_folding_models": [],
            "compute_codesignability": False,
            "codesignability_folding_models": [],
            "compute_co_sequence_recovery": False,
            "compute_novelty_pdb": False,
            "compute_novelty_afdb": False,
            "compute_novelty_afdb_rep_v4": False,
            "compute_novelty_afdb_rep_v4_geniefilters_maxlen512": False,
            "compute_motif_scaffolding": False,
            "compute_motif_backbone_designability": False,
            "compute_motif_aa_designability": False,
            "compute_motif_rmsd": False,
            "keep_folding_outputs": False,
            "compute_binder_metrics": False,
            "compute_multimer_metrics": False,
            "compute_fid": False,
            "metric_factory": None,
        },
    }

    try:
        generation_subdir.mkdir(parents=True, exist_ok=False)
        shutil.copy2(
            _REPO_ROOT / "configs" / "inference_base.yaml",
            config_subdir / "inference_base.yaml",
        )
        # Hydra's `--config_name X` looks up `X.yaml` in the search path --
        # it must match config_name exactly (confirmed live: naming this
        # file `inference.yaml` while passing `--config_name mcp_<token>`
        # raised `MissingConfigException: Cannot find primary config
        # 'mcp_<token>'`).
        (config_subdir / f"{config_name}.yaml").write_text(
            yaml.safe_dump(inference_yaml, sort_keys=False)
        )
        (generation_subdir / "gen.yaml").write_text(yaml.safe_dump(generation_yaml, sort_keys=False))

        proc = subprocess.run(
            [
                sys.executable,
                "proteinfoundation/generate.py",
                "--config_name",
                config_name,
                "--config_subdir",
                config_subdir_name,
                "--job_id",
                "0",
                # The loaded checkpoint's OWN saved hyperparameters (from
                # training) carry a metric_factory config that interpolates
                # ${oc.env:DATA_PATH} -- confirmed live: without this,
                # generate.py crashes with
                # `InterpolationResolutionError: Environment variable
                # 'DATA_PATH' not found`, even with every metric-computation
                # flag in THIS call's own generation.metric section turned
                # off (the interpolation is resolved while the checkpoint's
                # embedded config is loaded, not lazily on first use).
                # generate.py's own --data_path flag sets this env var for
                # us; the directory itself is never read for a
                # generation-only call, so this call's own scratch workdir
                # is as good a value as any.
                "--data_path",
                str(call_workdir),
            ],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
        )
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        if proc.returncode != 0:
            sys.exit(proc.returncode)

        if not repo_output_dir.exists():
            raise RuntimeError(
                f"La-Proteina exited 0 but did not create the expected output "
                f"directory {repo_output_dir}"
            )
        dest = call_workdir / "inference"
        dest.mkdir(exist_ok=True)
        shutil.copytree(repo_output_dir, dest / config_name)
    finally:
        shutil.rmtree(config_subdir, ignore_errors=True)
        shutil.rmtree(repo_output_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
