"""Wrapper for La-Proteina's unconditional generator (`run_la_proteina`).

La-Proteina's `proteinfoundation/generate.py` is not an installed package --
it does `sys.path.insert(0, os.path.abspath("."))` at import time, so it
MUST be run with the repo root as the process's own cwd, and its Hydra
config loader (`hydra.initialize(config_path, ...)`) resolves `config_path`
relative to `generate.py`'s own file location -- specifically
`realpath(dirname(__file__))` (confirmed from hydra's own
`_internal/utils.py::compute_search_path_dir`), so a plain SYMLINK to
`generate.py` does not help: hydra resolves straight through it back to the
real, read-only-mounted checkout. There is no CLI surface to point either
the config directory or the output directory at the dispatcher's scratch
workdir, so this wrapper builds a WRITABLE VIEW of the repo instead, rooted
at the call's own scratch workdir (`Path.cwd()`), and runs `generate.py`
from there:

1. `_build_repo_view()` symlinks every top-level entry of the real repo
   (`openfold/`, `assets/`, `.git`, etc.) straight into the scratch workdir
   -- cheap, and fine for ordinary Python imports/reads, which follow
   symlinks without caring whether the target is real. `proteinfoundation/`
   is handled specially: it becomes a REAL (non-symlink) directory in the
   scratch workdir, with every entry inside it symlinked EXCEPT
   `generate.py` itself, which is a real copy. `configs/`, `inference/`,
   `tmp/` and `tmp_ae/` are deliberately left out entirely -- see below.
2. Every one of La-Proteina's own cwd-relative writes (`./configs/<subdir>`
   for hydra's per-call config, `./inference/<config_name>` for its output,
   `./tmp` / `./tmp_ae` for checkpoint-loading scratch state --
   `proteinfoundation/proteina.py`'s and
   `partial_autoencoder/autoencoder.py`'s own `store_dir` defaults) is
   therefore relative to the SAME writable scratch workdir this wrapper
   itself was invoked in -- which is what this tool's manifest `outputs:`
   glob (`inference/**/*.pdb`) is relative to. No copy-back step is needed:
   the output lands exactly where the dispatcher will look for it.
3. `generate.py`'s own `__file__` is real (not a symlink) -- see step 1 --
   so hydra's `realpath(dirname(__file__))` resolves to
   `<scratch_workdir>/proteinfoundation`, a genuinely writable directory,
   and `../configs/<subdir>` (this wrapper's own `--config_subdir` value)
   resolves to `<scratch_workdir>/configs/<subdir>`, not the read-only
   mount. Regular Python imports of `proteinfoundation`'s OWN submodules
   (`proteinfoundation.datasets...`, `proteinfoundation.utils...`) are
   unaffected by any of this -- they resolve through the symlinked
   entries exactly as they would through the real ones.

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
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

_REPO_ROOT = Path("/home/jk661/projects/la-proteina")
_CKPT_DIR = _REPO_ROOT / "checkpoints_laproteina"
_CKPT_NAME = "LD1_ucond_notri_512.ckpt"
_AE_CKPT_PATH = _CKPT_DIR / "AE1_ucond_512.ckpt"

# Left out of the top-level symlink mirror in `_build_repo_view` -- each of
# these must be a REAL, writable path under the scratch workdir instead of
# a symlink into the (read-only, per the container's own mount contract)
# real checkout: `configs` because this wrapper creates a per-call subdir
# under it, `inference` because that is where generate.py writes its
# output, `tmp`/`tmp_ae` because `proteina.py`/`autoencoder.py` write
# checkpoint-loading scratch state there by default (cwd-relative).
# `lightning_logs` is the SAME class -- PyTorch Lightning's own default
# CSVLogger writes `./lightning_logs/version_N/` (cwd-relative) whenever no
# explicit logger is configured, confirmed live in-container: symlinking it
# (as an ordinary read-only entry, before this was added) let Lightning
# resolve straight through to the real repo's own `lightning_logs/`
# (present on this host from prior interactive runs) and fail with
# `OSError: [Errno 30] Read-only file system` the moment it tried to create
# a new version subdirectory there.
#
# This list is inherently a DENYLIST of write-targets discovered by
# actually running the engine, not something derivable from the repo's
# structure alone (nothing distinguishes `lightning_logs` from `assets` or
# `openfold` by name or type) -- a future La-Proteina call path that writes
# somewhere new under the repo root can reproduce this same failure mode
# for a directory not yet listed here. The in-container live proof is what
# is positioned to catch that when it happens, the same way it caught this
# one.
_SKIP_TOP_LEVEL = {"proteinfoundation", "configs", "inference", "tmp", "tmp_ae", "lightning_logs"}


def _build_repo_view(call_workdir: Path) -> None:
    """Give La-Proteina's subprocess a repo root it can write into.

    See this module's own docstring for why a plain symlinked mirror of
    `proteinfoundation/` is not enough: hydra resolves `generate.py`'s
    containing directory with `realpath()`, which follows a symlink
    straight back to the real, read-only-mounted checkout. Only
    `generate.py` itself (and its immediate containing directory) needs to
    be a real, physical file/directory under the scratch workdir; every
    other top-level entry, and every OTHER file inside
    `proteinfoundation/`, is a plain symlink -- ordinary Python imports
    (unlike hydra's config-path resolution) don't care whether the path
    they open runs through a symlink.
    """
    for entry in os.listdir(_REPO_ROOT):
        if entry in _SKIP_TOP_LEVEL:
            continue
        (call_workdir / entry).symlink_to(_REPO_ROOT / entry)

    real_pf = _REPO_ROOT / "proteinfoundation"
    pf_view = call_workdir / "proteinfoundation"
    pf_view.mkdir()
    for entry in os.listdir(real_pf):
        if entry == "generate.py":
            shutil.copy2(real_pf / entry, pf_view / entry)
        else:
            (pf_view / entry).symlink_to(real_pf / entry)


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

    _build_repo_view(call_workdir)

    # Both now relative to the WRITABLE scratch workdir, not the read-only
    # mounted checkout -- see _build_repo_view's and this module's own
    # docstrings.
    config_subdir = call_workdir / "configs" / config_subdir_name
    generation_subdir = config_subdir / "generation"
    repo_output_dir = call_workdir / "inference" / config_name

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
            # cwd is now the writable scratch workdir itself (see
            # _build_repo_view), not the real (read-only mounted) repo --
            # generate.py's own sys.path.insert(0, abspath(".")) resolves
            # `proteinfoundation` through the symlinked view planted there,
            # and every one of its cwd-relative writes (./configs/<subdir>,
            # ./inference/<config_name>, ./tmp, ./tmp_ae) lands in scratch.
            cwd=str(call_workdir),
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
        # No copy-back needed: repo_output_dir IS already
        # <call_workdir>/inference/<config_name>, exactly where the
        # manifest's outputs: pattern (inference/**/*.pdb) looks for it.
    finally:
        # Only the per-call config subdir needs explicit cleanup here --
        # everything else this wrapper created lives under call_workdir,
        # which the dispatcher itself removes on success (and preserves,
        # deliberately, for diagnosis on failure).
        shutil.rmtree(config_subdir, ignore_errors=True)


if __name__ == "__main__":
    main()
