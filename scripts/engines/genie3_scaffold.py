"""Wrapper for Genie 3's unconditional sampler (`run_genie3_scaffold`).

Genie 3's own CLI (`genie3.cli generate -c <config.yaml>`) takes a single
YAML file path -- no `key=value` override mechanism like Hydra -- so this
wrapper builds that file itself (unlike FrameFlow/MultiFlow/Genie 2, which
take everything as CLI flags or Hydra overrides directly in the adapter).
Every path this wrapper writes into the config (checkpoint, model config,
`paths.rootdir`, `paths.dataset`) is made ABSOLUTE, since Genie 3's own
config loader (`genie3.config.loader`) resolves every one of these as a
plain string against the process's OWN cwd, not the config file's location
-- confirmed from source (`resolve_config_path` in `genie3/config/loader.py`
does `Path(config_path).expanduser().resolve()`, and `io["outdir"]` is used
verbatim). Making everything absolute means this wrapper needs no special
cwd handling at all (unlike La-Proteina): the dispatcher's own scratch
workdir as cwd is fine as-is, and `paths.rootdir` is pointed AT that same
workdir directly so results land exactly where the manifest's `outputs:`
glob expects them, with no copy-back step required.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

_REPO_ROOT = Path("/home/jk661/projects/genie3")
_CHECKPOINTS = {
    "v1": {
        "checkpoint": str(_REPO_ROOT / "pretrained" / "v1" / "checkpoints" / "step=600000.ckpt"),
        "config": str(_REPO_ROOT / "pretrained" / "v1" / "config.yaml"),
        "sampler_name": "ddim",
    },
    "legacy": {
        "checkpoint": str(_REPO_ROOT / "pretrained" / "legacy" / "checkpoints" / "step=400000.ckpt"),
        "config": str(_REPO_ROOT / "pretrained" / "legacy" / "config.yaml"),
        "sampler_name": "ddpm",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-variant", choices=["v1", "legacy"], required=True)
    parser.add_argument("--min-length", type=int, required=True)
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--length-step", type=int, required=True)
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--direction-scale", type=float, required=True)
    parser.add_argument("--eta", type=float, required=True)
    parser.add_argument("--n-sample-step", type=int, required=True)
    parser.add_argument("--noise-scale", type=float, required=True)
    parser.add_argument("--predict-sidechain", choices=["true", "false"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    workdir = Path.cwd()
    out_dir = workdir / "output"
    model = _CHECKPOINTS[args.model_variant]

    if model["sampler_name"] == "ddim":
        sampler_cfg = {
            "name": "ddim",
            "sampler": {
                "direction_scale": args.direction_scale,
                "eta": args.eta,
                "n_sample_step": args.n_sample_step,
                "noise_scale": args.noise_scale,
                "verbose": False,
                "predict_sequence": False,
                "predict_sidechain": args.predict_sidechain == "true",
            },
        }
    else:
        sampler_cfg = {
            "name": "ddpm",
            "sampler": {"noise_scale": args.noise_scale, "verbose": False},
        }

    experiment = {
        "experiment": {"name": "run", "seed": args.seed},
        "paths": {"rootdir": str(out_dir)},
        "generation": {
            "base": {"checkpoint": model["checkpoint"], "config": model["config"]},
            "dataset": {
                "source": "unconditional",
                "min_length": args.min_length,
                "max_length": args.max_length,
                "length_step": args.length_step,
                "n_sample": args.num_samples,
                "batch_size": args.batch_size,
            },
            "inference": {"sampler": sampler_cfg},
        },
    }

    config_path = workdir / "experiment.yaml"
    config_path.write_text(yaml.safe_dump(experiment, sort_keys=False))

    cmd = [
        sys.executable,
        "-m",
        "genie3.cli",
        "generate",
        "-c",
        str(config_path),
        "--num-devices",
        "1",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
