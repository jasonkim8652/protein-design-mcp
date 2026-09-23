"""Wrapper for Genie 3's binder-design generator (`run_genie3_binder`).

Genie 3's own binder-design pipeline normally goes through
`scripts/problem/binder_design/prepare.py`, which (1) calls `colabfold_batch`
to build a target MSA and (2) computes an "extended interface" via a
Shrake-Rupley SASA calculation (`genie3.generation.utils.interface.extended
.compute_extended_interface`, which imports Biopython). The MSA step is
still never used here:

- The target MSA is confirmed, from source, to be read ONLY by evaluation/
  reward code (`target_msa_filepath` -- see the manifest doc's "MSA" note),
  never by generation itself, so this tool -- generation only, like every
  other tool in this wave -- never builds one. That keeps a design's target
  sequence on this machine, matching this server's own MSA policy.

**Interface expansion is now available** (`expand_interface: true`): the
`genie2_fixed` environment this tool now runs under (a clone of the original
`genie2` -- see this wave's report) has Biopython installed, so
`compute_extended_interface` can run. When requested, this wrapper calls it
exactly as Genie 3's own `prepare.py` does (`version_num=1`, which always
folds the literal hotspots into the returned set too) and sets
`generation.dataset.cond_strategy: "extended"` so Genie 3's own dataset
reads `target_interface_residues["extended"]` instead of `["hotspot"]`
(confirmed from source: `TargetFeaturePipeline`'s `interface_mode` is used
as a literal dict key into `target_interface_residues`, read in
`genie3.generation.utils.feat_utils`). The default remains
`expand_interface: false` (exact hotspot conditioning, unchanged behavior),
so this is additive, not a behavior change to existing callers.

This wrapper builds the minimal "problem" JSON Genie 3's inference dataset
actually reads at sampling time
(`genie3.generation.utils.feat_utils.create_np_features_from_target_config`,
read from source): `target_pdb_filepath`, `target_interface_residues`,
`binder_min_length`, `binder_max_length` -- nothing else. Everything else
mirrors `scripts/engines/genie3_scaffold.py`: an absolute-path experiment
YAML (Genie 3's config loader resolves plain strings against cwd, not the
config file's location), written fresh per call.
"""

from __future__ import annotations

import argparse
import json
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

_PROBLEM_KEY = "target"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-pdb", required=True)
    parser.add_argument("--hotspot-residues", required=True, help="JSON list of 'ChainResnum' tags")
    parser.add_argument("--binder-min-length", type=int, required=True)
    parser.add_argument("--binder-max-length", type=int, required=True)
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--model-variant", choices=["v1", "legacy"], required=True)
    parser.add_argument("--direction-scale", type=float, required=True)
    parser.add_argument("--eta", type=float, required=True)
    parser.add_argument("--n-sample-step", type=int, required=True)
    parser.add_argument("--noise-scale", type=float, required=True)
    parser.add_argument("--predict-sidechain", choices=["true", "false"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--expand-interface", choices=["true", "false"], required=True)
    parser.add_argument("--interface-cutoff-angstrom", type=float, required=True)
    parser.add_argument("--interface-rsa-threshold", type=float, required=True)
    parser.add_argument("--interface-abs-sasa-threshold", type=float, required=True)
    args = parser.parse_args()

    hotspot_residues = json.loads(args.hotspot_residues)
    target_pdb_path = str(Path(args.target_pdb).resolve())

    workdir = Path.cwd()
    out_dir = workdir / "output"
    dataset_dir = workdir / "problem_data"
    problems_dir = dataset_dir / "problems"
    problems_dir.mkdir(parents=True)

    expand_interface = args.expand_interface == "true"
    extended_interface_residues = None
    if expand_interface:
        # Imported here, not at module level: this import needs Biopython,
        # which is only guaranteed present when this wrapper is actually
        # dispatched under genie2_fixed (see the manifest). Importing it
        # unconditionally at module level would break --expand-interface
        # false callers too if this script were ever run under the
        # original genie2 env by mistake.
        from genie3.generation.utils.interface.extended import (
            compute_extended_interface,
        )

        extended_interface_residues = compute_extended_interface(
            target_pdb_filepath=target_pdb_path,
            target_hotspot_residues=hotspot_residues,
            version_num=1,
            cutoff=args.interface_cutoff_angstrom,
            rsa_thresh=args.interface_rsa_threshold,
            abs_sasa_thresh=args.interface_abs_sasa_threshold,
        )
        cond_strategy = "extended"
    else:
        cond_strategy = "hotspot"

    interface_residues = {"hotspot": hotspot_residues}
    if extended_interface_residues is not None:
        interface_residues["extended"] = extended_interface_residues

    problem = {
        "target_pdb_filepath": target_pdb_path,
        "target_interface_residues": interface_residues,
        "binder_min_length": args.binder_min_length,
        "binder_max_length": args.binder_max_length,
    }
    (problems_dir / f"{_PROBLEM_KEY}.json").write_text(json.dumps(problem, indent=2))

    # Always written (unlike target_interface_residues["extended"], which
    # is only present when expand_interface is true) so the manifest's
    # declared output exists on every call, not just expand_interface ones.
    (workdir / "interface_conditioning.json").write_text(
        json.dumps(
            {
                "cond_strategy": cond_strategy,
                "hotspot_residues": hotspot_residues,
                "extended_interface_residues": extended_interface_residues,
            },
            indent=2,
        )
    )

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
        "paths": {"rootdir": str(out_dir), "dataset": str(dataset_dir)},
        "generation": {
            "base": {"checkpoint": model["checkpoint"], "config": model["config"]},
            "dataset": {
                "source": "target",
                "n_sample": args.num_samples,
                "selections": _PROBLEM_KEY,
                "cond_strategy": cond_strategy,
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
