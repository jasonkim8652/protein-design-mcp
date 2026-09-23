"""Wrapper for Protpardelle-1c's multi-chain (binder) sampler
(`run_protpardelle`).

Protpardelle-1c's own CLI (`python -m protpardelle.sample <yaml>
--motif-pdb ...`) takes a whole SAMPLING CONFIG YAML, not individual flags
for the motif-scaffolding surface (contig, hotspots, per-model search
space) -- so this wrapper builds that file itself, the same pattern as
Genie 3's wrapper scripts. Per Protpardelle-1c's own CLI docstring, when
`--motif-pdb` is given the YAML's own `motifs:` entry for that search-space
slot must be `null` (read from `sample()`'s own docstring in
`src/protpardelle/sample.py`) -- this wrapper always does that.

`PROTPARDELLE_OUTPUT_DIR` is set via this manifest's `env_vars` to a
RELATIVE `results` -- Protpardelle-1c's own default (when the env var is
unset) is an ABSOLUTE path derived from the PACKAGE checkout's own root
(`PROJECT_ROOT_DIR / "results"`, read from `protpardelle/env.py`), not
cwd-relative, so leaving it unset would write into the shared checkout
instead of this call's own scratch workdir. A relative value resolves
against the dispatcher's own cwd (the scratch workdir), which is what
contains it.

`num_mpnn_seqs` is always 0: sequence design is `run_mpnn`'s job, the same
rule as every other generative tool in this wave.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

_MODELS = {
    # model_name -> (epoch, sampling_config) -- see run_protpardelle.yaml's
    # doc for what each is recommended for, sourced from Protpardelle-1c's
    # own README table.
    "cc83": ("2616", "sampling_sidechain_conditional"),
    "cc95": ("3490", "sampling_sidechain_conditional"),
    "cc94": ("3100", "sampling_sidechain_conditional_allatom_s1"),
    "cc78": ("1431", "sampling_sidechain_conditional"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-pdb", required=True)
    parser.add_argument("--contig", required=True)
    parser.add_argument("--total-lengths", required=True, help="JSON list of [min,max] pairs")
    parser.add_argument("--hotspots", required=True, help="JSON list of tags, or JSON null")
    parser.add_argument("--model", required=True, choices=list(_MODELS))
    parser.add_argument("--step-scale", type=float, required=True)
    parser.add_argument("--schurn", type=float, required=True)
    parser.add_argument("--crop-cond-start", type=float, required=True)
    parser.add_argument("--translation", required=True, help="JSON [x, y, z]")
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    total_lengths = json.loads(args.total_lengths)
    hotspots = json.loads(args.hotspots)
    translation = json.loads(args.translation)
    epoch, sampling_config = _MODELS[args.model]

    hotspots_str = ",".join(hotspots) if hotspots else None

    sampling_config_doc = {
        "search_space": {
            "models": [[args.model, epoch, sampling_config]],
            "step_scales": [args.step_scale],
            "schurns": [args.schurn],
            "crop_cond_starts": [args.crop_cond_start],
            "translations": [translation],
        },
        "motifs": [None],
        "motif_contigs": [args.contig],
        "total_lengths": [total_lengths],
        "hotspots": [hotspots_str],
        "ssadj": [None],
        "partial_diffusion": {"enabled": False, "rewind_steps": []},
    }

    config_path = Path.cwd() / "sampling_config.yaml"
    config_path.write_text(yaml.safe_dump(sampling_config_doc, sort_keys=False))

    cmd = [
        sys.executable,
        "-m",
        "protpardelle.sample",
        str(config_path),
        "--motif-pdb",
        str(Path(args.target_pdb).resolve()),
        "--num-samples",
        str(args.num_samples),
        "--num-mpnn-seqs",
        "0",
        "--batch-size",
        str(args.batch_size),
    ]
    if args.seed is not None:
        cmd += ["--seed", str(args.seed)]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
