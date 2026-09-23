"""Co-fold a target with AlphaFold2-Multimer via ColabFold. Runs inside the
`colabfold` environment.

Builds ColabFold's own multimer FASTA convention (`:`-joined chains in one
record, confirmed from `colabfold_batch --help`/README) when `msa` was
`null`, or uses the caller-supplied a3m file directly as ColabFold's input
(which "overwrites [--msa-mode]" per its own `--help`) when `msa` was a
path -- either way, `--msa-mode` is only ever passed the single_sequence
value; there is no code path here that can construct one of ColabFold's
mmseqs2_* (remote-server) modes, mirroring the restriction already built
into ``protein_design_mcp.adapters.alphafold2_multimer.build_args``.

Invokes the `colabfold_batch` console script directly (on PATH inside this
environment -- the verified invocation from the install report), pointing
`--data` at the host's existing weights cache explicitly rather than via
XDG_CACHE_HOME (see run_alphafold2_multimer.yaml's `engine.mounts` comment
for why).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_JOBNAME = "query"
_DATA_DIR = "/home/jk661/.cache/colabfold"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequences", required=True, help="JSON list of chain sequences")
    parser.add_argument("--msa-mode", default=None)
    parser.add_argument("--msa-path", default=None)
    parser.add_argument("--num-recycle", type=int, required=True)
    parser.add_argument("--num-models", type=int, required=True)
    parser.add_argument("--num-seeds", type=int, required=True)
    parser.add_argument("--random-seed", type=int, required=True)
    parser.add_argument("--num-ensemble", type=int, required=True)
    parser.add_argument("--pair-mode", required=True)
    parser.add_argument("--pair-strategy", required=True)
    parser.add_argument("--rank", required=True)
    parser.add_argument("--stop-at-score", type=float, required=True)
    parser.add_argument("--use-dropout", action="store_true")
    args = parser.parse_args()

    if bool(args.msa_mode) == bool(args.msa_path):
        raise SystemExit(
            "exactly one of --msa-mode or --msa-path must be given "
            f"(got msa_mode={args.msa_mode!r}, msa_path={args.msa_path!r})"
        )

    sequences = json.loads(args.sequences)

    if args.msa_path is not None:
        input_path = Path(args.msa_path)
    else:
        fasta = Path("query.fasta")
        fasta.write_text(f">{_JOBNAME}\n{':'.join(sequences)}\n")
        input_path = fasta

    results_dir = Path("results")

    cmd = [
        "colabfold_batch",
        "--data",
        _DATA_DIR,
        "--model-type",
        "alphafold2_multimer_v3",
        "--num-recycle",
        str(args.num_recycle),
        "--num-models",
        str(args.num_models),
        "--num-seeds",
        str(args.num_seeds),
        "--random-seed",
        str(args.random_seed),
        "--num-ensemble",
        str(args.num_ensemble),
        "--pair-mode",
        args.pair_mode,
        "--pair-strategy",
        args.pair_strategy,
        "--rank",
        args.rank,
        "--stop-at-score",
        str(args.stop_at_score),
    ]
    if args.use_dropout:
        cmd.append("--use-dropout")
    if args.msa_mode is not None:
        cmd += ["--msa-mode", args.msa_mode]
    cmd += [str(input_path), str(results_dir)]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
