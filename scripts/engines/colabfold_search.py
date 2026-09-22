"""Wrapper for ColabFold's local MMseqs2 search (``run_colabfold_search``).
Runs inside the ``colabfold`` environment.

Always shells out to the standalone ``colabfold_search`` binary directly
(never imports ``colabfold.batch`` or any function that defaults to
ColabFold's remote MSA server) against a FIXED, caller-never-controls local
``--db-root`` -- see the manifest's "Local search only" section for why this
structurally rules out ever reaching a remote host, not merely documents it.

The query is always written with a single fixed FASTA header (``>query``),
never derived from caller input, so ``colabfold_search``'s own output
renaming (see ``colabfold/mmseqs/search.py``: a monomer job's final a3m is
named ``{safe_filename(header)}.a3m``, and ``safe_filename`` passes
alphanumerics through unchanged) always lands at the same, predictable path
this tool's ``outputs:`` glob expects: ``results/query.a3m``.

``--unpack 1``, ``--use-templates 0`` and ``--pair-mode unpaired`` are
pinned rather than exposed as parameters -- see the manifest's "Not
exposed" section for why each is a knob with no effect (or no verifiable
output shape) at this tool's single-sequence granularity.

Reads one argv list built by ``adapters/colabfold_search.py``. Writes
``query.fasta`` and a ``results/`` directory into the current working
directory (the dispatcher's scratch workdir).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

QUERY_NAME = "query.fasta"
RESULTS_DIR = "results"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("sequence")
    parser.add_argument("--mmseqs-binary", required=True)
    parser.add_argument("--db-root", required=True)
    parser.add_argument("--db1", required=True)
    parser.add_argument("--db3", required=True)
    parser.add_argument("--use-env", dest="use_env", action="store_true")
    parser.add_argument("--no-use-env", dest="use_env", action="store_false")
    parser.add_argument("--prefilter-mode", type=int, required=True)
    parser.add_argument("--sensitivity", type=float, default=None)
    parser.add_argument("--filter", type=int, required=True)
    parser.add_argument("--expand-eval", type=float, required=True)
    # int, not float -- colabfold_search's own argparse defines --align-eval
    # as type=int (verified live, 2026-09-22, unlike expand-eval/qsc which
    # really are floats); a float string like "10.0" makes colabfold_search
    # itself reject the call with "invalid int value".
    parser.add_argument("--align-eval", type=int, required=True)
    parser.add_argument("--diff", type=int, required=True)
    parser.add_argument("--qsc", type=float, required=True)
    parser.add_argument("--max-accept", type=int, required=True)
    parser.add_argument("--db-load-mode", type=int, required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--use-gpu", dest="use_gpu", action="store_true")
    parser.add_argument("--no-use-gpu", dest="use_gpu", action="store_false")
    parser.add_argument("--gpu-server", dest="gpu_server", action="store_true")
    parser.add_argument("--no-gpu-server", dest="gpu_server", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    Path(QUERY_NAME).write_text(f">query\n{args.sequence}\n")
    Path(RESULTS_DIR).mkdir(exist_ok=True)

    cmd = [
        "colabfold_search",
        "--prefilter-mode",
        str(args.prefilter_mode),
        "--db1",
        args.db1,
        "--db3",
        args.db3,
        "--use-env",
        "1" if args.use_env else "0",
        # Pinned, not exposed -- see this file's and the manifest's own
        # module docstrings for why: pairing never runs for a
        # single-sequence query regardless of pair-mode, and templates
        # (use-templates) are not verifiable on this host (no database).
        "--use-templates",
        "0",
        "--use-env-pairing",
        "0",
        "--pair-mode",
        "unpaired",
        "--filter",
        str(args.filter),
        "--mmseqs",
        args.mmseqs_binary,
        "--expand-eval",
        str(args.expand_eval),
        "--align-eval",
        str(args.align_eval),
        "--diff",
        str(args.diff),
        "--qsc",
        str(args.qsc),
        "--max-accept",
        str(args.max_accept),
        "--db-load-mode",
        str(args.db_load_mode),
        # Pinned -- this tool's declared `a3m` output and its adapter both
        # depend on unpacked, loose a3m files rather than an MMseqs2
        # database.
        "--unpack",
        "1",
        "--threads",
        str(args.threads),
        "--gpu",
        "1" if args.use_gpu else "0",
        "--gpu-server",
        "1" if args.gpu_server else "0",
    ]
    if args.sensitivity is not None:
        cmd += ["-s", str(args.sensitivity)]
    cmd += [QUERY_NAME, args.db_root, RESULTS_DIR]

    result = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
