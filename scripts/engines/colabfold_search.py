"""Wrapper for ColabFold's MMseqs2 search (``run_colabfold_search``). Runs
inside the ``colabfold`` environment.

Two backends, selected by the required ``--backend`` argument (see the
manifest's ``backend`` schema entry -- no default, every call states one):

- ``local``: shells out to the standalone ``colabfold_search`` binary
  directly (never imports ``colabfold.colabfold``/``colabfold.batch`` or any
  function that defaults to ColabFold's remote MSA server) against a FIXED,
  caller-never-controls local ``--db-root``. Structurally cannot reach a
  network address under this path -- see the manifest doc's "backend"
  section.
- ``remote``: calls ``colabfold.colabfold.run_mmseqs2`` -- the same helper
  ``colabfold_batch`` itself uses -- against its fixed default
  ``host_url="https://api.colabfold.com"``, ColabFold's own public MSA
  server. This TRANSMITS THE QUERY SEQUENCE off this host; the manifest doc
  states this plainly and it is never the default. ``host_url`` is never
  read from caller input, so this path can only ever reach ColabFold's own
  official server, never an arbitrary one.

Either way, the query's a3m always lands at the same, predictable path this
tool's ``outputs:`` glob expects: ``results/query.a3m``, with a fixed
``>query`` header -- for ``local``, this falls out of
``colabfold_search``'s own output renaming (see ``colabfold/mmseqs/search.py``:
a monomer job's final a3m is named ``{safe_filename(header)}.a3m``, and
`safe_filename` passes alphanumerics through unchanged, and this wrapper
always writes the query FASTA with header ``>query``); for ``remote``,
``run_mmseqs2`` returns the merged a3m as a Python string (numbered ``>101``,
its own internal query index, not ``>query``) and this wrapper rewrites
just that first header line and writes the result itself.

``--unpack 1``, ``--use-templates 0`` and ``--pair-mode unpaired`` are
pinned rather than exposed as parameters for the ``local`` path (and
``use_templates=False``/``use_pairing=False`` are hardcoded for ``remote``)
-- see the manifest's "Not exposed" section for why each is a knob with no
effect (or no verifiable output shape) at this tool's single-sequence
granularity.

Reads one argv list built by ``adapters/colabfold_search.py``. Writes
``query.fasta`` (``local`` only) and a ``results/`` directory into the
current working directory (the dispatcher's scratch workdir).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

QUERY_NAME = "query.fasta"
RESULTS_DIR = "results"
RESULT_A3M = "results/query.a3m"

# Pinned -- never read from caller input (see this module's own docstring).
# Keeps the "remote" backend reachable ONLY at ColabFold's own official
# server, never an arbitrary caller-supplied host.
REMOTE_HOST_URL = "https://api.colabfold.com"
REMOTE_USER_AGENT = "protein-design-mcp/1.0"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("sequence")
    parser.add_argument("--backend", choices=["local", "remote"], required=True)
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


def _run_remote(sequence: str, use_env: bool, filter_level: int) -> None:
    """Submit ``sequence`` to ColabFold's own public MSA server and write
    the returned a3m to ``RESULT_A3M`` -- the same path the ``local``
    backend's ``colabfold_search`` invocation produces, so
    ``adapters/colabfold_search.py``'s ``parse_output`` needs no branch of
    its own.

    Imported lazily (not at module top) so the ``local`` backend's import
    time never depends on ``colabfold.colabfold`` (which imports
    ``matplotlib``/``jax`` at module scope) succeeding.
    """
    from colabfold.colabfold import run_mmseqs2

    a3m_lines = run_mmseqs2(
        sequence,
        "remote_query",
        use_env=use_env,
        use_filter=filter_level != 0,
        # Pinned -- see this module's own docstring: no observable effect
        # (use_pairing) or no way to verify the output shape (use_templates)
        # at this tool's single-sequence granularity.
        use_templates=False,
        use_pairing=False,
        host_url=REMOTE_HOST_URL,
        user_agent=REMOTE_USER_AGENT,
    )
    # run_mmseqs2(x=<a single string>, use_templates=False) returns a plain
    # list of a3m strings, one per input sequence -- index 0 is ours.
    text = a3m_lines[0]
    lines = text.splitlines()
    if lines:
        # run_mmseqs2's own header is its internal query index (e.g.
        # ">101"), not ">query" -- rewritten here so this backend's output
        # is byte-shape-identical (same fixed header) to the local
        # backend's.
        lines[0] = ">query"
    Path(RESULTS_DIR).mkdir(exist_ok=True)
    Path(RESULT_A3M).write_text("\n".join(lines) + "\n")


def main() -> None:
    args = _parse_args()

    if args.backend == "remote":
        _run_remote(args.sequence, args.use_env, args.filter)
        return

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
