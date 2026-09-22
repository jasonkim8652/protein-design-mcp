"""Adapter for ColabFold's local MMseqs2 search
(``scripts/engines/colabfold_search.py``, env ``colabfold``).

``MMSEQS_BINARY`` and ``DB_ROOT`` are deployment facts, not something a
caller should choose per call -- same reasoning as
``mmseqs_search.MMSEQS_BINARY``/``DB_ROOT``, and they must match
``engine.mounts`` in run_colabfold_search.yaml. ``DB_ROOT`` is currently an
EMPTY placeholder directory -- see the manifest's "Verification status"
section for why.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.schema import Manifest

MMSEQS_BINARY = "/usr/local/bin/mmseqs"
DB_ROOT = "/home/jk661/.cache/colabfold_dbs"


def build_args(manifest: Manifest, params: dict[str, Any]) -> list[str]:
    """Translate validated parameters into the wrapper script's argv.

    ``manifest`` is unused here -- this adapter backs exactly one tool --
    but is part of every adapter's signature (see ``adapters_discovery``).
    """
    del manifest
    args = [
        str(params["sequence"]),
        "--mmseqs-binary",
        MMSEQS_BINARY,
        "--db-root",
        DB_ROOT,
        "--db1",
        str(params["db1"]),
        "--db3",
        str(params["db3"]),
        "--use-env" if params["use_env"] else "--no-use-env",
        "--prefilter-mode",
        str(params["prefilter_mode"]),
        "--filter",
        str(params["filter"]),
        "--expand-eval",
        str(params["expand_eval"]),
        "--align-eval",
        str(params["align_eval"]),
        "--diff",
        str(params["diff"]),
        "--qsc",
        str(params["qsc"]),
        "--max-accept",
        str(params["max_accept"]),
        "--db-load-mode",
        str(params["db_load_mode"]),
        "--threads",
        str(params["threads"]),
        "--use-gpu" if params["use_gpu"] else "--no-use-gpu",
        "--gpu-server" if params["gpu_server"] else "--no-gpu-server",
    ]
    if params["sensitivity"] is not None:
        args += ["--sensitivity", str(params["sensitivity"])]
    return args


def parse_output(manifest: Manifest, run: CompletedRun) -> dict[str, Any]:
    """Read the collected ``a3m`` output and the query length. ``manifest``
    is unused (see ``build_args``)."""
    del manifest
    a3m_path = run.outputs.get("a3m")
    if not a3m_path:
        raise ValueError(
            "run_colabfold_search's declared 'a3m' output was not "
            f"collected -- no results/query.a3m file was found. "
            f"run.outputs was: {run.outputs}"
        )
    text = Path(str(a3m_path)).read_text()
    lines = text.splitlines()
    # The query record is always the first sequence line in an a3m/FASTA
    # (the second non-empty line, right after the ">query" header this
    # wrapper always writes) -- its length, independent of how many hit
    # rows follow, is a cheap sanity number a caller can compare against
    # the sequence they submitted.
    query_length = 0
    for line in lines[1:]:
        if line.startswith(">"):
            break
        if line.strip():
            query_length = len(line.strip())
            break

    return {
        "query_length": query_length,
    }
