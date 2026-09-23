"""Adapter for MMseqs2 MSA/template search (scripts/engines/run_mmseqs_search.py).

The wrapper script does the actual work (createdb/search/result2msa/unpackdb
per database, merging, deduplication — see its own module docstring for the
full rationale). This adapter only translates validated parameters into its
argv and reads back the JSON summary it declares as an output.

``MMSEQS_BINARY`` and ``DB_ROOT`` are deployment facts (where this server's
mmseqs binary and databases live), not something a caller should choose per
call — same reasoning as ``openmm_minimize.OUTPUT_NAME`` being a constant
rather than a schema parameter. They must match ``engine.mounts`` in
run_mmseqs_search.yaml.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.schema import Manifest

MMSEQS_BINARY = "/usr/local/bin/mmseqs"
DB_ROOT = "/opt/alphafold3_data/mmseqs_db/mmseqs"

_SUMMARY_KEYS = (
    "query_length",
    "unpaired_hit_count",
    "paired_hit_count",
    "template_hit_count",
    "unpaired_databases_searched",
    "pair_searched",
    "templates_searched",
    "used_gpu",
    "elapsed_seconds",
)


def build_args(manifest: Manifest, params: dict[str, Any]) -> list[str]:
    """Translate validated parameters into the wrapper script's argv.

    ``manifest`` is unused here — this adapter backs exactly one tool — but
    is part of every adapter's signature (see ``adapters_discovery``) so a
    future adapter module backing several tools on one engine repo can
    branch on ``manifest.name`` without changing the call site.
    """
    del manifest
    args = [
        str(params["sequence"]),
        "--output-dir",
        ".",
        "--mmseqs-binary",
        MMSEQS_BINARY,
        "--db-root",
        DB_ROOT,
        "--sensitivity",
        str(params["sensitivity"]),
        "--e-value",
        str(params["e_value"]),
        "--max-sequences",
        str(params["max_sequences"]),
        "--coverage",
        str(params["coverage"]),
        "--coverage-mode",
        str(params["coverage_mode"]),
        "--min-seq-id",
        str(params["min_seq_id"]),
        "--num-iterations",
        str(params["num_iterations"]),
        "--threads",
        str(params["threads"]),
        "--template-e-value",
        str(params["template_e_value"]),
        "--max-template-hits",
        str(params["max_template_hits"]),
        "--pair" if params["pair"] else "--no-pair",
        "--search-templates" if params["search_templates"] else "--no-search-templates",
        "--use-gpu" if params["use_gpu"] else "--no-use-gpu",
    ]
    for db in params["unpaired_databases"]:
        args += ["--unpaired-database", str(db)]
    return args


def parse_output(manifest: Manifest, run: CompletedRun) -> dict[str, Any]:
    """Read the wrapper's declared ``search_summary`` JSON output.

    ``run.workdir`` no longer exists by the time this runs (the dispatcher
    removes it on a successful run before returning) — this is why the
    wrapper's counts and settings are read from a declared, collected
    output file (``run.outputs["search_summary"]``) rather than from
    stdout or from re-reading the workdir directly. ``manifest`` is unused
    (see ``build_args``).
    """
    del manifest
    summary_path = run.outputs.get("search_summary")
    if not summary_path:
        raise ValueError(
            "run_mmseqs_search's declared 'search_summary' output was not "
            f"collected — no search_summary.json file was found. "
            f"run.outputs was: {run.outputs}"
        )
    data = json.loads(Path(str(summary_path)).read_text())
    missing = [key for key in _SUMMARY_KEYS if key not in data]
    if missing:
        raise ValueError(
            f"search_summary.json at {summary_path} is missing key(s): "
            f"{', '.join(missing)}. Contents: {data}"
        )
    return {key: data[key] for key in _SUMMARY_KEYS}
