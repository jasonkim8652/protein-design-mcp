"""Adapter for run_epitope_scan.

All the actual work happens in the wrapper script
(``scripts/engines/epitope_scan.py``), which is this server's own
``protein_design_mcp.utils.sasa``/``utils.conservation`` code exposed as a
tool, not a new algorithm. This adapter only serializes validated
parameters into the wrapper's argv and reads its ``results.json`` back.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.schema import Manifest


def build_args(manifest: Manifest, params: dict[str, Any]) -> list[str]:
    """Translate validated parameters into the wrapper script's argv.

    ``msa`` is nullable (see the manifest doc): ``None`` is passed through
    as the literal string ``"null"``, which the wrapper checks for
    explicitly, since argv has no way to carry a real ``None``.
    """
    del manifest
    args = [
        "--target-pdb",
        str(params["target_pdb"]),
        "--chain",
        str(params["chain"]),
        "--exposure-threshold-a2",
        str(params["exposure_threshold_a2"]),
        "--conserved-threshold",
        str(params["conserved_threshold"]),
        "--top-n-hotspots",
        str(params["top_n_hotspots"]),
    ]
    if params["msa"] is not None:
        args += ["--msa", str(params["msa"])]
    return args


def parse_output(manifest: Manifest, run: CompletedRun) -> dict[str, Any]:
    """Read the wrapper's ``results.json`` back. ``manifest`` is unused (see
    ``build_args``)."""
    del manifest
    path = run.outputs.get("results_json")
    if not path:
        raise ValueError(
            "run_epitope_scan's declared 'results_json' output was not "
            f"collected. run.outputs was: {run.outputs}"
        )
    return json.loads(Path(path).read_text())
