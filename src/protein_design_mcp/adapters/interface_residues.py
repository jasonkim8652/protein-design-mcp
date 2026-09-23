"""Adapter for run_interface_residues.

All the actual work happens in the wrapper script
(``scripts/engines/interface_residues.py``), which is this server's own
``protein_design_mcp.utils.pdb``/``utils.sasa`` code exposed as a tool, not
a new algorithm. This adapter only serializes validated parameters into the
wrapper's argv and reads its ``results.json`` back.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.schema import Manifest


def build_args(manifest: Manifest, params: dict[str, Any]) -> list[str]:
    """Translate validated parameters into the wrapper script's argv.

    ``manifest`` is unused here -- this adapter backs exactly one tool --
    but is part of every adapter's signature (see run_prodigy's adapter for
    the full reasoning).
    """
    del manifest
    return [
        "--complex-pdb",
        str(params["complex_pdb"]),
        "--target-chain",
        str(params["target_chain"]),
        "--binder-chains",
        json.dumps(list(params["binder_chains"])),
        "--contact-cutoff",
        str(params["contact_cutoff"]),
    ]


def parse_output(manifest: Manifest, run: CompletedRun) -> dict[str, Any]:
    """Read the wrapper's ``results.json`` back. ``manifest`` is unused (see
    ``build_args``)."""
    del manifest
    path = run.outputs.get("results_json")
    if not path:
        raise ValueError(
            "run_interface_residues's declared 'results_json' output was "
            f"not collected. run.outputs was: {run.outputs}"
        )
    return json.loads(Path(path).read_text())
