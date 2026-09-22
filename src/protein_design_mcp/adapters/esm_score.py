"""Adapter for ESM2-650M pseudo-log-likelihood scoring
(``scripts/engines/esm_score.py``, env ``esm_env``).

The wrapper prints one JSON object as the LAST line of stdout (library
import warnings from torch/esm sometimes land on stdout before that, so the
adapter reads only the last line -- the same convention
``protein_design_mcp.mounts``'s own probe script already uses for exactly
this reason).
"""

from __future__ import annotations

import json
from typing import Any

from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.schema import Manifest

CAVEAT = (
    "This is a developability proxy (masked-marginal pseudo-log-likelihood "
    "under ESM2-650M) -- it reflects similarity to natural sequence "
    "statistics, not whether this sequence will fold, express, or bind its "
    "target. Never use it as the sole or primary ranking criterion for "
    "binder designs; use run_ipsae/run_prodigy/run_rosetta_interface for "
    "interface quality instead."
)


def build_args(manifest: Manifest, params: dict[str, Any]) -> list[str]:
    """Serialise validated parameters to one JSON argv token. ``manifest``
    is unused here -- this adapter backs exactly one tool -- but is part of
    every adapter's signature (see ``adapters_discovery``).
    """
    del manifest
    job = {
        "sequence": params["sequence"],
        "batch_size": params["batch_size"],
    }
    return [json.dumps(job)]


def parse_output(manifest: Manifest, run: CompletedRun) -> dict[str, Any]:
    """Read the wrapper's JSON result line and attach the developability
    caveat. ``manifest`` is unused (see ``build_args``).
    """
    del manifest
    lines = run.stdout.strip().splitlines()
    if not lines:
        raise ValueError(
            "run_esm_score produced no output on stdout. stderr was:\n"
            f"{run.stderr.strip()[-1000:]}"
        )
    try:
        result = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise ValueError(
            "run_esm_score's last stdout line was not valid JSON: "
            f"{lines[-1]!r}"
        ) from exc

    for key in ("pseudo_log_likelihood", "per_residue_log_likelihood", "sequence_length"):
        if key not in result:
            raise ValueError(
                f"run_esm_score's output JSON is missing {key!r}. Got: {result}"
            )

    return {
        "pseudo_log_likelihood": result["pseudo_log_likelihood"],
        "per_residue_log_likelihood": result["per_residue_log_likelihood"],
        "sequence_length": result["sequence_length"],
        "device": result.get("device"),
        "caveat": CAVEAT,
    }
