"""Adapter for La-Proteina's unconditional sampler (`run_la_proteina`).

Almost all translation work happens in the wrapper script
(``scripts/engines/la_proteina.py``) -- building La-Proteina's own Hydra
config pair, running the engine with the repo root as cwd (required for its
own import mechanism), and copying results back into this call's actual
scratch workdir. This adapter only serializes parameters into the wrapper's
argv and reads the collected PDBs back.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.schema import Manifest


def _bool_str(value: bool) -> str:
    return "true" if value else "false"


def build_args(manifest: Manifest, params: dict[str, Any]) -> list[str]:
    """Translate validated parameters into the wrapper script's argv."""
    del manifest
    return [
        "--lengths",
        json.dumps(list(params["lengths"])),
        "--num-samples",
        str(params["num_samples"]),
        "--max-nsamples-per-batch",
        str(params["max_nsamples_per_batch"]),
        "--nsteps",
        str(params["nsteps"]),
        "--self-cond",
        _bool_str(params["self_cond"]),
        "--sc-scale-noise",
        str(params["sc_scale_noise"]),
        "--sc-scale-score",
        str(params["sc_scale_score"]),
        "--guidance-w",
        str(params["guidance_w"]),
        "--seed",
        str(params["seed"]),
    ]


def _length_from_pdb(path: str) -> int:
    count = 0
    for line in Path(path).read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")) and line[12:16].strip() == "CA":
            count += 1
    return count


def parse_output(manifest: Manifest, run: CompletedRun) -> dict[str, Any]:
    """Read every generated backbone's length back out of its own PDB."""
    del manifest
    paths = run.outputs.get("backbones")
    if not paths:
        raise ValueError(
            "run_la_proteina's declared 'backbones' output was not "
            f"collected -- no PDB was found. run.outputs was: {run.outputs}"
        )
    if not isinstance(paths, list):
        paths = [paths]

    backbones = [
        {"id": Path(path).stem, "length": _length_from_pdb(path)}
        for path in sorted(paths)
    ]

    return {"backbones": backbones, "num_backbones": len(backbones)}
