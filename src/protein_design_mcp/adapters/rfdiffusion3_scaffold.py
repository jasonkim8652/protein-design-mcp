"""Adapter for RFdiffusion3's monomer/scaffold path
(`scripts/engines/rfd3.py`, env `foundry`).

Shares the wrapper and the underlying `rfd3 design` entry point with
`run_rfdiffusion3_binder` (see that adapter's module docstring for the
JSON shape both build); this one never sets `select_hotspots` since there
is no target.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.schema import Manifest


def build_args(manifest: Manifest, params: dict[str, Any]) -> list[str]:
    """``manifest`` is unused (this adapter backs exactly one tool) but is
    part of every adapter's signature -- see ``adapters_discovery``.
    """
    del manifest

    motif_pdb = params.get("motif_pdb")
    motif_contig = params.get("motif_contig")
    if bool(motif_pdb) != bool(motif_contig):
        raise ValueError(
            "motif_pdb and motif_contig must be given together (motif "
            "scaffolding) or both omitted (unconditional generation); got "
            f"motif_pdb={motif_pdb!r}, motif_contig={motif_contig!r}."
        )

    job: dict[str, Any] = {
        "length": params["length"],
        "redesign_motif_sidechains": params["redesign_motif_sidechains"],
    }
    if motif_pdb:
        job["input"] = motif_pdb
        job["contig"] = motif_contig

    engine = {
        "diffusion_batch_size": params["diffusion_batch_size"],
        "num_timesteps": params["num_timesteps"],
        "step_scale": params["step_scale"],
        "seed": params.get("seed"),
    }

    return [json.dumps({"job": job, "engine": engine})]


def parse_output(manifest: Manifest, run: CompletedRun) -> dict[str, Any]:
    """Read the rank-0 (first) sample's metadata sidecar plus the collected
    output paths. ``manifest`` is unused (see ``build_args``).
    """
    del manifest

    metadata_paths = run.outputs.get("metadata_json")
    if not metadata_paths:
        raise ValueError(
            "run_rfdiffusion3_scaffold's declared 'metadata_json' output "
            f"was not collected -- no metadata sidecar was found. "
            f"run.outputs was: {run.outputs}"
        )
    if not isinstance(metadata_paths, list):
        metadata_paths = [metadata_paths]

    structures = run.outputs.get("structure_cif")
    num_structures = len(structures) if isinstance(structures, list) else (
        1 if structures else 0
    )

    first = json.loads(Path(sorted(metadata_paths)[0]).read_text())

    return {
        "num_structures": num_structures,
        "metrics": first.get("metrics"),
        "ckpt_path": first.get("ckpt_path"),
    }
