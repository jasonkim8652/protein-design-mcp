"""Adapter for RFdiffusion3's binder (PPI) path
(`scripts/engines/rfd3.py`, env `foundry`).

Builds the one JSON argv the shared wrapper expects: a `DesignInputSpecification`-
shaped `"job"` dict (RFdiffusion3's own pydantic conditioning schema -- see
`rfd3.inference.input_parsing.DesignInputSpecification`, which is
`extra="forbid"`, so only its recognised field names may appear) plus an
`"engine"` dict of diffusion-sampling knobs that are separate Hydra overrides,
not part of that JSON object at all.
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

    job: dict[str, Any] = {
        "input": params["target_pdb"],
        "contig": params["contig"],
        "redesign_motif_sidechains": params["redesign_motif_sidechains"],
    }
    if params.get("select_hotspots") is not None:
        job["select_hotspots"] = params["select_hotspots"]
    if params.get("length") is not None:
        job["length"] = params["length"]
    if params.get("partial_t") is not None:
        job["partial_t"] = params["partial_t"]

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
            "run_rfdiffusion3_binder's declared 'metadata_json' output was "
            f"not collected -- no metadata sidecar was found. run.outputs "
            f"was: {run.outputs}"
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
        "diffused_index_map": first.get("diffused_index_map"),
        "ckpt_path": first.get("ckpt_path"),
    }
