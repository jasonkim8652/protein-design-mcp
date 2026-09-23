"""Adapter for RFdiffusion2 (`scripts/engines/rfdiffusion2.py`, env
`rfd2_fixed`). Dispatches through the `rfd2_fixed` conda prefix by default
(`params["backend"] == "conda"`, the manifest's default); the wrapper can
also launch the official container image as a sibling `docker run` when
the caller explicitly sets `backend: "docker"` -- see the wrapper's own
module docstring for why that mode is opt-in, never a silent fallback.

Serialises the validated parameters into one JSON argv token, mirroring
`adapters.boltz`'s reasoning: the wrapper needs a writable scratch cwd and
a resolved GPU/user context that only exist once dispatch has actually
started the subprocess, so the job description travels as data rather than
being partially built here.
"""

from __future__ import annotations

import json
from typing import Any

from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.schema import Manifest


def build_args(manifest: Manifest, params: dict[str, Any]) -> list[str]:
    """``manifest`` is unused (this adapter backs exactly one tool) but is
    part of every adapter's signature -- see ``adapters_discovery``.
    """
    del manifest

    job = {
        "target_pdb": params["target_pdb"],
        "contig": params["contig"],
        "backend": params["backend"],
        "num_designs": params["num_designs"],
        "diffusion_steps": params["diffusion_steps"],
        "noise_scale_ca": params["noise_scale_ca"],
        "noise_scale_frame": params["noise_scale_frame"],
        "ckpt_variant": params["ckpt_variant"],
    }
    return [json.dumps(job)]


def parse_output(manifest: Manifest, run: CompletedRun) -> dict[str, Any]:
    """``manifest`` is unused (see ``build_args``)."""
    del manifest

    structures = run.outputs.get("structures")
    if not structures:
        raise ValueError(
            "run_rfdiffusion2's declared 'structures' output was not "
            f"collected -- no backbone PDB was found. run.outputs was: "
            f"{run.outputs}"
        )
    if not isinstance(structures, list):
        structures = [structures]

    return {
        "num_structures": len(structures),
        "sequence_caveat": (
            "Backbone only -- diffused (binder) residues carry no real "
            "sequence. Run run_mpnn on these structures to design an "
            "actual sequence."
        ),
    }
