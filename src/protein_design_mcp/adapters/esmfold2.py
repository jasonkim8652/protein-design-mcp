"""Adapter for ESMFold2 (esm==3.4.0, esm.models.esmfold2.EsmFold2Model).

The wrapper script (``scripts/engines/esmfold2.py``) prints ``key: value``
lines the same way ``openmm_minimize.py`` does, and writes the predicted
structure to a fixed relative filename in the dispatcher's scratch working
directory -- there is no other machine-readable channel (ESMFold2's Python
API returns in-memory tensors/a PDB string, not files).
"""

from __future__ import annotations

import re
from typing import Any

from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.schema import Manifest

OUTPUT_NAME = "model.pdb"

_PLDDT_RE = re.compile(r"mean_plddt:\s*(-?[\d.]+)")
_NRES_RE = re.compile(r"num_residues:\s*(\d+)")
_SEQLEN_RE = re.compile(r"sequence_length:\s*(\d+)")


def build_args(manifest: Manifest, params: dict[str, Any]) -> list[str]:
    """Translate validated parameters into the engine script's argv.

    ``manifest`` is unused here -- ESMFold2 backs exactly one tool -- but is
    part of every adapter's signature (see ``protein_design_mcp.app.ADAPTERS``,
    keyed on manifest.name) so a future adapter module backing several tools
    on one engine repo can branch on ``manifest.name`` without changing the
    call site.
    """
    del manifest
    return [
        str(params["sequence"]),
        OUTPUT_NAME,
        "--num-recycles",
        str(params["num_recycles"]),
        "--num-diffusion-samples",
        str(params["num_diffusion_samples"]),
        "--num-sampling-steps",
        str(params["num_sampling_steps"]),
    ]


def parse_output(manifest: Manifest, run: CompletedRun) -> dict[str, Any]:
    """Extract the confidence summary the engine script printed. ``manifest``
    is unused (see ``build_args``)."""
    del manifest
    plddt = _PLDDT_RE.search(run.stdout)
    if plddt is None:
        raise ValueError(
            "ESMFold2 wrapper printed no mean_plddt line. Output was:\n"
            f"{run.stdout.strip()[-1000:]}"
        )
    nres = _NRES_RE.search(run.stdout)
    seqlen = _SEQLEN_RE.search(run.stdout)
    return {
        "mean_plddt": float(plddt.group(1)),
        "num_residues": int(nres.group(1)) if nres else None,
        "sequence_length": int(seqlen.group(1)) if seqlen else None,
    }
