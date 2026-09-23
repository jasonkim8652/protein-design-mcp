"""Adapter for the OpenMM minimisation script run inside the `md` environment."""

from __future__ import annotations

import re
from typing import Any

from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.schema import Manifest

OUTPUT_NAME = "minimized.pdb"

_INITIAL_RE = re.compile(r"initial_potential_energy_kj_mol:\s*(-?[\d.]+)")
_FINAL_RE = re.compile(r"final_potential_energy_kj_mol:\s*(-?[\d.]+)")
_ITER_RE = re.compile(r"iterations:\s*(\d+)")


def build_args(manifest: Manifest, params: dict[str, Any]) -> list[str]:
    """Translate validated parameters into the engine script's argv.

    The output path is relative, so the file lands in the dispatcher's scratch
    directory where the manifest's ``outputs:`` pattern can find it.
    """
    return [
        str(params["input_pdb"]),
        OUTPUT_NAME,
        "--max-iterations",
        str(params["max_iterations"]),
        "--forcefield",
        str(params["forcefield"]),
    ]


def parse_output(manifest: Manifest, run: CompletedRun) -> dict[str, Any]:
    """Extract the energies the engine script printed."""
    initial = _INITIAL_RE.search(run.stdout)
    final = _FINAL_RE.search(run.stdout)
    if initial is None or final is None:
        raise ValueError(
            "OpenMM minimisation printed no energy lines. Output was:\n"
            f"{run.stdout.strip()[-1000:]}"
        )

    initial_value = float(initial.group(1))
    final_value = float(final.group(1))
    iterations = _ITER_RE.search(run.stdout)
    return {
        "initial_potential_energy_kj_mol": initial_value,
        "final_potential_energy_kj_mol": final_value,
        "energy_change_kj_mol": final_value - initial_value,
        "iterations": int(iterations.group(1)) if iterations else None,
    }
