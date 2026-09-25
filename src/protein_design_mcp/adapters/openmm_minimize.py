"""Adapter for the OpenMM minimisation script run inside the `md` environment."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.schema import Manifest
from protein_design_mcp.validation import ToolInputError

OUTPUT_NAME = "minimized.pdb"

#: The heavy atoms every residue has regardless of type. GLY has ONLY these,
#: so a backbone-only GLY is complete and must not be refused.
_BACKBONE = frozenset({"N", "CA", "C", "O", "OXT"})
_NO_SIDE_CHAIN = frozenset({"GLY"})


def _require_complete_residues(structure: Path) -> None:
    """Refuse a structure whose residues are missing their side chains.

    OpenMM's ``Modeller.addHydrogens`` needs every residue's full heavy-atom
    set and reports a shortfall from deep inside itself, naming an index the
    caller never chose::

        ValueError: HIS residue (118) has the wrong set of atoms

    A live round reached that by minimising ``run_rebuild_backbone``'s output
    -- a reconstructed BACKBONE, which is exactly right for a sequence
    designer and unusable for molecular mechanics. Minimise the FOLDED
    complex, not the backbone it was designed on.

    Deliberately shallow: it only asks whether a residue that should have a
    side chain has any atom beyond the backbone. Checking each residue type's
    full atom set is the force field's job, and a guess at it would refuse
    real structures. Unreadable files pass through -- this is a precondition,
    not a PDB validator.
    """
    try:
        text = structure.read_text(errors="replace")
    except OSError:
        return
    residues: dict[tuple[str, str, str], set[str]] = {}
    for line in text.splitlines():
        if line.startswith("ENDMDL"):
            break
        if not line.startswith(("ATOM", "HETATM")) or len(line) < 27:
            continue
        key = (line[21], line[22:27].strip(), line[17:20].strip())
        residues.setdefault(key, set()).add(line[12:16].strip())
    for (chain, number, resname), atoms in residues.items():
        if resname in _NO_SIDE_CHAIN or not atoms:
            continue
        if atoms - _BACKBONE:
            continue
        raise ToolInputError(
            f"run_openmm_minimize.input_pdb = {str(structure)!r} has residue "
            f"{resname} {chain}{number} with backbone atoms only "
            f"({', '.join(sorted(atoms))}) and no side chain. OpenMM needs "
            "every residue's complete heavy-atom set and fails inside its own "
            "Modeller otherwise. This is what a reconstructed backbone looks "
            "like (see run_rebuild_backbone): minimise the FOLDED structure a "
            "co-folding tool produced, not the backbone a design was built on."
        )

_INITIAL_RE = re.compile(r"initial_potential_energy_kj_mol:\s*(-?[\d.]+)")
_FINAL_RE = re.compile(r"final_potential_energy_kj_mol:\s*(-?[\d.]+)")
_ITER_RE = re.compile(r"iterations:\s*(\d+)")


def build_args(manifest: Manifest, params: dict[str, Any]) -> list[str]:
    """Translate validated parameters into the engine script's argv.

    The output path is relative, so the file lands in the dispatcher's scratch
    directory where the manifest's ``outputs:`` pattern can find it.
    """
    _require_complete_residues(Path(str(params["input_pdb"])))
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
