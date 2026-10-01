"""Adapter for the OpenMM minimisation script run inside the `md` environment."""

from __future__ import annotations

import gzip
import json
import math
import re
import zlib
from pathlib import Path
from typing import Any

from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.schema import Manifest
from protein_design_mcp.staging import validate_source_file
from protein_design_mcp.validation import ToolInputError

OUTPUT_NAME = "minimized.pdb"

# Canonical residue heavy atoms, independent of protonation and force field.
# OXT is deliberately optional: the native engine prepares missing terminals.
_BACKBONE = frozenset({"N", "CA", "C", "O"})
_CANONICAL_HEAVY_ATOMS = {
    name: _BACKBONE | frozenset(sidechain.split())
    for name, sidechain in {
        "ALA": "CB",
        "ARG": "CB CG CD NE CZ NH1 NH2",
        "ASN": "CB CG OD1 ND2",
        "ASP": "CB CG OD1 OD2",
        "CYS": "CB SG",
        "GLN": "CB CG CD OE1 NE2",
        "GLU": "CB CG CD OE1 OE2",
        "GLY": "",
        "HIS": "CB CG ND1 CD2 CE1 NE2",
        "ILE": "CB CG1 CG2 CD1",
        "LEU": "CB CG CD1 CD2",
        "LYS": "CB CG CD CE NZ",
        "MET": "CB CG SD CE",
        "PHE": "CB CG CD1 CD2 CE1 CE2 CZ",
        "PRO": "CB CG CD",
        "SER": "CB OG",
        "THR": "CB OG1 CG2",
        "TRP": "CB CG CD1 CD2 NE1 CE2 CE3 CZ2 CZ3 CH2",
        "TYR": "CB CG CD1 CD2 CE1 CE2 CZ OH",
        "VAL": "CB CG1 CG2",
    }.items()
}


def _require_complete_residues(structure: Path) -> None:
    """Check the first model's canonical residues before engine dispatch.

    Require every nonterminal heavy atom, including partial side chains and
    backbone atoms. Hydrogens and terminal OXT are optional. Unknown/modified
    residue templates and force-field compatibility remain the engine's job;
    this check neither reconstructs coordinates nor verifies the sequence.
    """
    from Bio.PDB import MMCIFParser, PDBParser
    from Bio.PDB.PDBExceptions import PDBConstructionException

    label = "run_openmm_minimize.input_pdb"
    structure = validate_source_file(structure, label)
    compressed = structure.suffix.lower() == ".gz"
    suffix = structure.with_suffix("").suffix.lower() if compressed else structure.suffix.lower()
    opener = gzip.open if compressed else open
    parser = (MMCIFParser(QUIET=True) if suffix in {".cif", ".mmcif"}
              else PDBParser(QUIET=True, PERMISSIVE=False))
    try:
        with opener(structure, "rt") as handle:
            model = next(parser.get_structure("input", handle).get_models())
    except (PDBConstructionException, ValueError, KeyError, IndexError,
            StopIteration, EOFError, gzip.BadGzipFile, zlib.error, UnicodeError) as exc:
        # Do not include parser text: it can contain unbounded input content.
        # Other OSErrors (permission/storage failures) remain infrastructure errors.
        raise ToolInputError(
            f"{label}: cannot parse a nonempty PDB or CIF/mmCIF structure. "
            "Supply a readable structure file in the declared format."
        ) from exc

    residue_count = 0
    missing_count = 0
    atom_count = 0
    details = []
    for chain in model:
        for residue in chain:
            residue_count += 1
            expected = _CANONICAL_HEAVY_ATOMS.get(residue.resname)
            if expected is None:
                continue
            missing = expected - {atom.name for atom in residue}
            if not missing:
                continue
            missing_count += 1
            atom_count += len(missing)
            if len(details) < 10:
                # CIF identifiers may be arbitrarily long; bound each field too.
                number = str(residue.id[1])[:16] + residue.id[2].strip()[:8]
                identity = f"{residue.resname} {chain.id[:24]}{number}"
                details.append(f"{identity} missing {', '.join(sorted(missing))}")
    if not residue_count:
        raise ToolInputError(f"{label}: structure contains no residues in its first model.")
    if missing_count:
        remainder = f"; {missing_count - len(details)} more residues" if missing_count > len(details) else ""
        raise ToolInputError(
            f"{label}: {missing_count} residues have {atom_count} missing "
            f"nonterminal heavy atoms: {'; '.join(details)}{remainder}. "
            "Supply a structure with complete backbone and side-chain heavy atoms "
            "for the intended sequence in every chain. Hydrogens and terminal OXT "
            "may be absent; internal missing atoms are not reconstructed."
        )


_INITIAL_RE = re.compile(r"initial_potential_energy_kj_mol:\s*(\S+)")
_FINAL_RE = re.compile(r"final_potential_energy_kj_mol:\s*(\S+)")
_ITER_RE = re.compile(r"iterations:\s*(\d+)")
_TERMINAL_RE = re.compile(r"added_terminal_atoms:\s*(\d+)")


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
        "--platform", str(params.get("platform", "CUDA")),
        "--precision", str(params.get("precision", "double")),
    ]


def parse_output(manifest: Manifest, run: CompletedRun) -> dict[str, Any]:
    """Extract the energies the engine script printed."""
    # The dispatcher removes scratch before parsing: prefer collected outputs.
    diagnostics_path = Path(run.outputs.get("minimization_diagnostics", run.workdir / "openmm_diagnostics.json"))
    diagnostics = json.loads(diagnostics_path.read_text()) if diagnostics_path.is_file() else None
    if diagnostics is not None and diagnostics.get("status") == "numerical_failure":
        stages = diagnostics.get("stages", [])
        final = stages[-1] if stages else {}
        platform = final.get("platform", diagnostics.get("requested_platform"))
        native_precision = (diagnostics.get("requested_precision") if platform == "CUDA"
                            else "double" if platform == "Reference" else "platform_default")
        precision = final.get("platform_properties", {}).get("Precision", native_precision)
        return {
            "initial_potential_energy_kj_mol": diagnostics.get("initial_potential_energy_kj_mol"),
            "final_potential_energy_kj_mol": None, "energy_change_kj_mol": None,
            "iterations": sum(s["reporter_calls"] for s in stages),
            "numerical_failure": True, "geometry_passed": False,
            "minimization_protocol": diagnostics.get("protocol"),
            "platform": platform,
            "precision": precision,
            "minimization_diagnostics": diagnostics,
        }
    if diagnostics is not None and diagnostics.get("status") in {"completed", "geometry_failed"}:
        minimized = run.outputs.get("minimized_pdb")
        if not isinstance(minimized, str) or not Path(minimized).is_file():
            raise ValueError("OpenMM completed without its collected minimized PDB")
    initial = _INITIAL_RE.search(run.stdout)
    final = _FINAL_RE.search(run.stdout)
    if initial is None or final is None:
        raise ValueError(
            "OpenMM minimisation printed no energy lines. Output was:\n"
            f"{run.stdout.strip()[-1000:]}"
        )

    initial_value = float(initial.group(1))
    final_value = float(final.group(1))
    if not math.isfinite(initial_value) or not math.isfinite(final_value):
        raise ValueError("OpenMM reported nonfinite energy")
    iterations = _ITER_RE.search(run.stdout)
    terminals = _TERMINAL_RE.search(run.stdout)
    result = {
        "initial_potential_energy_kj_mol": initial_value,
        "final_potential_energy_kj_mol": final_value,
        "energy_change_kj_mol": final_value - initial_value,
        "iterations": int(iterations.group(1)) if iterations else None,
        "added_terminal_atoms": int(terminals.group(1)) if terminals else None,
    }
    for key in ("force_field", "solvent_model", "energy_units"):
        match = re.search(rf"(?m)^{key}:\s*(\S+)", run.stdout)
        if match:
            result[key] = match.group(1)
    if diagnostics is not None:
        result["minimization_diagnostics"] = diagnostics
        result["minimization_protocol"] = diagnostics.get("protocol")
        result["geometry_passed"] = diagnostics.get("geometry_passed", False)
        result["iterations"] = diagnostics.get("iterations")
        if diagnostics.get("stages"):
            final_stage = diagnostics["stages"][-1]
            result["platform"] = final_stage["platform"]
            result["precision"] = final_stage["platform_properties"].get("Precision", "double" if result["platform"] == "Reference" else "platform_default")
    return result
