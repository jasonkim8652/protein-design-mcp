"""Adapter for PyRosetta's InterfaceAnalyzerMover
(``scripts/engines/rosetta_interface.py``, env ``pyrosetta``).

The wrapper prints one JSON object as the LAST line of stdout (PyRosetta is
extremely chatty on stdout/stderr even with ``-mute all``; the adapter reads
only the last line -- the same convention ``run_esm_score``'s adapter and
``protein_design_mcp.mounts``'s own probe script already use for exactly
this reason).

See the manifest's "Verification status" section for why this tool cannot
currently be exercised end to end on this host (the shipped wheel is
missing its compiled extension) -- that does not change what this adapter
does: it is built against the API this wave verified live in a different,
working PyRosetta install (see the manifest doc and the wave report).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.schema import Manifest

_REQUIRED_RESULT_KEYS = (
    "dG",
    "dSASA",
    "shape_complementarity",
    "interface_hbonds",
    "delta_unsat_hbonds",
    "num_interface_residues",
)


def build_args(manifest: Manifest, params: dict[str, Any]) -> list[str]:
    """Serialise validated parameters to one JSON argv token. ``manifest``
    is unused here -- this adapter backs exactly one tool -- but is part of
    every adapter's signature (see ``adapters_discovery``).
    """
    del manifest
    # InterfaceAnalyzerMover can return zero-valued scores for a monomer
    # whose requested partner does not exist. Refuse that input before launch.
    from Bio.PDB import MMCIFParser, PDBParser

    groups = [set(group) for group in params["interface"].split("_")]
    if len(groups) != 2 or not all(groups) or groups[0] & groups[1]:
        raise ValueError("Interface chain groups must be nonempty and disjoint")
    path = Path(params["complex_pdb"])
    parser = MMCIFParser(QUIET=True) if path.suffix.lower() == ".cif" else PDBParser(QUIET=True)
    structure = parser.get_structure("interface", str(path))
    model = next(structure.get_models())
    present = {chain.id for chain in model if any(residue.id[0] == " " for residue in chain)}
    missing = set.union(*groups) - present
    if missing:
        raise ValueError(f"Interface {params['interface']} has missing protein chains {sorted(missing)}; "
                         f"chains present: {sorted(present)}. Supply the predicted complex structure.")
    job = {
        "complex_pdb": params["complex_pdb"],
        "interface": params["interface"],
        "score_function": params["score_function"],
        "pack_separated": params["pack_separated"],
        "pack_input": params["pack_input"],
        "pack_rounds": params["pack_rounds"],
        "compute_packstat": params["compute_packstat"],
        "compute_interface_sc": params["compute_interface_sc"],
    }
    return [json.dumps(job)]


def parse_output(manifest: Manifest, run: CompletedRun) -> dict[str, Any]:
    """Read the wrapper's JSON result line. ``manifest`` is unused (see
    ``build_args``)."""
    del manifest
    lines = run.stdout.strip().splitlines()
    if not lines:
        raise ValueError(
            "run_rosetta_interface produced no output on stdout. stderr was:\n"
            f"{run.stderr.strip()[-1000:]}"
        )
    try:
        result = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise ValueError(
            "run_rosetta_interface's last stdout line was not valid JSON: "
            f"{lines[-1]!r}"
        ) from exc

    missing = [key for key in _REQUIRED_RESULT_KEYS if key not in result]
    if missing:
        raise ValueError(
            f"run_rosetta_interface's output JSON is missing {missing}. "
            f"Got: {result}"
        )

    return {
        "dG": result["dG"],
        "dSASA": result["dSASA"],
        "shape_complementarity": result["shape_complementarity"],
        "interface_hbonds": result["interface_hbonds"],
        "delta_unsat_hbonds": result["delta_unsat_hbonds"],
        "num_interface_residues": result["num_interface_residues"],
        "packstat": result.get("packstat"),
        "interface": result.get("interface"),
        "score_function": result.get("score_function"),
    }
