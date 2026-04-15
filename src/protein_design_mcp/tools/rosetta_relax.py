"""
Rosetta FastRelax tool.

Relaxes a PDB structure using Rosetta FastRelax protocol to find a
low-energy conformation.  Returns the relaxed PDB, energy change,
and CA-RMSD from the initial structure.
"""

from typing import Any

from protein_design_mcp.pipelines.pyrosetta_runner import PyRosettaRunner, PyRosettaConfig


async def rosetta_relax(
    pdb_path: str,
    nstruct: int = 1,
    score_function: str = "ref2015",
) -> dict[str, Any]:
    """Relax a protein structure with Rosetta FastRelax.

    Args:
        pdb_path: Path to input PDB file.
        nstruct: Number of relaxation trajectories (best is kept).
        score_function: Rosetta score function name.

    Returns:
        Dict with relaxed_pdb_path, energy_before, energy_after, ca_rmsd.
    """
    config = PyRosettaConfig(
        score_function=score_function,
        relax_nstruct=nstruct,
    )
    runner = PyRosettaRunner(config=config)
    return await runner.relax(
        pdb_path=pdb_path,
        nstruct=nstruct,
        score_fn=score_function,
    )
