"""
Energy minimization tool using OpenMM.

Provides AMBER14 all-atom energy minimization with optional implicit
solvent (GBn2).  Useful for refining predicted or designed structures.
"""

from typing import Any

from protein_design_mcp.pipelines.openmm_runner import OpenMMRunner, OpenMMConfig


async def energy_minimize(
    pdb_path: str,
    force_field: str = "amber14-all.xml",
    num_steps: int = 500,
    solvent: str = "implicit",
) -> dict[str, Any]:
    """Energy-minimize a protein structure.

    Args:
        pdb_path: Path to input PDB file.
        force_field: OpenMM force field XML (default: ``"amber14-all.xml"``).
        num_steps: Maximum minimization iterations.
        solvent: ``"implicit"`` (GBn2) or ``"none"`` (vacuum).

    Returns:
        Dict with ``minimized_pdb_path``, ``initial_energy``, ``final_energy``,
        ``energy_change``, ``rmsd_from_initial``.
    """
    config = OpenMMConfig(
        force_field=force_field,
        solvent=solvent,
        max_iterations=num_steps,
    )
    runner = OpenMMRunner(config=config)

    return await runner.minimize(
        pdb_path=pdb_path,
        num_steps=num_steps,
    )
