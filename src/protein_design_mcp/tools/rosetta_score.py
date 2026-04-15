"""
Rosetta energy scoring tool.

Scores a PDB structure using a Rosetta energy function (default: ref2015).
Returns total score, per-residue energies, and energy component breakdown.
"""

from typing import Any

from protein_design_mcp.pipelines.pyrosetta_runner import PyRosettaRunner, PyRosettaConfig


async def rosetta_score(
    pdb_path: str,
    score_function: str = "ref2015",
) -> dict[str, Any]:
    """Score a protein structure using Rosetta energy function.

    Args:
        pdb_path: Path to input PDB file.
        score_function: Rosetta score function name (default: ref2015).

    Returns:
        Dict with total_score, per_residue, energy_components.
    """
    config = PyRosettaConfig(score_function=score_function)
    runner = PyRosettaRunner(config=config)
    return await runner.score(pdb_path=pdb_path, score_fn=score_function)
