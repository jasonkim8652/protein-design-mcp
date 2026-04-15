"""
Rosetta fixed-backbone design tool (composite).

Convenience pipeline that chains: score → PackRotamers → MinMover → score.
In benchmark mode, agents must call rosetta_score, rosetta_relax, etc.
individually to demonstrate orchestration skill.
"""

from typing import Any

from protein_design_mcp.pipelines.pyrosetta_runner import PyRosettaRunner, PyRosettaConfig


async def rosetta_design(
    pdb_path: str,
    chains: str = "A_B",
    fixed_positions: list[int] | None = None,
    score_function: str = "ref2015",
) -> dict[str, Any]:
    """Design sequences using Rosetta fixed-backbone protocol.

    Args:
        pdb_path: Path to input PDB file.
        chains: Chain grouping for interface detection.
        fixed_positions: 1-indexed residue positions to keep fixed.
        score_function: Rosetta score function name.

    Returns:
        Dict with designed_pdb_path, sequence, mutations, energy change.
    """
    config = PyRosettaConfig(score_function=score_function)
    runner = PyRosettaRunner(config=config)
    return await runner.design(
        pdb_path=pdb_path,
        chains=chains,
        fixed_positions=fixed_positions,
        score_fn=score_function,
    )
