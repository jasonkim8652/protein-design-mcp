"""
Rosetta interface scoring tool.

Computes interface energy metrics for a protein complex using Rosetta's
InterfaceAnalyzerMover: binding energy (dG_separated), buried surface
area (dSASA), interface hydrogen bonds, and packing statistics.
"""

from typing import Any

from protein_design_mcp.pipelines.pyrosetta_runner import PyRosettaRunner, PyRosettaConfig


async def rosetta_interface_score(
    pdb_path: str,
    chains: str = "A_B",
    score_function: str = "ref2015",
) -> dict[str, Any]:
    """Score a protein-protein interface using Rosetta.

    Args:
        pdb_path: Path to complex PDB file.
        chains: Chain grouping, e.g. "A_B" or "AB_C".
        score_function: Rosetta score function name.

    Returns:
        Dict with dG_separated, dSASA, interface_hbonds, packstat, etc.
    """
    config = PyRosettaConfig(score_function=score_function)
    runner = PyRosettaRunner(config=config)
    return await runner.interface_score(
        pdb_path=pdb_path,
        chains=chains,
        score_fn=score_function,
    )
