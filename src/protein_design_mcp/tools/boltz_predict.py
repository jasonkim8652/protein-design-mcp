"""
Boltz structure prediction and affinity tools.

Provides two tools:
- predict_structure_boltz: Single-chain structure prediction
- predict_affinity_boltz: Multi-chain affinity prediction
"""

from typing import Any

from protein_design_mcp.pipelines.boltz_runner import BoltzRunner, BoltzConfig


async def predict_structure_boltz(
    sequence: str,
    model: str = "boltz2",
    num_samples: int = 1,
) -> dict[str, Any]:
    """Predict protein structure using Boltz.

    Args:
        sequence: Amino acid sequence.
        model: Model name (default: boltz2).
        num_samples: Number of structure samples.

    Returns:
        Dict with predicted_structure_pdb, plddt, ptm.
    """
    config = BoltzConfig(model=model, num_samples=num_samples)
    runner = BoltzRunner(config=config)
    return await runner.predict_structure(
        sequence=sequence,
        model=model,
        num_samples=num_samples,
    )


async def predict_affinity_boltz(
    sequences: list[str],
    model: str = "boltz2",
) -> dict[str, Any]:
    """Predict binding affinity for a protein complex using Boltz.

    Args:
        sequences: List of amino acid sequences (one per chain).
        model: Model name (default: boltz2).

    Returns:
        Dict with affinity_score, predicted structure, confidence metrics.
    """
    config = BoltzConfig(model=model)
    runner = BoltzRunner(config=config)
    return await runner.predict_affinity(
        sequences=sequences,
        model=model,
    )
