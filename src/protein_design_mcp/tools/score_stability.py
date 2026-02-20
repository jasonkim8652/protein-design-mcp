"""
Stability scoring tool using ESM2 pseudo-log-likelihood.

Provides a proxy for thermodynamic stability without requiring
PyRosetta or Rosetta installation.
"""

from typing import Any

from protein_design_mcp.pipelines.esm2_scorer import ESM2Scorer


async def score_stability(
    sequence: str,
    mutations: list[str] | None = None,
    reference_sequence: str | None = None,
) -> dict[str, Any]:
    """Score protein stability using ESM2 masked marginal likelihood.

    Args:
        sequence: Amino acid sequence to score.
        mutations: Optional list of mutations in ``"X42Y"`` format.
            When provided, computes per-mutation ΔΔ log-likelihood.
        reference_sequence: Optional wild-type sequence for mutation scoring.
            Inferred from ``mutations`` if not provided.

    Returns:
        Dict with ``sequence_score``, ``per_residue_scores``,
        and optionally ``mutation_effects``, ``reference_score``, ``delta_score``.
    """
    scorer = ESM2Scorer()

    if mutations:
        return await scorer.score_mutations(
            sequence=sequence,
            mutations=mutations,
            reference_sequence=reference_sequence,
        )
    else:
        return await scorer.score_sequence(sequence)
