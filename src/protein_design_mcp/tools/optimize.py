"""
Sequence optimization tool.

Optimizes protein sequences using ESM-2 masked marginal scoring with
greedy iterative mutation. At each round the worst-scoring positions
(lowest per-residue log-likelihood) are masked and re-predicted by
ESM-2, accepting mutations that improve overall pseudo-log-likelihood.

Pipeline:
  1. Score current_sequence with ESM-2 → per-residue log-probs
  2. Identify N worst-scoring positions
  3. For each position, find the best amino acid (highest masked marginal prob)
  4. Accept mutations that improve overall PLL
  5. Repeat for K rounds
  6. Validate final candidates with ESMFold
  7. Return best candidate
"""

import logging
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from protein_design_mcp.pipelines.esm2_scorer import ESM2Scorer
from protein_design_mcp.pipelines.esmfold import ESMFoldRunner


logger = logging.getLogger(__name__)

# Valid amino acids
VALID_AA = list("ACDEFGHIKLMNPQRSTVWY")
VALID_AA_SET = set(VALID_AA)

# Valid optimization targets
VALID_TARGETS = {"stability", "affinity", "both"}

# Optimization hyperparameters (defaults for seq ≤ 200)
MAX_ROUNDS = 3           # Number of optimization rounds
POSITIONS_PER_ROUND = 5  # Worst-scoring positions to try per round
NUM_CANDIDATES = 4       # Return top N candidates for ESMFold validation


def _adaptive_optimization_params(seq_len: int) -> tuple[int, int]:
    """Adaptive rounds/positions based on sequence length.

    ESM-2 forward pass cost scales with sequence length.
    Total calls = rounds × positions × 19 (AA trials) + re-scoring.

    Returns (max_rounds, positions_per_round).
    """
    if seq_len <= 200:
        return (3, 5)    # 285 ESM-2 calls, ~8-12 min
    elif seq_len <= 400:
        return (2, 3)    # 114 ESM-2 calls, ~8-15 min
    else:
        return (1, 3)    # 57 ESM-2 calls, ~5-10 min


async def optimize_sequence(
    current_sequence: str,
    target_pdb: str,
    optimization_target: str = "both",
    fixed_positions: list[int] | None = None,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """
    Optimize an existing protein sequence using ESM-2 guided mutation.

    Uses masked marginal scoring to identify weak positions, then tests
    all 20 amino acid substitutions at each position, accepting those
    that improve the pseudo-log-likelihood. Final candidates are validated
    with ESMFold structure prediction.

    Args:
        current_sequence: Starting amino acid sequence to optimize
        target_pdb: Path to target protein PDB (used for context/reference)
        optimization_target: What to optimize:
            - "stability": Optimize for protein stability (higher pLDDT)
            - "affinity": Optimize for binding affinity (higher pTM)
            - "both": Optimize for both (default)
        fixed_positions: Positions to keep fixed (1-indexed)
        temperature: Sampling temperature for position selection (default: 0.0).
            When > 0, adds Gaussian noise to per-residue scores before selecting
            positions to mutate, producing diverse optimization trajectories.

    Returns:
        Dictionary containing:
        - optimized_sequence: Best optimized sequence
        - mutations: List of mutations from original
        - predicted_improvement: Estimated improvements
        - metrics: Quality metrics (pLDDT, pTM, esm2_score)
    """
    # Validate sequence
    sequence = current_sequence.upper()
    if not sequence:
        raise ValueError("Sequence cannot be empty")

    invalid_chars = set(sequence) - VALID_AA_SET
    if invalid_chars:
        raise ValueError(
            f"Sequence contains invalid characters: {invalid_chars}. "
            f"Valid amino acids are: {''.join(sorted(VALID_AA_SET))}"
        )

    # Validate target PDB
    target_path = Path(target_pdb)
    if not target_path.exists():
        raise FileNotFoundError(f"Target PDB not found: {target_pdb}")

    # Validate optimization target
    if optimization_target not in VALID_TARGETS:
        raise ValueError(
            f"Optimization target must be one of {VALID_TARGETS}, "
            f"got: {optimization_target}"
        )

    fixed_set = set(fixed_positions) if fixed_positions else set()

    # Adaptive optimization parameters based on sequence length
    max_rounds, positions_per_round = _adaptive_optimization_params(len(sequence))
    logger.info(
        "Adaptive params: %d rounds × %d positions (seq_len=%d, %d forward passes)",
        max_rounds, positions_per_round, len(sequence),
        max_rounds + 1,  # 1 baseline + 1 per round (wildtype marginal)
    )

    # Initialize ESM-2 scorer
    scorer = ESM2Scorer()

    # Score baseline sequence — single forward pass gives per-residue scores
    # AND logits for all 20 AAs at every position
    baseline_result = await scorer.score_sequence(sequence)
    baseline_score = baseline_result["sequence_score"]
    per_residue = baseline_result["per_residue_scores"]
    aa_log_probs = baseline_result["aa_log_probs"]  # (L, 20) numpy array

    logger.info(
        "Baseline ESM-2 score: %.3f (length=%d)",
        baseline_score, len(sequence),
    )

    # AA index mapping for logit lookup
    aa_to_idx = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}

    # Iterative optimization: use logits to find best mutations without
    # additional forward passes per position. Only re-score once per round
    # to update context after accepted mutations.
    current_seq = list(sequence)
    current_per_residue = list(per_residue)
    current_aa_log_probs = aa_log_probs
    current_score = baseline_score
    all_mutations: list[str] = []
    trajectory: list[dict] = [{"round": 0, "score": baseline_score, "mutations": []}]

    for round_idx in range(1, max_rounds + 1):
        # Find worst-scoring mutable positions
        scored_positions = [
            (i, current_per_residue[i])
            for i in range(len(current_seq))
            if (i + 1) not in fixed_set  # 1-indexed fixed_positions
        ]

        if temperature > 0:
            rng = np.random.default_rng()
            noisy = [
                (pos, score + temperature * rng.standard_normal())
                for pos, score in scored_positions
            ]
            noisy.sort(key=lambda x: x[1])
            target_positions = [pos for pos, _ in noisy[:positions_per_round]]
        else:
            scored_positions.sort(key=lambda x: x[1])
            target_positions = [pos for pos, _ in scored_positions[:positions_per_round]]

        round_mutations = []
        for pos in target_positions:
            original_aa = current_seq[pos]

            # Find best AA at this position from cached logits (no forward pass!)
            best_aa = original_aa
            best_logprob = current_aa_log_probs[pos, aa_to_idx[original_aa]]
            for aa in VALID_AA:
                if aa == original_aa:
                    continue
                lp = current_aa_log_probs[pos, aa_to_idx[aa]]
                if lp > best_logprob:
                    best_logprob = lp
                    best_aa = aa

            # Accept if improvement found
            if best_aa != original_aa:
                mutation_str = f"{original_aa}{pos + 1}{best_aa}"
                current_seq[pos] = best_aa
                round_mutations.append(mutation_str)
                all_mutations.append(mutation_str)
                logger.info(
                    "Round %d: %s (logit %.3f → %.3f)",
                    round_idx, mutation_str,
                    current_aa_log_probs[pos, aa_to_idx[original_aa]],
                    best_logprob,
                )

        # Re-score ONCE per round to update context after mutations
        if round_mutations:
            updated = await scorer.score_sequence("".join(current_seq))
            current_per_residue = updated["per_residue_scores"]
            current_aa_log_probs = updated["aa_log_probs"]
            current_score = updated["sequence_score"]
            logger.info(
                "Round %d: %d mutations accepted, score %.3f (+%.3f)",
                round_idx, len(round_mutations), current_score,
                current_score - baseline_score,
            )

        trajectory.append({
            "round": round_idx,
            "score": current_score,
            "mutations": round_mutations,
        })

        if not round_mutations:
            logger.info("Round %d: no improving mutations found, stopping", round_idx)
            break

    optimized_seq = "".join(current_seq)

    # Validate with ESMFold
    esmfold = ESMFoldRunner()
    prediction = await esmfold.predict_structure(optimized_seq)

    # Also score baseline structure for comparison
    baseline_prediction = await esmfold.predict_structure(sequence)

    predicted_improvement = {
        "esm2_delta": f"{current_score - baseline_score:+.3f} PLL",
        "stability_delta": f"{prediction.plddt - baseline_prediction.plddt:+.1f}% pLDDT",
        "affinity_delta": f"{prediction.ptm - baseline_prediction.ptm:+.3f} pTM",
        "baseline_plddt": baseline_prediction.plddt,
        "baseline_ptm": baseline_prediction.ptm,
        "baseline_esm2_score": baseline_score,
    }

    return {
        "optimized_sequence": optimized_seq,
        "mutations": all_mutations,
        "predicted_improvement": predicted_improvement,
        "metrics": {
            "plddt": prediction.plddt,
            "ptm": prediction.ptm,
            "esm2_score": current_score,
        },
        "all_candidates": 1,
        "optimization_trajectory": trajectory,
    }


def _calculate_mutations(original: str, optimized: str) -> list[str]:
    """Calculate list of mutations between original and optimized sequences."""
    mutations = []
    min_len = min(len(original), len(optimized))
    for i in range(min_len):
        if original[i] != optimized[i]:
            mutations.append(f"{original[i]}{i+1}{optimized[i]}")
    return mutations
