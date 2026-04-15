"""
ESM2 pseudo-log-likelihood scorer for protein stability estimation.

Uses masked marginal scoring: mask each position, sum log-probabilities
of the true amino acid at each position. Higher scores indicate more
"natural" sequences according to the language model.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def _default_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


@dataclass
class ESM2ScorerConfig:
    """Configuration for ESM2 scoring."""

    model_name: str = os.environ.get("ESM2_MODEL", "esm2_t33_650M_UR50D")
    device: str | None = None

    def __post_init__(self):
        if self.device is None:
            self.device = _default_device()


class ESM2Scorer:
    """Pseudo-log-likelihood scoring using ESM2.

    Computes a per-residue masked marginal probability: for each position,
    the input token is replaced with ``<mask>`` and the model predicts a
    distribution over amino acids.  The log-probability of the true residue
    under that distribution is collected.  The *sequence score* is the mean
    of these per-residue log-probs (higher = more "natural").
    """

    def __init__(self, config: ESM2ScorerConfig | None = None):
        self.config = config or ESM2ScorerConfig()
        self._model = None
        self._alphabet = None
        self._batch_converter = None

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _load_model(self):
        if self._model is not None:
            return

        import torch
        import esm

        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        model = model.eval()

        if self.config.device == "cuda" and torch.cuda.is_available():
            model = model.cuda()

        self._model = model
        self._alphabet = alphabet
        self._batch_converter = alphabet.get_batch_converter()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def score_sequence(self, sequence: str) -> dict[str, Any]:
        """Compute wildtype marginal log-likelihood for a protein sequence.

        Single forward pass: run the unmasked sequence through ESM-2 and
        collect log P(true_aa | context) at each position.  This is ~L×
        faster than masked marginal scoring while correlating well with it
        for mutation scanning (Meier et al., 2021).

        Args:
            sequence: Amino acid sequence (uppercase, standard 20 AAs).

        Returns:
            Dict with ``sequence_score`` (mean log-prob), ``per_residue_scores``
            (list of floats), ``logits`` (L×20 numpy array for mutation
            scanning), and ``sequence_length``.
        """
        import torch

        sequence = sequence.upper().strip()
        if not sequence or not all(aa in VALID_AA for aa in sequence):
            raise ValueError("Sequence must contain only standard amino acids")

        self._load_model()

        data = [("protein", sequence)]
        _, _, batch_tokens = self._batch_converter(data)

        if self.config.device == "cuda" and torch.cuda.is_available():
            batch_tokens = batch_tokens.cuda()

        with torch.no_grad():
            # Single forward pass — no masking
            logits = self._model(batch_tokens)["logits"]  # (1, L+2, vocab)
            # Extract positions 1..L (skip BOS at 0, EOS at L+1)
            seq_logits = logits[0, 1 : len(sequence) + 1]  # (L, vocab)
            log_probs = torch.log_softmax(seq_logits, dim=-1)  # (L, vocab)

        # Per-residue score: log P(true_aa | context)
        per_residue: list[float] = []
        for i, aa in enumerate(sequence):
            token_idx = self._alphabet.get_idx(aa)
            per_residue.append(log_probs[i, token_idx].item())

        # Extract logits for the 20 standard AAs (for fast mutation scanning)
        aa_indices = [self._alphabet.get_idx(aa) for aa in "ACDEFGHIKLMNPQRSTVWY"]
        aa_log_probs = log_probs[:, aa_indices].cpu().numpy()  # (L, 20)

        mean_score = float(np.mean(per_residue))

        return {
            "sequence_score": mean_score,
            "per_residue_scores": per_residue,
            "aa_log_probs": aa_log_probs,  # (L, 20) for fast mutation scan
            "sequence_length": len(sequence),
        }

    async def score_mutations(
        self,
        sequence: str,
        mutations: list[str],
        reference_sequence: str | None = None,
    ) -> dict[str, Any]:
        """Score the effect of mutations via delta log-likelihood.

        Each mutation is specified as ``"X{pos}Y"`` where *X* is the wild-type
        residue, *pos* is the 1-indexed position, and *Y* is the mutant residue
        (e.g. ``"A42G"``).

        Args:
            sequence: The mutant sequence (with mutations already applied).
            mutations: List of mutation strings.
            reference_sequence: Optional wild-type sequence.  If not provided,
                the reference is reconstructed by reverting ``mutations`` on
                ``sequence``.

        Returns:
            Dict with ``sequence_score``, ``reference_score``, ``delta_score``,
            and ``mutation_effects`` (per-mutation breakdown).
        """
        sequence = sequence.upper().strip()
        mutations_parsed = self._parse_mutations(mutations)

        if reference_sequence is None:
            ref_list = list(sequence)
            for wt, pos, _mt in mutations_parsed:
                ref_list[pos - 1] = wt
            reference_sequence = "".join(ref_list)

        mutant_result = await self.score_sequence(sequence)
        ref_result = await self.score_sequence(reference_sequence)

        mutation_effects: list[dict[str, Any]] = []
        for mut_str, (wt, pos, mt) in zip(mutations, mutations_parsed):
            idx = pos - 1
            if idx < len(mutant_result["per_residue_scores"]) and idx < len(
                ref_result["per_residue_scores"]
            ):
                delta = (
                    mutant_result["per_residue_scores"][idx]
                    - ref_result["per_residue_scores"][idx]
                )
            else:
                delta = 0.0
            mutation_effects.append({
                "mutation": mut_str,
                "wt_residue": wt,
                "position": pos,
                "mt_residue": mt,
                "delta_log_likelihood": delta,
                "classification": "stabilizing" if delta > 0 else "destabilizing",
            })

        return {
            "sequence_score": mutant_result["sequence_score"],
            "reference_score": ref_result["sequence_score"],
            "delta_score": mutant_result["sequence_score"] - ref_result["sequence_score"],
            "mutation_effects": mutation_effects,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_mutations(mutations: list[str]) -> list[tuple[str, int, str]]:
        """Parse mutation strings like ``"A42G"`` into (wt, pos, mt) tuples."""
        import re

        parsed: list[tuple[str, int, str]] = []
        for m in mutations:
            match = re.match(r"^([A-Z])(\d+)([A-Z])$", m.upper().strip())
            if not match:
                raise ValueError(f"Invalid mutation format: {m!r} (expected e.g. 'A42G')")
            parsed.append((match.group(1), int(match.group(2)), match.group(3)))
        return parsed
