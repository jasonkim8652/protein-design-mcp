"""
ESMFold wrapper for structure prediction.

ESMFold is a fast protein structure prediction model based on
the ESM-2 language model.
"""

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from protein_design_mcp.exceptions import ESMFoldError


# Valid amino acid characters
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def _cuda_available() -> bool:
    """Check if CUDA is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _default_device() -> str:
    """Get default device based on CUDA availability."""
    return "cuda" if _cuda_available() else "cpu"


@dataclass
class ESMFoldConfig:
    """Configuration for ESMFold predictions."""

    model_name: str = os.environ.get("ESMFOLD_MODEL", "esmfold_v1")
    chunk_size: int | None = None  # For long sequences
    device: str | None = None

    def __post_init__(self):
        """Set default device if not specified."""
        if self.device is None:
            self.device = _default_device()


@dataclass
class PredictionResult:
    """Result from ESMFold prediction."""

    sequence: str
    pdb_string: str
    plddt: float
    ptm: float
    plddt_per_residue: np.ndarray
    pae_matrix: np.ndarray | None = None


class ESMFoldRunner:
    """Wrapper for running ESMFold predictions."""

    def __init__(self, config: ESMFoldConfig | None = None):
        """Initialize ESMFold runner."""
        self.config = config or ESMFoldConfig()
        self._model = None

    def _load_model(self):
        """Load ESMFold model (lazy loading)."""
        if self._model is None:
            try:
                import sys
                import types
                import torch

                # Mock the missing openfold CUDA extension if not compiled.
                # ESMFold falls back to pure PyTorch attention when this is absent.
                if "attn_core_inplace_cuda" not in sys.modules:
                    sys.modules["attn_core_inplace_cuda"] = types.ModuleType(
                        "attn_core_inplace_cuda"
                    )

                import esm

                self._model = esm.pretrained.esmfold_v1()
                self._model = self._model.eval()

                if self.config.device == "cuda" and torch.cuda.is_available():
                    self._model = self._model.cuda()

                # Set chunk size for long sequences
                if self.config.chunk_size:
                    self._model.set_chunk_size(self.config.chunk_size)
            except ImportError as e:
                raise ESMFoldError(
                    "ESMFold requires 'esm' package. Install with: pip install fair-esm"
                ) from e
            except Exception as e:
                raise ESMFoldError(f"Failed to load ESMFold model: {e}") from e

        return self._model

    def _validate_sequence(self, sequence: str) -> bool:
        """
        Validate amino acid sequence.

        Args:
            sequence: Amino acid sequence

        Returns:
            True if valid, False otherwise
        """
        if not sequence:
            return False

        # Convert to uppercase for validation
        seq_upper = sequence.upper()

        # Check all characters are valid amino acids
        return all(aa in VALID_AA for aa in seq_upper)

    async def _predict_with_model(
        self,
        sequence: str,
    ) -> PredictionResult:
        """
        Run actual ESMFold prediction.

        Args:
            sequence: Amino acid sequence

        Returns:
            PredictionResult with structure and metrics
        """
        import torch

        model = self._load_model()

        with torch.no_grad():
            output = model.infer_pdb(sequence)

        # Extract pLDDT from B-factor column of PDB
        plddt_per_residue = self._extract_plddt_from_pdb(output)
        mean_plddt = float(np.mean(plddt_per_residue))

        # Get pTM score if available
        # ESMFold's infer_pdb doesn't directly return pTM, so we estimate from pLDDT
        # For accurate pTM, use model.infer() instead
        ptm = self._estimate_ptm(plddt_per_residue)

        return PredictionResult(
            sequence=sequence,
            pdb_string=output,
            plddt=mean_plddt,
            ptm=ptm,
            plddt_per_residue=plddt_per_residue,
            pae_matrix=None,
        )

    def _extract_plddt_from_pdb(self, pdb_string: str) -> np.ndarray:
        """Extract pLDDT scores from B-factor column of PDB."""
        plddt_values = []
        seen_residues = set()

        for line in pdb_string.split("\n"):
            if line.startswith("ATOM"):
                # PDB format: columns 61-66 are B-factor
                try:
                    res_num = int(line[22:26].strip())
                    if res_num not in seen_residues:
                        bfactor = float(line[60:66].strip())
                        plddt_values.append(bfactor)
                        seen_residues.add(res_num)
                except (ValueError, IndexError):
                    continue

        return np.array(plddt_values) if plddt_values else np.array([50.0])

    def _estimate_ptm(self, plddt_per_residue: np.ndarray) -> float:
        """
        Estimate pTM from pLDDT values.

        This is an approximation; for accurate pTM, use model.infer().
        """
        # Simple correlation-based estimate
        mean_plddt = np.mean(plddt_per_residue)
        # pTM roughly correlates with mean pLDDT / 100
        return min(0.99, max(0.01, mean_plddt / 100.0))

    async def predict_structure(
        self,
        sequence: str,
        output_pdb: str | None = None,
    ) -> PredictionResult:
        """
        Predict structure for a protein sequence.

        Args:
            sequence: Amino acid sequence
            output_pdb: Optional path to save PDB file

        Returns:
            PredictionResult with structure and metrics

        Raises:
            ValueError: If sequence is invalid
            ESMFoldError: If prediction fails
        """
        # Clean sequence: remove chain separators, gaps, whitespace
        sequence = sequence.upper().replace("/", "").replace("-", "").replace(" ", "")
        # Remove any non-AA characters
        sequence = "".join(c for c in sequence if c in VALID_AA)

        if not sequence:
            raise ValueError(
                f"Invalid sequence. Must contain valid amino acids: {VALID_AA}"
            )

        # Run prediction
        result = await self._predict_with_model(sequence.upper())

        # Save PDB if requested
        if output_pdb:
            output_path = Path(output_pdb)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(result.pdb_string)

        return result

    async def predict_batch(
        self,
        sequences: list[str],
        output_dir: str,
    ) -> list[PredictionResult]:
        """
        Predict structures for multiple sequences.

        Args:
            sequences: List of amino acid sequences
            output_dir: Directory to save PDB files

        Returns:
            List of PredictionResults
        """
        results = []
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for i, seq in enumerate(sequences):
            pdb_path = output_path / f"prediction_{i:04d}.pdb"
            result = await self.predict_structure(seq, output_pdb=str(pdb_path))
            results.append(result)

        return results

    def calculate_rmsd(
        self,
        predicted_pdb: str,
        reference_pdb: str,
        align: bool = True,
    ) -> float:
        """
        Calculate RMSD between predicted and reference structures.

        Args:
            predicted_pdb: Path to predicted structure
            reference_pdb: Path to reference structure
            align: Whether to align structures before RMSD calculation

        Returns:
            RMSD value in Angstroms
        """
        # Extract CA coordinates from both structures
        pred_coords = self._get_ca_coordinates(predicted_pdb)
        ref_coords = self._get_ca_coordinates(reference_pdb)

        if len(pred_coords) != len(ref_coords):
            raise ValueError(
                f"Structure length mismatch: {len(pred_coords)} vs {len(ref_coords)}"
            )

        pred_coords = np.array(pred_coords)
        ref_coords = np.array(ref_coords)

        if align:
            # Kabsch alignment
            pred_coords = self._kabsch_align(pred_coords, ref_coords)

        # Calculate RMSD
        diff = pred_coords - ref_coords
        rmsd = np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))

        return float(rmsd)

    def _get_ca_coordinates(self, pdb_path: str) -> list[list[float]]:
        """Extract CA atom coordinates from PDB file."""
        coords = []
        with open(pdb_path, "r") as f:
            for line in f:
                if line.startswith("ATOM") and " CA " in line:
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    coords.append([x, y, z])
        return coords

    def _kabsch_align(
        self,
        mobile: np.ndarray,
        target: np.ndarray,
    ) -> np.ndarray:
        """
        Align mobile coordinates to target using Kabsch algorithm.

        Args:
            mobile: Coordinates to align (N x 3)
            target: Reference coordinates (N x 3)

        Returns:
            Aligned mobile coordinates
        """
        # Center both coordinate sets
        mobile_center = np.mean(mobile, axis=0)
        target_center = np.mean(target, axis=0)

        mobile_centered = mobile - mobile_center
        target_centered = target - target_center

        # Compute covariance matrix
        H = mobile_centered.T @ target_centered

        # SVD
        U, S, Vt = np.linalg.svd(H)

        # Compute rotation matrix
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        R = Vt.T @ np.diag([1, 1, d]) @ U.T

        # Apply rotation and translation
        aligned = (mobile_centered @ R.T) + target_center

        return aligned
