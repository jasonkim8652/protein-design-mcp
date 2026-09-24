"""
Verify that the edits to esmfold and esm2_scorer converge with the original esmfold and esm2_scorer implementations.
"""


#original implementation
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
import asyncio
import numpy as np
import pytest
from pathlib import Path
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
    """Result from structure prediction (ESMFold or AF2)."""

    sequence: str
    pdb_string: str
    plddt: float
    ptm: float
    plddt_per_residue: np.ndarray
    pae_matrix: np.ndarray | None = None
    iptm: float | None = None  # Interface pTM (multimer only)


class FairESMFoldRunner:
    """Wrapper for running ESMFold predictions."""

    def __init__(self, config: ESMFoldConfig | None = None):
        """Initialize ESMFold runner."""
        self.config = config or ESMFoldConfig()
        self._model = None

    def _load_model(self):
        """Load ESMFold model with IPA-key compatibility shim for esmfold_3B_v1."""
        if self._model is None:
            try:
                import sys
                import types
                import torch

                if "attn_core_inplace_cuda" not in sys.modules:
                    sys.modules["attn_core_inplace_cuda"] = types.ModuleType(
                        "attn_core_inplace_cuda"
                    )

                import esm
                import esm.esmfold.v1.pretrained as _esm_pretrained

                _orig_load = _esm_pretrained._load_model

                def _patched_load(model_name: str):
                    from esm.esmfold.v1.esmfold import ESMFold

                    url = f"https://dl.fbaipublicfiles.com/fair-esm/models/{model_name}.pt"
                    model_data = torch.hub.load_state_dict_from_url(
                        url,
                        map_location="cpu",
                        weights_only=False,  # cfg is omegaconf.DictConfig, not a plain tensor
                    )

                    cfg = model_data["cfg"]["model"]
                    old_state = model_data["model"]

                    # esmfold_3B_v1 stores IPA point projections as bare nn.Linear
                    # (no .linear. infix); current fair-esm wraps them in a submodule.
                    # Confirmed by key inspection: the 4 affected keys are:
                    #   linear_kv_points.{weight,bias}  →  linear_kv_points.linear.{weight,bias}
                    #   linear_q_points.{weight,bias}   →  linear_q_points.linear.{weight,bias}
                    # linear_kv.{weight,bias} is a separate module and maps fine as-is.
                    remap = {
                        "trunk.structure_module.ipa.linear_kv_points.weight":
                            "trunk.structure_module.ipa.linear_kv_points.linear.weight",
                        "trunk.structure_module.ipa.linear_kv_points.bias":
                            "trunk.structure_module.ipa.linear_kv_points.linear.bias",
                        "trunk.structure_module.ipa.linear_q_points.weight":
                            "trunk.structure_module.ipa.linear_q_points.linear.weight",
                        "trunk.structure_module.ipa.linear_q_points.bias":
                            "trunk.structure_module.ipa.linear_q_points.linear.bias",
                    }
                    new_state = {remap.get(k, k): v for k, v in old_state.items()}

                    model = ESMFold(esmfold_config=cfg)
                    missing, unexpected = model.load_state_dict(new_state, strict=False)

                    # After the remap there should be nothing essential missing.
                    # Surface any residual gaps rather than silently corrupting inference.
                    if missing:
                        import warnings
                        warnings.warn(
                            f"[FairESMFold] Still-missing keys after IPA remap "
                            f"(first 5): {missing[:5]}"
                        )
                    return model

                _esm_pretrained._load_model = _patched_load
                try:
                    self._model = esm.pretrained.esmfold_v1()
                finally:
                    _esm_pretrained._load_model = _orig_load  # restore unconditionally

                self._model = self._model.eval()

                if self.config.device == "cuda" and torch.cuda.is_available():
                    self._model = self._model.cuda()

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
        Run actual ESMFold prediction using model.infer() for accurate pTM.

        Args:
            sequence: Amino acid sequence

        Returns:
            PredictionResult with structure and metrics
        """
        import torch

        model = self._load_model()

        with torch.no_grad():
            # Use model.infer() to get full output including pTM tensor,
            # then convert to PDB string via output_to_pdb().
            raw_output = model.infer([sequence])
            pdb_string = model.output_to_pdb(raw_output)[0]

            # Extract real pTM from model output
            ptm = float(raw_output["ptm"].item())

            # Extract pAE matrix if available
            pae_matrix = None
            if "predicted_aligned_error" in raw_output:
                pae = raw_output["predicted_aligned_error"]
                if pae is not None and pae.numel() > 0:
                    pae_matrix = pae[0].cpu().numpy()

        # Extract pLDDT from B-factor column of PDB
        plddt_per_residue = self._extract_plddt_from_pdb(pdb_string)
        mean_plddt = float(np.mean(plddt_per_residue))

        return PredictionResult(
            sequence=sequence,
            pdb_string=pdb_string,
            plddt=mean_plddt,
            ptm=ptm,
            plddt_per_residue=plddt_per_residue,
            pae_matrix=pae_matrix,
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


class HFESMFoldRunner:
    """Wrapper for running ESMFold predictions."""

    def __init__(self, config: ESMFoldConfig | None = None):
        """Initialize ESMFold runner."""
        self.config = config or ESMFoldConfig()
        self._model = None

    def _load_model(self):
        """Load ESMFold model (lazy loading) via Hugging Face Transformers."""
        if self._model is None:
            try:
                import torch
                from transformers import AutoTokenizer, EsmForProteinFolding

                # Hugging Face uses the unified identifier for ESMFold v1
                model_name = "facebook/esmfold_v1"

                # 1. Load the tokenizer (replaces the old alphabet/batch_converter)
                self._tokenizer = AutoTokenizer.from_pretrained(model_name)

                # 2. Load the folding model architecture
                self._model = EsmForProteinFolding.from_pretrained(model_name)
                self._model.eval()

                # 3. Handle device placement natively
                if self.config.device == "cuda" and torch.cuda.is_available():
                    self._model = self._model.to("cuda")
                    # Optional: use half-precision on GPU for faster inference/lower VRAM footprint
                    # self._model = self._model.half()

                # 4. Handle chunk size scaling for long sequences if configured
                if self.config.chunk_size:
                    # Hugging Face implementation uses 'esm.trunk.set_chunk_size' internally
                    self._model.esm.trunk.set_chunk_size(self.config.chunk_size)

            except ImportError as e:
                raise ESMFoldError(
                    "ESMFold requires 'transformers' and 'accelerate' packages. "
                    "Install with: uv add transformers accelerate"
                ) from e
            except Exception as e:
                raise ESMFoldError(f"Failed to load Hugging Face ESMFold model: {e}") from e

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

    async def _predict_with_model(self,sequence: str) -> PredictionResult:
        """
        Run actual ESMFold prediction using Hugging Face transformers.

        Args:
            sequence: Amino acid sequence

        Returns:
            PredictionResult with structure and metrics
        """
        import torch
        import numpy as np

        # 1. Ensure the model and its matching tokenizer are initialized
        model = self._load_model()
        tokenizer = self._tokenizer

        # 2. Tokenize the sequence (add_special_tokens=False mimics fair-esm behavior)
        inputs = tokenizer([sequence], return_tensors="pt", add_special_tokens=False)
        
        # Move tokenized tensors to the target hardware device
        if self.config.device == "cuda" and torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            # 3. Execute the forward pass
            outputs = model(**inputs)

            # 4. Generate the PDB coordinates payload string using the native HF parser
            pdb_strings = model.output_to_pdb(outputs)
            pdb_string = pdb_strings[0]

            # 5. Extract structural confidence metrics natively
            ptm = None
            if hasattr(outputs, "ptm") and outputs.ptm is not None:
                ptm = float(outputs.ptm.item())

            # 6. Extract the Predicted Aligned Error (pAE) matrix
            pae_matrix = None
            if hasattr(outputs, "predicted_aligned_error") and outputs.predicted_aligned_error is not None:
                pae = outputs.predicted_aligned_error
                if pae.numel() > 0:
                    # Shape is generally (batch_size, num_residues, num_residues)
                    pae_matrix = pae[0].cpu().numpy()

        # 7. Extract pLDDT from the B-factor column of the generated PDB payload
        plddt_per_residue = self._extract_plddt_from_pdb(pdb_string)
        mean_plddt = float(np.mean(plddt_per_residue))

        return PredictionResult(
            sequence=sequence,
            pdb_string=pdb_string,
            plddt=mean_plddt,
            ptm=ptm,
            plddt_per_residue=plddt_per_residue,
            pae_matrix=pae_matrix,
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

#since we only changed the _load_model and _predict_with_model methods we will test these

async def verify_runners_convergence():
    """
    Verifies that HFESMFoldRunner and FairESMFoldRunner converge
    to the same structural outputs and confidence metrics.
    """
    test_sequence = "ACDEFGHIKLMNPQRSTVWY"
    
    # 1. Initialize and run FairESMFoldRunner
    fair_runner = FairESMFoldRunner()
    fair_result = await fair_runner.predict_structure(test_sequence)
    
    # 2. Initialize and run HFESMFoldRunner
    hf_runner = HFESMFoldRunner()
    hf_result = await hf_runner.predict_structure(test_sequence)
    
    # 3. Assert exact match on input handling
    assert hf_result.sequence == fair_result.sequence, "Sequences mismatch."
    
    # 4. Verify metric convergence (pLDDT & pTM)
    # Allows a tiny tolerance due to potential difference in attention implementation fallbacks
    assert np.allclose(hf_result.plddt, fair_result.plddt, atol=1e-2), (
        f"Mean pLDDT diverged: HF={hf_result.plddt}, Fair={fair_result.plddt}"
    )
    
    if hf_result.ptm is not None and fair_result.ptm is not None:
        assert np.isclose(hf_result.ptm, fair_result.ptm, atol=1e-2), (
            f"pTM diverged: HF={hf_result.ptm}, Fair={fair_result.ptm}"
        )

    # 5. Verify Structural Coordination Convergence using RMSD
    # Write temporary PDBs to utilize the built-in coordinate parsing and Kabsch alignment
    tmp_fair_pdb = Path("tmp_fair_converge.pdb")
    tmp_hf_pdb = Path("tmp_hf_converge.pdb")
    
    try:
        tmp_fair_pdb.write_text(fair_result.pdb_string)
        tmp_hf_pdb.write_text(hf_result.pdb_string)
        
        # Calculate RMSD between both predictions using the runner's native alignment tool
        rmsd = fair_runner.calculate_rmsd(
            predicted_pdb=str(tmp_hf_pdb),
            reference_pdb=str(tmp_fair_pdb),
            align=True
        )
        
        print(f"Calculated RMSD between Fair and HF implementations: {rmsd:.4f} Å")
        
        # An RMSD < 0.1 Angstrom means the backbone coordinates are identical for all intents
        assert rmsd < 1e-1, f"Structures drifted significantly! RMSD: {rmsd:.4f} Å"
        
    finally:
        # Cleanup temporary testing payloads safely
        if tmp_fair_pdb.exists():
            tmp_fair_pdb.unlink()
        if tmp_hf_pdb.exists():
            tmp_hf_pdb.unlink()

def test_esmfold_runners_convergence():
    """Synchronous wrapper for test runners."""
    asyncio.run(verify_runners_convergence())

if __name__ == "__main__":
    try:
        test_esmfold_runners_convergence()
        print("Convergence validation successful: Both runners output equivalent geometries.")
    except AssertionError as e:
        print(f"Convergence validation failed:\n{e}")