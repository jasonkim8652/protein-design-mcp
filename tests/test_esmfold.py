"""Tests for ESMFold pipeline runner - TDD RED phase first."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from protein_design_mcp.pipelines.esmfold import (
    ESMFoldConfig,
    ESMFoldRunner,
    PredictionResult,
)


class TestESMFoldConfig:
    """Tests for ESMFoldConfig dataclass."""

    def test_default_config(self):
        """Default config should have sensible values."""
        config = ESMFoldConfig()
        assert config.model_name == "esmfold_v1"
        assert config.device in ["cuda", "cpu"]

    def test_custom_config(self):
        """Custom config values should be set."""
        config = ESMFoldConfig(
            model_name="custom_model",
            chunk_size=128,
            device="cpu",
        )
        assert config.model_name == "custom_model"
        assert config.chunk_size == 128
        assert config.device == "cpu"


class TestPredictionResult:
    """Tests for PredictionResult dataclass."""

    def test_prediction_result_fields(self):
        """PredictionResult should have all required fields."""
        result = PredictionResult(
            sequence="MKVGA",
            pdb_string="ATOM...",
            plddt=85.0,
            ptm=0.78,
            plddt_per_residue=np.array([80, 85, 90, 85, 80]),
            pae_matrix=None,
        )
        assert result.sequence == "MKVGA"
        assert result.plddt == 85.0
        assert result.ptm == 0.78
        assert len(result.plddt_per_residue) == 5


class TestESMFoldRunner:
    """Tests for ESMFoldRunner class."""

    def test_runner_init_default(self):
        """Runner should initialize with default config."""
        runner = ESMFoldRunner()
        assert runner.config is not None
        assert runner._model is None  # Lazy loading

    def test_runner_init_custom_config(self):
        """Runner should accept custom config."""
        config = ESMFoldConfig(device="cpu")
        runner = ESMFoldRunner(config=config)
        assert runner.config.device == "cpu"

    @pytest.mark.asyncio
    async def test_predict_structure_returns_result(self):
        """predict_structure should return PredictionResult."""
        runner = ESMFoldRunner()

        # Mock the model
        with patch.object(runner, "_predict_with_model") as mock_predict:
            mock_predict.return_value = PredictionResult(
                sequence="MKVGA",
                pdb_string="ATOM      1  N   MET A   1...",
                plddt=85.0,
                ptm=0.78,
                plddt_per_residue=np.array([80, 85, 90, 85, 80]),
                pae_matrix=None,
            )

            result = await runner.predict_structure("MKVGA")

            assert isinstance(result, PredictionResult)
            assert result.sequence == "MKVGA"
            assert result.plddt > 0
            assert result.ptm > 0

    @pytest.mark.asyncio
    async def test_predict_structure_validates_sequence(self):
        """Invalid sequences should raise ValueError."""
        runner = ESMFoldRunner()

        # Invalid character in sequence
        with pytest.raises(ValueError, match="[Ii]nvalid"):
            await runner.predict_structure("MKVGA123")

    @pytest.mark.asyncio
    async def test_predict_structure_empty_sequence(self):
        """Empty sequence should raise ValueError."""
        runner = ESMFoldRunner()

        with pytest.raises(ValueError):
            await runner.predict_structure("")

    @pytest.mark.asyncio
    async def test_predict_structure_saves_pdb(self, tmp_path):
        """predict_structure should save PDB when output_pdb is specified."""
        runner = ESMFoldRunner()

        pdb_content = """ATOM      1  N   MET A   1       0.000   0.000   0.000  1.00 85.00           N
ATOM      2  CA  MET A   1       1.458   0.000   0.000  1.00 85.00           C
END
"""
        with patch.object(runner, "_predict_with_model") as mock_predict:
            mock_predict.return_value = PredictionResult(
                sequence="M",
                pdb_string=pdb_content,
                plddt=85.0,
                ptm=0.78,
                plddt_per_residue=np.array([85]),
                pae_matrix=None,
            )

            output_pdb = tmp_path / "prediction.pdb"
            result = await runner.predict_structure("M", output_pdb=str(output_pdb))

            assert output_pdb.exists()
            assert "ATOM" in output_pdb.read_text()

    @pytest.mark.asyncio
    async def test_predict_batch_returns_list(self):
        """predict_batch should return list of PredictionResults."""
        runner = ESMFoldRunner()

        with patch.object(runner, "_predict_with_model") as mock_predict:
            mock_predict.return_value = PredictionResult(
                sequence="MKVGA",
                pdb_string="ATOM...",
                plddt=85.0,
                ptm=0.78,
                plddt_per_residue=np.array([80, 85, 90, 85, 80]),
                pae_matrix=None,
            )

            sequences = ["MKVGA", "AAAA", "GGGG"]
            results = await runner.predict_batch(sequences, str(Path("/tmp")))

            assert len(results) == 3
            assert all(isinstance(r, PredictionResult) for r in results)


class TestRMSDCalculation:
    """Tests for RMSD calculation."""

    def test_rmsd_identical_structures(self, tmp_path):
        """RMSD of identical structures should be 0."""
        runner = ESMFoldRunner()

        # Create identical PDB files
        pdb_content = """ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 50.00           C
ATOM      2  CA  GLY A   2       3.800   0.000   0.000  1.00 50.00           C
ATOM      3  CA  SER A   3       7.600   0.000   0.000  1.00 50.00           C
END
"""
        pdb1 = tmp_path / "struct1.pdb"
        pdb2 = tmp_path / "struct2.pdb"
        pdb1.write_text(pdb_content)
        pdb2.write_text(pdb_content)

        rmsd = runner.calculate_rmsd(str(pdb1), str(pdb2))
        assert rmsd == pytest.approx(0.0, abs=0.01)

    def test_rmsd_different_structures(self, tmp_path):
        """RMSD of different structures should be > 0."""
        runner = ESMFoldRunner()

        pdb1_content = """ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 50.00           C
ATOM      2  CA  GLY A   2       3.800   0.000   0.000  1.00 50.00           C
END
"""
        pdb2_content = """ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 50.00           C
ATOM      2  CA  GLY A   2       3.800   5.000   0.000  1.00 50.00           C
END
"""
        pdb1 = tmp_path / "struct1.pdb"
        pdb2 = tmp_path / "struct2.pdb"
        pdb1.write_text(pdb1_content)
        pdb2.write_text(pdb2_content)

        rmsd = runner.calculate_rmsd(str(pdb1), str(pdb2))
        assert rmsd > 0


class TestSequenceValidation:
    """Tests for sequence validation."""

    def test_valid_sequences(self):
        """Valid amino acid sequences should pass."""
        runner = ESMFoldRunner()

        valid_sequences = [
            "MKVGA",
            "ACDEFGHIKLMNPQRSTVWY",
            "m",  # Single residue, lowercase
            "A" * 100,  # Long sequence
        ]

        for seq in valid_sequences:
            assert runner._validate_sequence(seq) is True

    def test_invalid_sequences(self):
        """Invalid sequences should be rejected."""
        runner = ESMFoldRunner()

        invalid_sequences = [
            "",  # Empty
            "MKVGA1",  # Numbers
            "MKV GA",  # Space
            "MKVGA!",  # Special chars
            "BJOUX",  # Invalid amino acids (B, J, O, U, X may be invalid)
        ]

        for seq in invalid_sequences:
            assert runner._validate_sequence(seq) is False


# Corner cases
class TestEdgeCases:
    """Test edge cases."""

    @pytest.mark.asyncio
    async def test_single_residue_prediction(self):
        """Single residue should still produce valid prediction."""
        runner = ESMFoldRunner()

        with patch.object(runner, "_predict_with_model") as mock_predict:
            mock_predict.return_value = PredictionResult(
                sequence="A",
                pdb_string="ATOM...",
                plddt=75.0,
                ptm=0.5,
                plddt_per_residue=np.array([75]),
                pae_matrix=None,
            )

            result = await runner.predict_structure("A")
            assert len(result.plddt_per_residue) == 1

    @pytest.mark.asyncio
    async def test_plddt_range(self):
        """pLDDT scores should be in valid range [0, 100]."""
        runner = ESMFoldRunner()

        with patch.object(runner, "_predict_with_model") as mock_predict:
            mock_predict.return_value = PredictionResult(
                sequence="MKVGA",
                pdb_string="ATOM...",
                plddt=85.0,
                ptm=0.78,
                plddt_per_residue=np.array([80, 85, 90, 85, 80]),
                pae_matrix=None,
            )

            result = await runner.predict_structure("MKVGA")
            assert 0 <= result.plddt <= 100
            assert all(0 <= x <= 100 for x in result.plddt_per_residue)

    @pytest.mark.asyncio
    async def test_ptm_range(self):
        """pTM scores should be in valid range [0, 1]."""
        runner = ESMFoldRunner()

        with patch.object(runner, "_predict_with_model") as mock_predict:
            mock_predict.return_value = PredictionResult(
                sequence="MKVGA",
                pdb_string="ATOM...",
                plddt=85.0,
                ptm=0.78,
                plddt_per_residue=np.array([80, 85, 90, 85, 80]),
                pae_matrix=None,
            )

            result = await runner.predict_structure("MKVGA")
            assert 0 <= result.ptm <= 1
