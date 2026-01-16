"""Tests for validate_design tool - TDD RED phase first."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from protein_design_mcp.tools.validate import validate_design
from protein_design_mcp.pipelines.esmfold import PredictionResult


# Test fixtures path
FIXTURES_DIR = Path(__file__).parent / "fixtures" / "test_pdbs"


class TestValidateDesign:
    """Tests for validate_design function."""

    @pytest.mark.asyncio
    async def test_validate_design_returns_dict(self):
        """validate_design should return a dictionary with results."""
        with patch("protein_design_mcp.tools.validate.ESMFoldRunner") as mock_runner:
            mock_instance = MagicMock()
            mock_instance.predict_structure = AsyncMock(
                return_value=PredictionResult(
                    sequence="MKVGA",
                    pdb_string="ATOM...",
                    plddt=85.0,
                    ptm=0.78,
                    plddt_per_residue=np.array([80, 85, 90, 85, 80]),
                    pae_matrix=None,
                )
            )
            mock_runner.return_value = mock_instance

            result = await validate_design("MKVGA")

            assert isinstance(result, dict)
            assert "plddt" in result
            assert "ptm" in result

    @pytest.mark.asyncio
    async def test_validate_design_structure_fields(self):
        """Result should have expected structure fields."""
        with patch("protein_design_mcp.tools.validate.ESMFoldRunner") as mock_runner:
            mock_instance = MagicMock()
            mock_instance.predict_structure = AsyncMock(
                return_value=PredictionResult(
                    sequence="MKVGA",
                    pdb_string="ATOM      1  N   MET A   1...\nEND",
                    plddt=85.0,
                    ptm=0.78,
                    plddt_per_residue=np.array([80, 85, 90, 85, 80]),
                    pae_matrix=None,
                )
            )
            mock_runner.return_value = mock_instance

            result = await validate_design("MKVGA")

            # Required fields from PRD
            assert "predicted_structure_pdb" in result
            assert "plddt" in result
            assert "ptm" in result
            assert "plddt_per_residue" in result

    @pytest.mark.asyncio
    async def test_validate_design_plddt_value(self):
        """pLDDT should be in valid range."""
        with patch("protein_design_mcp.tools.validate.ESMFoldRunner") as mock_runner:
            mock_instance = MagicMock()
            mock_instance.predict_structure = AsyncMock(
                return_value=PredictionResult(
                    sequence="MKVGA",
                    pdb_string="ATOM...",
                    plddt=85.0,
                    ptm=0.78,
                    plddt_per_residue=np.array([80, 85, 90, 85, 80]),
                    pae_matrix=None,
                )
            )
            mock_runner.return_value = mock_instance

            result = await validate_design("MKVGA")

            assert 0 <= result["plddt"] <= 100

    @pytest.mark.asyncio
    async def test_validate_design_ptm_value(self):
        """pTM should be in valid range."""
        with patch("protein_design_mcp.tools.validate.ESMFoldRunner") as mock_runner:
            mock_instance = MagicMock()
            mock_instance.predict_structure = AsyncMock(
                return_value=PredictionResult(
                    sequence="MKVGA",
                    pdb_string="ATOM...",
                    plddt=85.0,
                    ptm=0.78,
                    plddt_per_residue=np.array([80, 85, 90, 85, 80]),
                    pae_matrix=None,
                )
            )
            mock_runner.return_value = mock_instance

            result = await validate_design("MKVGA")

            assert 0 <= result["ptm"] <= 1

    @pytest.mark.asyncio
    async def test_validate_design_invalid_sequence(self):
        """Invalid sequence should raise ValueError."""
        with pytest.raises(ValueError, match="[Ii]nvalid"):
            await validate_design("MKVGA123")

    @pytest.mark.asyncio
    async def test_validate_design_empty_sequence(self):
        """Empty sequence should raise ValueError."""
        with pytest.raises(ValueError):
            await validate_design("")

    @pytest.mark.asyncio
    async def test_validate_design_with_expected_structure(self, tmp_path):
        """Should calculate RMSD when expected_structure provided."""
        # Create expected structure PDB
        expected_pdb = tmp_path / "expected.pdb"
        expected_pdb.write_text("""ATOM      1  CA  MET A   1       0.000   0.000   0.000  1.00 50.00           C
ATOM      2  CA  LYS A   2       3.800   0.000   0.000  1.00 50.00           C
ATOM      3  CA  VAL A   3       7.600   0.000   0.000  1.00 50.00           C
END
""")

        with patch("protein_design_mcp.tools.validate.ESMFoldRunner") as mock_runner:
            mock_instance = MagicMock()
            mock_instance.predict_structure = AsyncMock(
                return_value=PredictionResult(
                    sequence="MKV",
                    pdb_string="""ATOM      1  CA  MET A   1       0.000   0.000   0.000  1.00 85.00           C
ATOM      2  CA  LYS A   2       3.800   0.000   0.000  1.00 85.00           C
ATOM      3  CA  VAL A   3       7.600   0.000   0.000  1.00 85.00           C
END
""",
                    plddt=85.0,
                    ptm=0.78,
                    plddt_per_residue=np.array([85, 85, 85]),
                    pae_matrix=None,
                )
            )
            mock_instance.calculate_rmsd = MagicMock(return_value=0.0)
            mock_runner.return_value = mock_instance

            result = await validate_design("MKV", expected_structure=str(expected_pdb))

            assert "rmsd_to_expected" in result
            assert result["rmsd_to_expected"] >= 0

    @pytest.mark.asyncio
    async def test_validate_design_secondary_structure(self):
        """Should include secondary structure prediction."""
        with patch("protein_design_mcp.tools.validate.ESMFoldRunner") as mock_runner:
            mock_instance = MagicMock()
            mock_instance.predict_structure = AsyncMock(
                return_value=PredictionResult(
                    sequence="MKVGA",
                    pdb_string="ATOM...",
                    plddt=85.0,
                    ptm=0.78,
                    plddt_per_residue=np.array([80, 85, 90, 85, 80]),
                    pae_matrix=None,
                )
            )
            mock_runner.return_value = mock_instance

            result = await validate_design("MKVGA")

            # Secondary structure is optional based on PRD
            # but if present, should be a string
            if "secondary_structure" in result:
                assert isinstance(result["secondary_structure"], str)


class TestValidateDesignEdgeCases:
    """Edge case tests for validate_design."""

    @pytest.mark.asyncio
    async def test_single_residue(self):
        """Single residue sequence should work."""
        with patch("protein_design_mcp.tools.validate.ESMFoldRunner") as mock_runner:
            mock_instance = MagicMock()
            mock_instance.predict_structure = AsyncMock(
                return_value=PredictionResult(
                    sequence="A",
                    pdb_string="ATOM...",
                    plddt=75.0,
                    ptm=0.5,
                    plddt_per_residue=np.array([75]),
                    pae_matrix=None,
                )
            )
            mock_runner.return_value = mock_instance

            result = await validate_design("A")

            assert result["plddt"] == 75.0
            assert len(result["plddt_per_residue"]) == 1

    @pytest.mark.asyncio
    async def test_lowercase_sequence(self):
        """Lowercase sequences should be accepted."""
        with patch("protein_design_mcp.tools.validate.ESMFoldRunner") as mock_runner:
            mock_instance = MagicMock()
            mock_instance.predict_structure = AsyncMock(
                return_value=PredictionResult(
                    sequence="MKVGA",
                    pdb_string="ATOM...",
                    plddt=85.0,
                    ptm=0.78,
                    plddt_per_residue=np.array([80, 85, 90, 85, 80]),
                    pae_matrix=None,
                )
            )
            mock_runner.return_value = mock_instance

            result = await validate_design("mkvga")

            assert result["plddt"] > 0

    @pytest.mark.asyncio
    async def test_plddt_per_residue_length(self):
        """plddt_per_residue length should match sequence length."""
        seq = "MKVGAAAAAA"  # 10 residues

        with patch("protein_design_mcp.tools.validate.ESMFoldRunner") as mock_runner:
            mock_instance = MagicMock()
            mock_instance.predict_structure = AsyncMock(
                return_value=PredictionResult(
                    sequence=seq,
                    pdb_string="ATOM...",
                    plddt=85.0,
                    ptm=0.78,
                    plddt_per_residue=np.array([80] * 10),
                    pae_matrix=None,
                )
            )
            mock_runner.return_value = mock_instance

            result = await validate_design(seq)

            assert len(result["plddt_per_residue"]) == len(seq)
