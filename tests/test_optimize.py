"""Tests for optimize_sequence tool."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import numpy as np

from protein_design_mcp.pipelines.esmfold import PredictionResult


# Test fixtures path
FIXTURES_DIR = Path(__file__).parent / "fixtures" / "test_pdbs"
MINI_PROTEIN_PDB = FIXTURES_DIR / "mini_protein.pdb"

# Get module to patch
optimize_module = sys.modules.get("protein_design_mcp.tools.optimize")


def create_mock_mpnn():
    """Create mock ProteinMPNN runner."""
    mock = MagicMock()
    mock.design_sequences = AsyncMock(return_value=[
        {"id": "opt_0", "sequence": "MKVVLAAAA", "score": -2.0},
        {"id": "opt_1", "sequence": "MKVGVAAAA", "score": -1.8},
    ])
    return mock


def create_mock_esm(plddt=85.0, ptm=0.8):
    """Create mock ESMFold runner."""
    mock = MagicMock()
    mock.predict_structure = AsyncMock(return_value=PredictionResult(
        sequence="MKVVLAAAA",
        pdb_string="ATOM...",
        plddt=plddt,
        ptm=ptm,
        plddt_per_residue=np.array([plddt] * 9),
    ))
    return mock


class TestOptimizeSequence:
    """Tests for optimize_sequence function."""

    @pytest.mark.asyncio
    async def test_optimize_sequence_returns_dict(self, tmp_path):
        """optimize_sequence should return a dictionary."""
        from protein_design_mcp.tools.optimize import optimize_sequence
        optimize_mod = sys.modules["protein_design_mcp.tools.optimize"]

        mock_mpnn = create_mock_mpnn()
        mock_esm = create_mock_esm()

        with patch.object(optimize_mod, "ProteinMPNNRunner", return_value=mock_mpnn):
            with patch.object(optimize_mod, "ESMFoldRunner", return_value=mock_esm):
                result = await optimize_sequence(
                    current_sequence="MKVGAAAAA",
                    target_pdb=str(MINI_PROTEIN_PDB),
                )
                assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_optimize_sequence_has_optimized_sequence(self, tmp_path):
        """Result should include optimized sequence."""
        from protein_design_mcp.tools.optimize import optimize_sequence
        optimize_mod = sys.modules["protein_design_mcp.tools.optimize"]

        mock_mpnn = create_mock_mpnn()
        mock_esm = create_mock_esm()

        with patch.object(optimize_mod, "ProteinMPNNRunner", return_value=mock_mpnn):
            with patch.object(optimize_mod, "ESMFoldRunner", return_value=mock_esm):
                result = await optimize_sequence(
                    current_sequence="MKVGAAAAA",
                    target_pdb=str(MINI_PROTEIN_PDB),
                )
                assert "optimized_sequence" in result
                assert isinstance(result["optimized_sequence"], str)

    @pytest.mark.asyncio
    async def test_optimize_sequence_has_mutations(self, tmp_path):
        """Result should list mutations made."""
        from protein_design_mcp.tools.optimize import optimize_sequence
        optimize_mod = sys.modules["protein_design_mcp.tools.optimize"]

        mock_mpnn = create_mock_mpnn()
        mock_esm = create_mock_esm()

        with patch.object(optimize_mod, "ProteinMPNNRunner", return_value=mock_mpnn):
            with patch.object(optimize_mod, "ESMFoldRunner", return_value=mock_esm):
                result = await optimize_sequence(
                    current_sequence="MKVGAAAAA",
                    target_pdb=str(MINI_PROTEIN_PDB),
                )
                assert "mutations" in result
                assert isinstance(result["mutations"], list)

    @pytest.mark.asyncio
    async def test_optimize_sequence_has_metrics(self, tmp_path):
        """Result should include quality metrics."""
        from protein_design_mcp.tools.optimize import optimize_sequence
        optimize_mod = sys.modules["protein_design_mcp.tools.optimize"]

        mock_mpnn = create_mock_mpnn()
        mock_esm = create_mock_esm()

        with patch.object(optimize_mod, "ProteinMPNNRunner", return_value=mock_mpnn):
            with patch.object(optimize_mod, "ESMFoldRunner", return_value=mock_esm):
                result = await optimize_sequence(
                    current_sequence="MKVGAAAAA",
                    target_pdb=str(MINI_PROTEIN_PDB),
                )
                assert "metrics" in result
                assert "plddt" in result["metrics"]

    @pytest.mark.asyncio
    async def test_optimize_sequence_validates_target(self):
        """Should raise error for invalid target PDB."""
        from protein_design_mcp.tools.optimize import optimize_sequence

        with pytest.raises((FileNotFoundError, ValueError)):
            await optimize_sequence(
                current_sequence="MKVGAAAAA",
                target_pdb="/nonexistent/target.pdb",
            )

    @pytest.mark.asyncio
    async def test_optimize_sequence_validates_sequence(self):
        """Should raise error for invalid sequence."""
        from protein_design_mcp.tools.optimize import optimize_sequence

        with pytest.raises(ValueError, match="[Ss]equence"):
            await optimize_sequence(
                current_sequence="INVALID123",  # Contains digits
                target_pdb=str(MINI_PROTEIN_PDB),
            )

    @pytest.mark.asyncio
    async def test_optimize_sequence_empty_sequence(self):
        """Should raise error for empty sequence."""
        from protein_design_mcp.tools.optimize import optimize_sequence

        with pytest.raises(ValueError, match="[Ss]equence"):
            await optimize_sequence(
                current_sequence="",
                target_pdb=str(MINI_PROTEIN_PDB),
            )


class TestOptimizationTargets:
    """Tests for different optimization targets."""

    @pytest.mark.asyncio
    async def test_stability_optimization(self, tmp_path):
        """Should optimize for stability."""
        from protein_design_mcp.tools.optimize import optimize_sequence
        optimize_mod = sys.modules["protein_design_mcp.tools.optimize"]

        mock_mpnn = create_mock_mpnn()
        mock_esm = create_mock_esm()

        with patch.object(optimize_mod, "ProteinMPNNRunner", return_value=mock_mpnn):
            with patch.object(optimize_mod, "ESMFoldRunner", return_value=mock_esm):
                result = await optimize_sequence(
                    current_sequence="MKVGAAAAA",
                    target_pdb=str(MINI_PROTEIN_PDB),
                    optimization_target="stability",
                )
                assert "optimized_sequence" in result

    @pytest.mark.asyncio
    async def test_affinity_optimization(self, tmp_path):
        """Should optimize for affinity."""
        from protein_design_mcp.tools.optimize import optimize_sequence
        optimize_mod = sys.modules["protein_design_mcp.tools.optimize"]

        mock_mpnn = create_mock_mpnn()
        mock_esm = create_mock_esm()

        with patch.object(optimize_mod, "ProteinMPNNRunner", return_value=mock_mpnn):
            with patch.object(optimize_mod, "ESMFoldRunner", return_value=mock_esm):
                result = await optimize_sequence(
                    current_sequence="MKVGAAAAA",
                    target_pdb=str(MINI_PROTEIN_PDB),
                    optimization_target="affinity",
                )
                assert "optimized_sequence" in result

    @pytest.mark.asyncio
    async def test_invalid_optimization_target(self):
        """Should raise error for invalid optimization target."""
        from protein_design_mcp.tools.optimize import optimize_sequence

        with pytest.raises(ValueError, match="[Oo]ptimization"):
            await optimize_sequence(
                current_sequence="MKVGAAAAA",
                target_pdb=str(MINI_PROTEIN_PDB),
                optimization_target="invalid_target",
            )
