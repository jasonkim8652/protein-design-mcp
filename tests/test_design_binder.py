"""Tests for design_binder tool."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np

import pytest

from protein_design_mcp.tools.design_binder import design_binder, DesignResult
from protein_design_mcp.pipelines.esmfold import PredictionResult

# Get the actual module (not the function) to patch
design_binder_module = sys.modules["protein_design_mcp.tools.design_binder"]


# Test fixtures path
FIXTURES_DIR = Path(__file__).parent / "fixtures" / "test_pdbs"
MINI_PROTEIN_PDB = FIXTURES_DIR / "mini_protein.pdb"


# Helper to create mock runners
def create_mock_runners(tmp_path, num_backbones=1, num_seqs=1, plddt=80.0, ptm=0.7):
    """Create mock runners for testing."""
    mock_rfd = MagicMock()
    mock_rfd.generate_backbones = AsyncMock(return_value=[
        {"id": f"design_{i:04d}", "pdb_path": str(tmp_path / f"backbone_{i}.pdb")}
        for i in range(num_backbones)
    ])

    mock_mpnn = MagicMock()
    mock_mpnn.design_sequences = AsyncMock(return_value=[
        {"id": f"seq_{i}", "sequence": "MKVGAAAAAA", "score": -1.5}
        for i in range(num_seqs)
    ])

    mock_esm = MagicMock()
    mock_esm.predict_structure = AsyncMock(return_value=PredictionResult(
        sequence="MKVGAAAAAA",
        pdb_string="ATOM...",
        plddt=plddt,
        ptm=ptm,
        plddt_per_residue=np.array([plddt] * 10),
    ))

    return mock_rfd, mock_mpnn, mock_esm


class TestDesignBinder:
    """Tests for design_binder function."""

    @pytest.mark.asyncio
    async def test_design_binder_returns_dict(self, tmp_path):
        """design_binder should return a dictionary with results."""
        mock_rfd, mock_mpnn, mock_esm = create_mock_runners(tmp_path)

        with patch.object(design_binder_module, "RFdiffusionRunner", return_value=mock_rfd):
            with patch.object(design_binder_module, "ProteinMPNNRunner", return_value=mock_mpnn):
                with patch.object(design_binder_module, "ESMFoldRunner", return_value=mock_esm):
                    result = await design_binder(
                        target_pdb=str(MINI_PROTEIN_PDB),
                        hotspot_residues=["A1", "A2"],
                        num_designs=1,
                    )

                    assert isinstance(result, dict)
                    assert "designs" in result
                    assert "summary" in result

    @pytest.mark.asyncio
    async def test_design_binder_validates_target(self):
        """Should raise error for invalid target PDB."""
        with pytest.raises((FileNotFoundError, ValueError)):
            await design_binder(
                target_pdb="/nonexistent/target.pdb",
                hotspot_residues=["A1"],
            )

    @pytest.mark.asyncio
    async def test_design_binder_validates_hotspots(self):
        """Should raise error for empty hotspot list."""
        with pytest.raises(ValueError, match="[Hh]otspot"):
            await design_binder(
                target_pdb=str(MINI_PROTEIN_PDB),
                hotspot_residues=[],
            )

    @pytest.mark.asyncio
    async def test_design_binder_returns_designs_list(self, tmp_path):
        """Should return list of designs with expected structure."""
        mock_rfd, mock_mpnn, mock_esm = create_mock_runners(tmp_path)

        with patch.object(design_binder_module, "RFdiffusionRunner", return_value=mock_rfd):
            with patch.object(design_binder_module, "ProteinMPNNRunner", return_value=mock_mpnn):
                with patch.object(design_binder_module, "ESMFoldRunner", return_value=mock_esm):
                    result = await design_binder(
                        target_pdb=str(MINI_PROTEIN_PDB),
                        hotspot_residues=["A1"],
                        num_designs=1,
                    )

                    designs = result["designs"]
                    assert isinstance(designs, list)
                    for design in designs:
                        assert "id" in design
                        assert "sequence" in design
                        assert "metrics" in design
                        assert "plddt" in design["metrics"]
                        assert "ptm" in design["metrics"]

    @pytest.mark.asyncio
    async def test_design_binder_returns_summary(self, tmp_path):
        """Should return summary with statistics."""
        mock_rfd, mock_mpnn, mock_esm = create_mock_runners(tmp_path)

        with patch.object(design_binder_module, "RFdiffusionRunner", return_value=mock_rfd):
            with patch.object(design_binder_module, "ProteinMPNNRunner", return_value=mock_mpnn):
                with patch.object(design_binder_module, "ESMFoldRunner", return_value=mock_esm):
                    result = await design_binder(
                        target_pdb=str(MINI_PROTEIN_PDB),
                        hotspot_residues=["A1"],
                        num_designs=1,
                    )

                    summary = result["summary"]
                    assert "total_generated" in summary
                    assert "passed_filters" in summary


class TestDesignResult:
    """Tests for DesignResult dataclass."""

    def test_design_result_fields(self):
        """DesignResult should have all required fields."""
        result = DesignResult(
            id="design_001",
            sequence="MKVGA",
            structure_pdb="ATOM...",
            metrics={"plddt": 85.0, "ptm": 0.78},
        )
        assert result.id == "design_001"
        assert result.sequence == "MKVGA"
        assert result.metrics["plddt"] == 85.0


class TestPipelineIntegration:
    """Integration tests for pipeline orchestration."""

    @pytest.mark.asyncio
    async def test_pipeline_runs_in_order(self, tmp_path):
        """Pipeline should run RFdiffusion -> ProteinMPNN -> ESMFold."""
        call_order = []

        mock_rfd = MagicMock()
        mock_mpnn = MagicMock()
        mock_esm = MagicMock()

        async def rfd_side_effect(*args, **kwargs):
            call_order.append("rfdiffusion")
            return [{"id": "d0", "pdb_path": str(tmp_path / "bb.pdb")}]

        async def mpnn_side_effect(*args, **kwargs):
            call_order.append("proteinmpnn")
            return [{"id": "s0", "sequence": "AAAA", "score": -1.0}]

        async def esm_side_effect(*args, **kwargs):
            call_order.append("esmfold")
            return PredictionResult(
                sequence="AAAA",
                pdb_string="ATOM...",
                plddt=80.0,
                ptm=0.7,
                plddt_per_residue=np.array([80, 80, 80, 80]),
            )

        mock_rfd.generate_backbones = AsyncMock(side_effect=rfd_side_effect)
        mock_mpnn.design_sequences = AsyncMock(side_effect=mpnn_side_effect)
        mock_esm.predict_structure = AsyncMock(side_effect=esm_side_effect)

        # Create backbone file so the pipeline continues
        (tmp_path / "bb.pdb").write_text("ATOM  1 CA ALA A 1 0.0 0.0 0.0\nEND\n")

        with patch.object(design_binder_module, "RFdiffusionRunner", return_value=mock_rfd):
            with patch.object(design_binder_module, "ProteinMPNNRunner", return_value=mock_mpnn):
                with patch.object(design_binder_module, "ESMFoldRunner", return_value=mock_esm):
                    await design_binder(
                        target_pdb=str(MINI_PROTEIN_PDB),
                        hotspot_residues=["A1"],
                        num_designs=1,
                    )

                    assert call_order == ["rfdiffusion", "proteinmpnn", "esmfold"]

    @pytest.mark.asyncio
    async def test_multiple_sequences_per_backbone(self, tmp_path):
        """Should design multiple sequences per backbone."""
        mock_rfd = MagicMock()
        mock_mpnn = MagicMock()
        mock_esm = MagicMock()

        # 1 backbone
        mock_rfd.generate_backbones = AsyncMock(return_value=[
            {"id": "d0", "pdb_path": str(tmp_path / "bb.pdb")}
        ])

        # 8 sequences per backbone (default)
        mock_mpnn.design_sequences = AsyncMock(return_value=[
            {"id": f"s{i}", "sequence": "AAAA", "score": -1.0}
            for i in range(8)
        ])

        mock_esm.predict_structure = AsyncMock(return_value=PredictionResult(
            sequence="AAAA",
            pdb_string="ATOM...",
            plddt=80.0,
            ptm=0.7,
            plddt_per_residue=np.array([80, 80, 80, 80]),
        ))

        # Create backbone file
        (tmp_path / "bb.pdb").write_text("ATOM  1 CA ALA A 1 0.0 0.0 0.0\nEND\n")

        with patch.object(design_binder_module, "RFdiffusionRunner", return_value=mock_rfd):
            with patch.object(design_binder_module, "ProteinMPNNRunner", return_value=mock_mpnn):
                with patch.object(design_binder_module, "ESMFoldRunner", return_value=mock_esm):
                    result = await design_binder(
                        target_pdb=str(MINI_PROTEIN_PDB),
                        hotspot_residues=["A1"],
                        num_designs=1,
                    )

                    # Should have 8 designs (8 sequences from ProteinMPNN)
                    assert len(result["designs"]) == 8


class TestEdgeCases:
    """Edge case tests."""

    @pytest.mark.asyncio
    async def test_single_design(self, tmp_path):
        """Should work with num_designs=1."""
        mock_rfd, mock_mpnn, mock_esm = create_mock_runners(tmp_path, plddt=75.0, ptm=0.6)

        # Create backbone file
        (tmp_path / "backbone_0.pdb").write_text("ATOM  1 CA ALA A 1 0.0 0.0 0.0\nEND\n")

        with patch.object(design_binder_module, "RFdiffusionRunner", return_value=mock_rfd):
            with patch.object(design_binder_module, "ProteinMPNNRunner", return_value=mock_mpnn):
                with patch.object(design_binder_module, "ESMFoldRunner", return_value=mock_esm):
                    result = await design_binder(
                        target_pdb=str(MINI_PROTEIN_PDB),
                        hotspot_residues=["A1"],
                        num_designs=1,
                    )

                    assert result["summary"]["total_generated"] >= 1

    @pytest.mark.asyncio
    async def test_custom_binder_length(self, tmp_path):
        """Should pass custom binder length to RFdiffusion."""
        mock_rfd = MagicMock()
        mock_rfd.generate_backbones = AsyncMock(return_value=[])

        with patch.object(design_binder_module, "RFdiffusionRunner", return_value=mock_rfd):
            with patch.object(design_binder_module, "ProteinMPNNRunner"):
                with patch.object(design_binder_module, "ESMFoldRunner"):
                    await design_binder(
                        target_pdb=str(MINI_PROTEIN_PDB),
                        hotspot_residues=["A1"],
                        binder_length=100,
                    )

                    # Verify binder_length was passed
                    call_args = mock_rfd.generate_backbones.call_args
                    assert call_args.kwargs.get("binder_length") == 100
