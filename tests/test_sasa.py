"""Tests for SASA calculation and pocket detection."""

import pytest
from pathlib import Path

from protein_design_mcp.utils.sasa import (
    calculate_sasa,
    detect_pockets,
    SASAResult,
    Pocket,
)


# Test fixtures path
FIXTURES_DIR = Path(__file__).parent / "fixtures" / "test_pdbs"
MINI_PROTEIN_PDB = FIXTURES_DIR / "mini_protein.pdb"


class TestCalculateSASA:
    """Tests for calculate_sasa function."""

    def test_returns_sasa_result(self):
        """Should return SASAResult object."""
        result = calculate_sasa(str(MINI_PROTEIN_PDB))
        assert isinstance(result, SASAResult)

    def test_has_total_sasa(self):
        """Should calculate total SASA."""
        result = calculate_sasa(str(MINI_PROTEIN_PDB))
        assert result.total_sasa > 0

    def test_has_per_residue_sasa(self):
        """Should provide per-residue SASA values."""
        result = calculate_sasa(str(MINI_PROTEIN_PDB))
        assert len(result.per_residue) > 0
        assert all(sasa >= 0 for sasa in result.per_residue.values())

    def test_identifies_exposed_residues(self):
        """Should identify surface-exposed residues."""
        result = calculate_sasa(str(MINI_PROTEIN_PDB))
        assert len(result.exposed_residues) > 0

    def test_invalid_pdb_raises_error(self):
        """Should raise error for invalid PDB."""
        with pytest.raises((FileNotFoundError, ValueError)):
            calculate_sasa("/nonexistent/file.pdb")


class TestDetectPockets:
    """Tests for detect_pockets function."""

    def test_returns_list_of_pockets(self):
        """Should return list of Pocket objects."""
        result = detect_pockets(str(MINI_PROTEIN_PDB))
        assert isinstance(result, list)
        for pocket in result:
            assert isinstance(pocket, Pocket)

    def test_pocket_has_required_fields(self):
        """Each pocket should have required fields."""
        result = detect_pockets(str(MINI_PROTEIN_PDB))
        for pocket in result:
            assert hasattr(pocket, "center")
            assert hasattr(pocket, "volume")
            assert hasattr(pocket, "residues")
            assert hasattr(pocket, "druggability")

    def test_pocket_residues_list(self):
        """Pocket should have list of residue identifiers."""
        result = detect_pockets(str(MINI_PROTEIN_PDB))
        for pocket in result:
            assert isinstance(pocket.residues, list)

    def test_druggability_score_range(self):
        """Druggability score should be between 0 and 1."""
        result = detect_pockets(str(MINI_PROTEIN_PDB))
        for pocket in result:
            assert 0.0 <= pocket.druggability <= 1.0


class TestSASAResult:
    """Tests for SASAResult dataclass."""

    def test_sasa_result_creation(self):
        """Should create SASAResult with required fields."""
        result = SASAResult(
            total_sasa=1500.0,
            per_residue={"A1": 50.0, "A2": 75.0},
            exposed_residues=["A1"],
        )
        assert result.total_sasa == 1500.0
        assert result.per_residue["A1"] == 50.0

    def test_get_residue_exposure(self):
        """Should get SASA for specific residue."""
        result = SASAResult(
            total_sasa=1500.0,
            per_residue={"A1": 50.0, "A2": 75.0},
            exposed_residues=["A1"],
        )
        assert result.get_residue_sasa("A1") == 50.0
        assert result.get_residue_sasa("A2") == 75.0


class TestPocket:
    """Tests for Pocket dataclass."""

    def test_pocket_creation(self):
        """Should create Pocket with required fields."""
        pocket = Pocket(
            center=(10.0, 20.0, 30.0),
            volume=250.0,
            residues=["A10", "A11", "A12"],
            druggability=0.75,
        )
        assert pocket.center == (10.0, 20.0, 30.0)
        assert pocket.volume == 250.0
        assert len(pocket.residues) == 3
        assert pocket.druggability == 0.75
