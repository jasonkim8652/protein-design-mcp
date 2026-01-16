"""Tests for analyze_interface tool."""

from pathlib import Path

import pytest

from protein_design_mcp.tools.analyze import analyze_interface


# Test fixtures path
FIXTURES_DIR = Path(__file__).parent / "fixtures" / "test_pdbs"
TWO_CHAIN_PDB = FIXTURES_DIR / "two_chain_complex.pdb"
MINI_PROTEIN_PDB = FIXTURES_DIR / "mini_protein.pdb"


class TestAnalyzeInterface:
    """Tests for analyze_interface function."""

    @pytest.mark.asyncio
    async def test_analyze_interface_returns_dict(self):
        """analyze_interface should return a dictionary."""
        result = await analyze_interface(
            complex_pdb=str(TWO_CHAIN_PDB),
            chain_a="A",
            chain_b="B",
        )
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_analyze_interface_has_interface_residues(self):
        """Result should include interface residues for both chains."""
        result = await analyze_interface(
            complex_pdb=str(TWO_CHAIN_PDB),
            chain_a="A",
            chain_b="B",
        )
        assert "interface_residues" in result
        assert "chain_a" in result["interface_residues"]
        assert "chain_b" in result["interface_residues"]
        assert isinstance(result["interface_residues"]["chain_a"], list)
        assert isinstance(result["interface_residues"]["chain_b"], list)

    @pytest.mark.asyncio
    async def test_analyze_interface_has_buried_surface_area(self):
        """Result should include buried surface area."""
        result = await analyze_interface(
            complex_pdb=str(TWO_CHAIN_PDB),
            chain_a="A",
            chain_b="B",
        )
        assert "buried_surface_area" in result
        assert isinstance(result["buried_surface_area"], (int, float))
        assert result["buried_surface_area"] >= 0

    @pytest.mark.asyncio
    async def test_analyze_interface_has_contact_counts(self):
        """Result should include contact type counts."""
        result = await analyze_interface(
            complex_pdb=str(TWO_CHAIN_PDB),
            chain_a="A",
            chain_b="B",
        )
        assert "hydrogen_bonds" in result
        assert "salt_bridges" in result
        assert "hydrophobic_contacts" in result

    @pytest.mark.asyncio
    async def test_analyze_interface_validates_pdb(self):
        """Should raise error for invalid PDB path."""
        with pytest.raises((FileNotFoundError, ValueError)):
            await analyze_interface(
                complex_pdb="/nonexistent/complex.pdb",
                chain_a="A",
                chain_b="B",
            )

    @pytest.mark.asyncio
    async def test_analyze_interface_validates_chains(self):
        """Should raise error for invalid chain IDs."""
        with pytest.raises(ValueError, match="[Cc]hain"):
            await analyze_interface(
                complex_pdb=str(TWO_CHAIN_PDB),
                chain_a="X",  # Invalid chain
                chain_b="Y",
            )

    @pytest.mark.asyncio
    async def test_analyze_interface_single_chain_error(self):
        """Should raise error when analyzing single chain PDB."""
        with pytest.raises(ValueError, match="[Cc]hain"):
            await analyze_interface(
                complex_pdb=str(MINI_PROTEIN_PDB),
                chain_a="A",
                chain_b="B",  # Chain B doesn't exist
            )


class TestAnalyzeInterfaceMetrics:
    """Tests for interface metrics calculation."""

    @pytest.mark.asyncio
    async def test_interface_residue_count(self):
        """Should find reasonable number of interface residues."""
        result = await analyze_interface(
            complex_pdb=str(TWO_CHAIN_PDB),
            chain_a="A",
            chain_b="B",
        )
        # Two chain complex should have at least some interface residues
        chain_a_residues = result["interface_residues"]["chain_a"]
        chain_b_residues = result["interface_residues"]["chain_b"]

        # At least one residue per chain should be at interface
        assert len(chain_a_residues) >= 1
        assert len(chain_b_residues) >= 1

    @pytest.mark.asyncio
    async def test_contact_counts_non_negative(self):
        """All contact counts should be non-negative."""
        result = await analyze_interface(
            complex_pdb=str(TWO_CHAIN_PDB),
            chain_a="A",
            chain_b="B",
        )
        assert result["hydrogen_bonds"] >= 0
        assert result["salt_bridges"] >= 0
        assert result["hydrophobic_contacts"] >= 0
