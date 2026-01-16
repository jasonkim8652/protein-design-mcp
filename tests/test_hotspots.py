"""Tests for suggest_hotspots tool."""

from pathlib import Path

import pytest

from protein_design_mcp.tools.hotspots import suggest_hotspots


# Test fixtures path
FIXTURES_DIR = Path(__file__).parent / "fixtures" / "test_pdbs"
MINI_PROTEIN_PDB = FIXTURES_DIR / "mini_protein.pdb"
TWO_CHAIN_PDB = FIXTURES_DIR / "two_chain_complex.pdb"


class TestSuggestHotspots:
    """Tests for suggest_hotspots function."""

    @pytest.mark.asyncio
    async def test_suggest_hotspots_returns_dict(self):
        """suggest_hotspots should return a dictionary."""
        result = await suggest_hotspots(target_pdb=str(MINI_PROTEIN_PDB))
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_suggest_hotspots_has_suggestions(self):
        """Result should include suggested_hotspots list."""
        result = await suggest_hotspots(target_pdb=str(MINI_PROTEIN_PDB))
        assert "suggested_hotspots" in result
        assert isinstance(result["suggested_hotspots"], list)

    @pytest.mark.asyncio
    async def test_hotspot_structure(self):
        """Each hotspot suggestion should have required fields."""
        result = await suggest_hotspots(target_pdb=str(MINI_PROTEIN_PDB))
        for hotspot in result["suggested_hotspots"]:
            assert "residues" in hotspot
            assert "score" in hotspot
            assert "rationale" in hotspot
            assert isinstance(hotspot["residues"], list)
            assert 0.0 <= hotspot["score"] <= 1.0

    @pytest.mark.asyncio
    async def test_suggest_hotspots_validates_pdb(self):
        """Should raise error for invalid PDB path."""
        with pytest.raises((FileNotFoundError, ValueError)):
            await suggest_hotspots(target_pdb="/nonexistent/target.pdb")

    @pytest.mark.asyncio
    async def test_suggest_hotspots_specific_chain(self):
        """Should analyze specific chain when specified."""
        result = await suggest_hotspots(
            target_pdb=str(TWO_CHAIN_PDB),
            chain_id="A",
        )
        # All residues should be from chain A
        for hotspot in result["suggested_hotspots"]:
            for residue in hotspot["residues"]:
                assert residue.startswith("A")

    @pytest.mark.asyncio
    async def test_suggest_hotspots_invalid_chain(self):
        """Should raise error for invalid chain ID."""
        with pytest.raises(ValueError, match="[Cc]hain"):
            await suggest_hotspots(
                target_pdb=str(MINI_PROTEIN_PDB),
                chain_id="Z",
            )

    @pytest.mark.asyncio
    async def test_suggest_hotspots_exposed_criteria(self):
        """Should find exposed residues by default."""
        result = await suggest_hotspots(
            target_pdb=str(MINI_PROTEIN_PDB),
            criteria="exposed",
        )
        # Should return at least one suggestion
        assert len(result["suggested_hotspots"]) >= 1

    @pytest.mark.asyncio
    async def test_suggest_hotspots_surface_analysis(self):
        """Should include surface analysis."""
        result = await suggest_hotspots(target_pdb=str(MINI_PROTEIN_PDB))
        assert "surface_analysis" in result
        assert "total_surface_area" in result["surface_analysis"]


class TestHotspotCriteria:
    """Tests for different hotspot criteria."""

    @pytest.mark.asyncio
    async def test_druggable_criteria(self):
        """Should find druggable sites."""
        result = await suggest_hotspots(
            target_pdb=str(MINI_PROTEIN_PDB),
            criteria="druggable",
        )
        assert "suggested_hotspots" in result

    @pytest.mark.asyncio
    async def test_exposed_criteria(self):
        """Should find exposed surfaces."""
        result = await suggest_hotspots(
            target_pdb=str(MINI_PROTEIN_PDB),
            criteria="exposed",
        )
        assert "suggested_hotspots" in result

    @pytest.mark.asyncio
    async def test_invalid_criteria(self):
        """Should raise error for invalid criteria."""
        with pytest.raises(ValueError, match="[Cc]riteria"):
            await suggest_hotspots(
                target_pdb=str(MINI_PROTEIN_PDB),
                criteria="invalid_criteria",
            )

    @pytest.mark.asyncio
    async def test_conserved_criteria(self):
        """Should find conserved sites."""
        result = await suggest_hotspots(
            target_pdb=str(MINI_PROTEIN_PDB),
            criteria="conserved",
        )
        assert "suggested_hotspots" in result


class TestEnhancedHotspots:
    """Tests for enhanced hotspot features."""

    @pytest.mark.asyncio
    async def test_hotspot_has_evidence(self):
        """Hotspots should include evidence dict."""
        result = await suggest_hotspots(target_pdb=str(MINI_PROTEIN_PDB))
        for hotspot in result["suggested_hotspots"]:
            assert "evidence" in hotspot
            assert isinstance(hotspot["evidence"], dict)

    @pytest.mark.asyncio
    async def test_surface_analysis_has_sasa(self):
        """Surface analysis should use SASA calculation."""
        result = await suggest_hotspots(target_pdb=str(MINI_PROTEIN_PDB))
        assert "total_surface_area" in result["surface_analysis"]
        assert result["surface_analysis"]["total_surface_area"] > 0

    @pytest.mark.asyncio
    async def test_surface_analysis_has_pockets(self):
        """Surface analysis may include detected pockets."""
        result = await suggest_hotspots(target_pdb=str(MINI_PROTEIN_PDB))
        # Pockets may or may not be detected depending on structure
        assert "surface_analysis" in result

    @pytest.mark.asyncio
    async def test_uniprot_id_parameter(self):
        """Should accept uniprot_id parameter."""
        # Should not raise - API call will fail but function should handle it
        result = await suggest_hotspots(
            target_pdb=str(MINI_PROTEIN_PDB),
            uniprot_id="P12345",  # Fake ID
        )
        assert "suggested_hotspots" in result

    @pytest.mark.asyncio
    async def test_include_literature_parameter(self):
        """Should accept include_literature parameter."""
        # Should not raise - API call will fail but function should handle it
        result = await suggest_hotspots(
            target_pdb=str(MINI_PROTEIN_PDB),
            include_literature=True,
        )
        assert "suggested_hotspots" in result
