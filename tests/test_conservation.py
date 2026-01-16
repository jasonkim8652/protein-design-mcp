"""Tests for conservation scoring."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from protein_design_mcp.utils.conservation import (
    calculate_conservation_scores,
    parse_blast_results,
    ConservationProfile,
)


class TestParseBlastResults:
    """Tests for parse_blast_results function."""

    def test_returns_list_of_sequences(self):
        """Should return list of aligned sequences."""
        mock_blast_output = """
>sp|P12345|EXAMPLE Protein [Homo sapiens]
MKVLAAAAVL
>sp|Q67890|HOMOLOG Protein [Mus musculus]
MKVLAAAAVL
>sp|P11111|ORTHOLOG Protein [Rattus norvegicus]
MKVLAAA-VL
"""
        result = parse_blast_results(mock_blast_output)
        assert len(result) >= 2
        assert isinstance(result, list)

    def test_handles_empty_blast(self):
        """Should handle empty BLAST results."""
        result = parse_blast_results("")
        assert result == []


class TestCalculateConservationScores:
    """Tests for calculate_conservation_scores function."""

    @pytest.mark.asyncio
    async def test_returns_conservation_profile(self):
        """Should return ConservationProfile object."""
        mock_blast_output = """
>sp|P12345|EXAMPLE Protein
MKVLAAAAVL
>sp|Q67890|HOMOLOG Protein
MKVLAAAAVL
"""
        with patch("protein_design_mcp.utils.conservation.run_blast",
                   new_callable=AsyncMock) as mock_blast:
            mock_blast.return_value = mock_blast_output

            result = await calculate_conservation_scores("MKVLAAAAVL")
            assert isinstance(result, ConservationProfile)

    @pytest.mark.asyncio
    async def test_conservation_scores_per_residue(self):
        """Should provide conservation score for each residue."""
        mock_blast_output = """
>sp|P12345|EXAMPLE Protein
MKVLAAAAVL
>sp|Q67890|HOMOLOG Protein
MKVLAAAAVL
>sp|P11111|ORTHOLOG Protein
MKVLAAAAVL
"""
        with patch("protein_design_mcp.utils.conservation.run_blast",
                   new_callable=AsyncMock) as mock_blast:
            mock_blast.return_value = mock_blast_output

            result = await calculate_conservation_scores("MKVLAAAAVL")
            assert len(result.scores) == 10  # 10 residues
            assert all(0.0 <= s <= 1.0 for s in result.scores)

    @pytest.mark.asyncio
    async def test_highly_conserved_residues(self):
        """Should identify highly conserved positions."""
        # All sequences have same amino acids at all positions
        mock_blast_output = """
>sp|P12345|EXAMPLE Protein
MKVL
>sp|Q67890|HOMOLOG Protein
MKVL
>sp|P11111|ORTHOLOG Protein
MKVL
"""
        with patch("protein_design_mcp.utils.conservation.run_blast",
                   new_callable=AsyncMock) as mock_blast:
            mock_blast.return_value = mock_blast_output

            result = await calculate_conservation_scores("MKVL")
            # All positions should be highly conserved
            assert len(result.highly_conserved) == 4

    @pytest.mark.asyncio
    async def test_variable_positions(self):
        """Should detect variable positions."""
        # Position 2 varies (K vs R vs H)
        mock_blast_output = """
>sp|P12345|EXAMPLE Protein
MKVL
>sp|Q67890|HOMOLOG Protein
MRVL
>sp|P11111|ORTHOLOG Protein
MHVL
"""
        with patch("protein_design_mcp.utils.conservation.run_blast",
                   new_callable=AsyncMock) as mock_blast:
            mock_blast.return_value = mock_blast_output

            result = await calculate_conservation_scores("MKVL")
            # Position 2 (index 1) should have lower conservation
            assert result.scores[1] < 1.0

    @pytest.mark.asyncio
    async def test_handles_no_blast_hits(self):
        """Should handle case with no BLAST hits gracefully."""
        with patch("protein_design_mcp.utils.conservation.run_blast",
                   new_callable=AsyncMock) as mock_blast:
            mock_blast.return_value = ""

            result = await calculate_conservation_scores("MKVLAAAAVL")
            # Should return default scores (e.g., 0.5 or based on input sequence)
            assert isinstance(result, ConservationProfile)
            assert len(result.scores) == 10

    @pytest.mark.asyncio
    async def test_average_conservation(self):
        """Should calculate average conservation score."""
        mock_blast_output = """
>sp|P12345|EXAMPLE Protein
MKVL
>sp|Q67890|HOMOLOG Protein
MKVL
"""
        with patch("protein_design_mcp.utils.conservation.run_blast",
                   new_callable=AsyncMock) as mock_blast:
            mock_blast.return_value = mock_blast_output

            result = await calculate_conservation_scores("MKVL")
            assert 0.0 <= result.average_conservation <= 1.0


class TestConservationProfile:
    """Tests for ConservationProfile dataclass."""

    def test_profile_creation(self):
        """Should create ConservationProfile with required fields."""
        profile = ConservationProfile(
            sequence="MKVL",
            scores=[0.9, 0.8, 0.7, 0.6],
            highly_conserved=[1],
            num_homologs=5,
        )
        assert profile.sequence == "MKVL"
        assert len(profile.scores) == 4
        assert profile.num_homologs == 5

    def test_average_conservation_property(self):
        """Should calculate average conservation."""
        profile = ConservationProfile(
            sequence="MKVL",
            scores=[1.0, 0.8, 0.6, 0.4],
            highly_conserved=[1],
            num_homologs=5,
        )
        assert profile.average_conservation == 0.7

    def test_get_score_for_position(self):
        """Should get conservation score for specific position."""
        profile = ConservationProfile(
            sequence="MKVL",
            scores=[0.9, 0.8, 0.7, 0.6],
            highly_conserved=[1],
            num_homologs=5,
        )
        assert profile.get_score(1) == 0.9  # 1-indexed
        assert profile.get_score(4) == 0.6
