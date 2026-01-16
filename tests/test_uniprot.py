"""Tests for UniProt API integration."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from protein_design_mcp.utils.uniprot import (
    fetch_uniprot_features,
    parse_uniprot_response,
    UniProtFeatures,
    BindingSite,
    ActiveSite,
)


class TestParseUniProtResponse:
    """Tests for parse_uniprot_response function (no network calls)."""

    def test_returns_uniprot_features_object(self):
        """Should return a UniProtFeatures dataclass."""
        mock_response = {
            "features": [
                {
                    "type": "Binding site",
                    "location": {"start": {"value": 45}, "end": {"value": 49}},
                    "ligand": {"name": "ATP"},
                },
            ],
            "sequence": {"value": "MKVLAAAA"},
        }

        result = parse_uniprot_response("P12345", mock_response)
        assert isinstance(result, UniProtFeatures)

    def test_extracts_binding_sites(self):
        """Should extract binding site annotations."""
        mock_response = {
            "features": [
                {
                    "type": "Binding site",
                    "location": {"start": {"value": 45}, "end": {"value": 49}},
                    "ligand": {"name": "ATP"},
                },
                {
                    "type": "Binding site",
                    "location": {"start": {"value": 120}, "end": {"value": 125}},
                    "ligand": {"name": "Mg2+"},
                },
            ],
            "sequence": {"value": "MKVLAAAA"},
        }

        result = parse_uniprot_response("P12345", mock_response)
        assert len(result.binding_sites) == 2
        assert result.binding_sites[0].start == 45
        assert result.binding_sites[0].end == 49
        assert result.binding_sites[0].ligand == "ATP"

    def test_extracts_active_sites(self):
        """Should extract active site annotations."""
        mock_response = {
            "features": [
                {
                    "type": "Active site",
                    "location": {"start": {"value": 120}, "end": {"value": 120}},
                    "description": "Proton acceptor",
                },
            ],
            "sequence": {"value": "MKVLAAAA"},
        }

        result = parse_uniprot_response("P12345", mock_response)
        assert len(result.active_sites) == 1
        assert result.active_sites[0].position == 120
        assert result.active_sites[0].description == "Proton acceptor"

    def test_extracts_known_interactors(self):
        """Should extract known protein interactors."""
        mock_response = {
            "features": [],
            "comments": [
                {
                    "commentType": "INTERACTION",
                    "interactions": [
                        {"interactantOne": {"uniProtKBAccession": "P12345"},
                         "interactantTwo": {"uniProtKBAccession": "Q67890"}},
                        {"interactantOne": {"uniProtKBAccession": "P12345"},
                         "interactantTwo": {"uniProtKBAccession": "P11111"}},
                    ],
                },
            ],
            "sequence": {"value": "MKVLAAAA"},
        }

        result = parse_uniprot_response("P12345", mock_response)
        assert "Q67890" in result.known_interactors
        assert "P11111" in result.known_interactors

    def test_handles_empty_features(self):
        """Should handle proteins with no annotated features."""
        mock_response = {
            "features": [],
            "sequence": {"value": "MKVLAAAA"},
        }

        result = parse_uniprot_response("P12345", mock_response)
        assert result.binding_sites == []
        assert result.active_sites == []


class TestFetchUniProtFeatures:
    """Tests for fetch_uniprot_features function."""

    @pytest.mark.asyncio
    async def test_calls_api_with_correct_url(self):
        """Should call UniProt API with correct URL."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "features": [],
            "sequence": {"value": "MKVLAAAA"},
        })

        with patch("protein_design_mcp.utils.uniprot.aiohttp.ClientSession") as mock_session:
            mock_get = AsyncMock()
            mock_get.__aenter__ = AsyncMock(return_value=mock_response)
            mock_get.__aexit__ = AsyncMock(return_value=None)

            mock_session_instance = MagicMock()
            mock_session_instance.get = MagicMock(return_value=mock_get)
            mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
            mock_session_instance.__aexit__ = AsyncMock(return_value=None)

            mock_session.return_value = mock_session_instance

            await fetch_uniprot_features("P12345")

            mock_session_instance.get.assert_called_once()
            call_args = mock_session_instance.get.call_args
            assert "P12345" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_handles_invalid_uniprot_id(self):
        """Should raise error for invalid UniProt ID."""
        mock_response = MagicMock()
        mock_response.status = 404

        with patch("protein_design_mcp.utils.uniprot.aiohttp.ClientSession") as mock_session:
            mock_get = AsyncMock()
            mock_get.__aenter__ = AsyncMock(return_value=mock_response)
            mock_get.__aexit__ = AsyncMock(return_value=None)

            mock_session_instance = MagicMock()
            mock_session_instance.get = MagicMock(return_value=mock_get)
            mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
            mock_session_instance.__aexit__ = AsyncMock(return_value=None)

            mock_session.return_value = mock_session_instance

            with pytest.raises(ValueError, match="[Nn]ot found"):
                await fetch_uniprot_features("INVALID_ID")


class TestBindingSite:
    """Tests for BindingSite dataclass."""

    def test_binding_site_creation(self):
        """Should create BindingSite with required fields."""
        site = BindingSite(start=45, end=49, ligand="ATP")
        assert site.start == 45
        assert site.end == 49
        assert site.ligand == "ATP"

    def test_binding_site_residue_range(self):
        """Should provide residue range as list."""
        site = BindingSite(start=45, end=49, ligand="ATP")
        assert site.residue_range() == [45, 46, 47, 48, 49]


class TestActiveSite:
    """Tests for ActiveSite dataclass."""

    def test_active_site_creation(self):
        """Should create ActiveSite with required fields."""
        site = ActiveSite(position=120, description="Catalytic")
        assert site.position == 120
        assert site.description == "Catalytic"
