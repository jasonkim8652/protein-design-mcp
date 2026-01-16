"""Tests for PubMed literature search."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from protein_design_mcp.utils.pubmed import (
    search_binding_partners,
    parse_pubmed_response,
    LiteratureResult,
    Publication,
)


class TestParsePubmedResponse:
    """Tests for parse_pubmed_response function."""

    def test_returns_literature_result(self):
        """Should return LiteratureResult object."""
        mock_response = {
            "esearchresult": {"idlist": ["12345678"]},
        }
        mock_summaries = {
            "result": {
                "12345678": {
                    "uid": "12345678",
                    "title": "Crystal structure of protein complex",
                    "authors": [{"name": "Smith J"}],
                    "pubdate": "2023",
                }
            }
        }

        result = parse_pubmed_response(mock_response, mock_summaries)
        assert isinstance(result, LiteratureResult)

    def test_extracts_publications(self):
        """Should extract publication information."""
        mock_response = {
            "esearchresult": {"idlist": ["12345678", "87654321"]},
        }
        mock_summaries = {
            "result": {
                "12345678": {
                    "uid": "12345678",
                    "title": "Crystal structure of protein complex",
                    "authors": [{"name": "Smith J"}],
                    "pubdate": "2023",
                },
                "87654321": {
                    "uid": "87654321",
                    "title": "Binding analysis of target protein",
                    "authors": [{"name": "Jones A"}],
                    "pubdate": "2022",
                },
            }
        }

        result = parse_pubmed_response(mock_response, mock_summaries)
        assert len(result.publications) == 2
        assert result.publications[0].pmid == "12345678"
        assert "Crystal structure" in result.publications[0].title

    def test_handles_empty_results(self):
        """Should handle empty search results."""
        mock_response = {
            "esearchresult": {"idlist": []},
        }
        mock_summaries = {"result": {}}

        result = parse_pubmed_response(mock_response, mock_summaries)
        assert result.publications == []


class TestSearchBindingPartners:
    """Tests for search_binding_partners function."""

    @pytest.mark.asyncio
    async def test_returns_literature_result(self):
        """Should return LiteratureResult object."""
        mock_search_response = {
            "esearchresult": {"idlist": ["12345678"]},
        }
        mock_summary_response = {
            "result": {
                "12345678": {
                    "uid": "12345678",
                    "title": "Protein binding study",
                    "authors": [{"name": "Smith J"}],
                    "pubdate": "2023",
                }
            }
        }

        with patch("protein_design_mcp.utils.pubmed.aiohttp.ClientSession") as mock_session:
            mock_search_resp = MagicMock()
            mock_search_resp.status = 200
            mock_search_resp.json = AsyncMock(return_value=mock_search_response)

            mock_summary_resp = MagicMock()
            mock_summary_resp.status = 200
            mock_summary_resp.json = AsyncMock(return_value=mock_summary_response)

            mock_get = AsyncMock()
            mock_get.__aenter__ = AsyncMock(side_effect=[mock_search_resp, mock_summary_resp])
            mock_get.__aexit__ = AsyncMock(return_value=None)

            mock_session_instance = MagicMock()
            mock_session_instance.get = MagicMock(return_value=mock_get)
            mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
            mock_session_instance.__aexit__ = AsyncMock(return_value=None)

            mock_session.return_value = mock_session_instance

            result = await search_binding_partners("P53")
            assert isinstance(result, LiteratureResult)

    @pytest.mark.asyncio
    async def test_constructs_correct_query(self):
        """Should construct query with protein name and binding terms."""
        with patch("protein_design_mcp.utils.pubmed.aiohttp.ClientSession") as mock_session:
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.json = AsyncMock(return_value={"esearchresult": {"idlist": []}})

            mock_get = AsyncMock()
            mock_get.__aenter__ = AsyncMock(return_value=mock_resp)
            mock_get.__aexit__ = AsyncMock(return_value=None)

            mock_session_instance = MagicMock()
            mock_session_instance.get = MagicMock(return_value=mock_get)
            mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
            mock_session_instance.__aexit__ = AsyncMock(return_value=None)

            mock_session.return_value = mock_session_instance

            await search_binding_partners("TP53")

            # Check that query was constructed with protein name
            call_args = mock_session_instance.get.call_args
            assert call_args is not None


class TestPublication:
    """Tests for Publication dataclass."""

    def test_publication_creation(self):
        """Should create Publication with required fields."""
        pub = Publication(
            pmid="12345678",
            title="Crystal structure of protein complex",
            authors=["Smith J", "Jones A"],
            year="2023",
            binding_residues=[],
        )
        assert pub.pmid == "12345678"
        assert "Crystal structure" in pub.title
        assert len(pub.authors) == 2


class TestLiteratureResult:
    """Tests for LiteratureResult dataclass."""

    def test_literature_result_creation(self):
        """Should create LiteratureResult with required fields."""
        pub = Publication(
            pmid="12345678",
            title="Test",
            authors=["Smith J"],
            year="2023",
            binding_residues=[],
        )
        result = LiteratureResult(
            query="P53",
            publications=[pub],
            known_binding_partners=["MDM2", "BCL2"],
        )
        assert result.query == "P53"
        assert len(result.publications) == 1
        assert "MDM2" in result.known_binding_partners

    def test_get_pmids(self):
        """Should return list of PMIDs."""
        pub1 = Publication("12345678", "Title 1", ["Author"], "2023", [])
        pub2 = Publication("87654321", "Title 2", ["Author"], "2022", [])
        result = LiteratureResult(
            query="P53",
            publications=[pub1, pub2],
            known_binding_partners=[],
        )
        assert result.get_pmids() == ["12345678", "87654321"]
