"""
PubMed literature search for protein binding information.

Searches PubMed for publications about protein-protein interactions,
binding sites, and binding partners.
"""

import aiohttp
from dataclasses import dataclass


@dataclass
class Publication:
    """Represents a publication from PubMed."""

    pmid: str
    title: str
    authors: list[str]
    year: str
    binding_residues: list[str]  # Extracted binding residue mentions


@dataclass
class LiteratureResult:
    """Container for literature search results."""

    query: str
    publications: list[Publication]
    known_binding_partners: list[str]

    def get_pmids(self) -> list[str]:
        """Get list of PMIDs from publications."""
        return [pub.pmid for pub in self.publications]


PUBMED_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"


def parse_pubmed_response(
    search_response: dict,
    summary_response: dict,
) -> LiteratureResult:
    """
    Parse PubMed API responses into LiteratureResult.

    Args:
        search_response: Response from esearch API
        summary_response: Response from esummary API

    Returns:
        LiteratureResult with parsed publications
    """
    publications = []
    known_partners = []

    id_list = search_response.get("esearchresult", {}).get("idlist", [])
    results = summary_response.get("result", {})

    for pmid in id_list:
        if pmid in results:
            pub_data = results[pmid]
            if isinstance(pub_data, dict):
                title = pub_data.get("title", "")
                authors = [a.get("name", "") for a in pub_data.get("authors", [])]
                year = pub_data.get("pubdate", "")[:4]  # First 4 chars = year

                publications.append(Publication(
                    pmid=pmid,
                    title=title,
                    authors=authors,
                    year=year,
                    binding_residues=[],  # Would require full-text analysis
                ))

                # Extract potential binding partners from title
                # Simple heuristic: look for protein names in title
                _extract_binding_partners(title, known_partners)

    return LiteratureResult(
        query="",  # Will be set by caller
        publications=publications,
        known_binding_partners=list(set(known_partners)),
    )


def _extract_binding_partners(title: str, partners: list[str]) -> None:
    """
    Extract potential binding partner names from publication title.

    Simple heuristic based on common protein naming patterns.
    """
    # Common keywords indicating binding partner mentions
    binding_keywords = ["binds", "binding", "interacts", "interaction", "complex"]

    title_lower = title.lower()
    if any(kw in title_lower for kw in binding_keywords):
        # Look for capitalized words that might be protein names
        words = title.split()
        for word in words:
            # Protein names are often capitalized, 2-10 chars, may have numbers
            clean_word = word.strip(",.;:-()[]")
            if (len(clean_word) >= 2 and
                len(clean_word) <= 15 and
                clean_word[0].isupper() and
                not clean_word.lower() in ["the", "and", "with", "from", "into"]):
                partners.append(clean_word)


async def search_binding_partners(
    protein_name: str,
    max_results: int = 20,
) -> LiteratureResult:
    """
    Search PubMed for publications about protein binding.

    Args:
        protein_name: Name or identifier of the protein
        max_results: Maximum number of publications to return

    Returns:
        LiteratureResult with relevant publications and binding partners
    """
    # Construct search query
    query = f'"{protein_name}"[Title/Abstract] AND (binding[Title/Abstract] OR interaction[Title/Abstract] OR complex[Title/Abstract])'

    # Search PubMed
    search_params = {
        "db": "pubmed",
        "term": query,
        "retmax": str(max_results),
        "retmode": "json",
    }

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
        # Search for PMIDs
        async with session.get(PUBMED_ESEARCH_URL, params=search_params) as response:
            if response.status != 200:
                return LiteratureResult(
                    query=protein_name,
                    publications=[],
                    known_binding_partners=[],
                )
            search_data = await response.json()

        # Get publication summaries
        id_list = search_data.get("esearchresult", {}).get("idlist", [])

        if not id_list:
            return LiteratureResult(
                query=protein_name,
                publications=[],
                known_binding_partners=[],
            )

        summary_params = {
            "db": "pubmed",
            "id": ",".join(id_list),
            "retmode": "json",
        }

        async with session.get(PUBMED_ESUMMARY_URL, params=summary_params) as response:
            if response.status != 200:
                return LiteratureResult(
                    query=protein_name,
                    publications=[],
                    known_binding_partners=[],
                )
            summary_data = await response.json()

    result = parse_pubmed_response(search_data, summary_data)
    result.query = protein_name

    return result
