"""
Fetch protein structures from RCSB PDB or AlphaFold Database.

Allows users to specify proteins by name, UniProt ID, or PDB ID
instead of providing local PDB files.
"""

import asyncio
import re
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import aiohttp

# HTTP timeouts: 30s for API calls, 60s for file downloads
_API_TIMEOUT = aiohttp.ClientTimeout(total=30)
_DOWNLOAD_TIMEOUT = aiohttp.ClientTimeout(total=60)


@dataclass
class FetchedStructure:
    """Result of fetching a protein structure."""
    pdb_path: str
    source: str  # "rcsb", "alphafold", or "local"
    pdb_id: Optional[str] = None
    uniprot_id: Optional[str] = None
    protein_name: Optional[str] = None
    resolution: Optional[float] = None


async def search_uniprot(query: str) -> Optional[dict]:
    """
    Search UniProt for a protein by name or gene.

    Returns the top match with UniProt ID and recommended name.
    """
    url = "https://rest.uniprot.org/uniprotkb/search"
    params = {
        "query": f"({query}) AND (reviewed:true)",  # Swiss-Prot only
        "format": "json",
        "size": 1,
        "fields": "accession,protein_name,gene_names,organism_name"
    }

    async with aiohttp.ClientSession(timeout=_API_TIMEOUT) as session:
        async with session.get(url, params=params) as response:
            if response.status != 200:
                return None
            data = await response.json()

            if not data.get("results"):
                return None

            result = data["results"][0]
            return {
                "uniprot_id": result.get("primaryAccession"),
                "protein_name": result.get("proteinDescription", {}).get(
                    "recommendedName", {}
                ).get("fullName", {}).get("value"),
                "gene_names": result.get("genes", [{}])[0].get("geneName", {}).get("value"),
                "organism": result.get("organism", {}).get("scientificName")
            }


async def search_pdb_by_uniprot(uniprot_id: str) -> list[dict]:
    """
    Search RCSB PDB for structures of a UniProt protein.

    Returns list of PDB entries sorted by resolution.
    """
    url = "https://search.rcsb.org/rcsbsearch/v2/query"
    query = {
        "query": {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
                "operator": "exact_match",
                "value": uniprot_id
            }
        },
        "return_type": "entry",
        "request_options": {
            "results_content_type": ["experimental"],
            "sort": [{"sort_by": "rcsb_entry_info.resolution_combined", "direction": "asc"}],
            "paginate": {"start": 0, "rows": 5}
        }
    }

    async with aiohttp.ClientSession(timeout=_API_TIMEOUT) as session:
        async with session.post(url, json=query) as response:
            if response.status != 200:
                return []
            data = await response.json()

            results = []
            for hit in data.get("result_set", []):
                pdb_id = hit.get("identifier")
                if pdb_id:
                    results.append({"pdb_id": pdb_id})

            return results


async def get_pdb_info(pdb_id: str) -> Optional[dict]:
    """Get metadata for a PDB entry."""
    url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"

    async with aiohttp.ClientSession(timeout=_API_TIMEOUT) as session:
        async with session.get(url) as response:
            if response.status != 200:
                return None
            data = await response.json()

            return {
                "pdb_id": pdb_id,
                "title": data.get("struct", {}).get("title"),
                "resolution": data.get("rcsb_entry_info", {}).get("resolution_combined", [None])[0],
                "method": data.get("exptl", [{}])[0].get("method")
            }


async def download_pdb(pdb_id: str, output_dir: Path) -> Optional[str]:
    """Download a PDB file from RCSB."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{pdb_id.lower()}.pdb"

    if output_path.exists():
        return str(output_path)

    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"

    async with aiohttp.ClientSession(timeout=_DOWNLOAD_TIMEOUT) as session:
        async with session.get(url) as response:
            if response.status != 200:
                return None

            content = await response.read()
            output_path.write_bytes(content)
            return str(output_path)


async def download_alphafold(uniprot_id: str, output_dir: Path) -> Optional[str]:
    """Download a structure from AlphaFold Database."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"AF-{uniprot_id}-F1-model_v4.pdb"

    if output_path.exists():
        return str(output_path)

    url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"

    async with aiohttp.ClientSession(timeout=_DOWNLOAD_TIMEOUT) as session:
        async with session.get(url) as response:
            if response.status != 200:
                return None

            content = await response.read()
            output_path.write_bytes(content)
            return str(output_path)


async def fetch_structure(
    query: str,
    output_dir: Optional[Path] = None,
    prefer_experimental: bool = True,
) -> FetchedStructure:
    """
    Fetch a protein structure by name, UniProt ID, PDB ID, or local path.

    Args:
        query: Protein name (e.g., "EGFR"), UniProt ID (e.g., "P00533"),
               PDB ID (e.g., "1IVO"), or local file path
        output_dir: Directory to save downloaded structures
        prefer_experimental: If True, prefer RCSB structures over AlphaFold

    Returns:
        FetchedStructure with path to PDB file and metadata

    Examples:
        >>> result = await fetch_structure("EGFR")
        >>> result = await fetch_structure("P00533")  # UniProt ID
        >>> result = await fetch_structure("1IVO")    # PDB ID
        >>> result = await fetch_structure("/path/to/protein.pdb")
    """
    if output_dir is None:
        output_dir = Path.home() / ".cache" / "protein-design-mcp" / "structures"

    output_dir = Path(output_dir)

    # Check if it's a local file path
    if Path(query).exists():
        return FetchedStructure(
            pdb_path=str(Path(query).resolve()),
            source="local"
        )

    # Check if it looks like a PDB ID (4 characters, starts with digit)
    if re.match(r"^\d[a-zA-Z0-9]{3}$", query):
        pdb_id = query.upper()
        pdb_path = await download_pdb(pdb_id, output_dir)
        if pdb_path:
            info = await get_pdb_info(pdb_id)
            return FetchedStructure(
                pdb_path=pdb_path,
                source="rcsb",
                pdb_id=pdb_id,
                resolution=info.get("resolution") if info else None
            )

    # Check if it looks like a UniProt ID
    uniprot_id = None
    if re.match(r"^[A-Z][0-9][A-Z0-9]{3}[0-9]$", query.upper()):
        uniprot_id = query.upper()
    else:
        # Search UniProt by protein name
        uniprot_result = await search_uniprot(query)
        if uniprot_result:
            uniprot_id = uniprot_result["uniprot_id"]

    if not uniprot_id:
        raise ValueError(
            f"Could not find protein: {query}. "
            "Try a UniProt ID (e.g., P00533), PDB ID (e.g., 1IVO), "
            "or protein name (e.g., EGFR human)"
        )

    # Try to find experimental structure
    if prefer_experimental:
        pdb_entries = await search_pdb_by_uniprot(uniprot_id)
        for entry in pdb_entries:
            pdb_path = await download_pdb(entry["pdb_id"], output_dir)
            if pdb_path:
                info = await get_pdb_info(entry["pdb_id"])
                return FetchedStructure(
                    pdb_path=pdb_path,
                    source="rcsb",
                    pdb_id=entry["pdb_id"],
                    uniprot_id=uniprot_id,
                    resolution=info.get("resolution") if info else None
                )

    # Fall back to AlphaFold
    pdb_path = await download_alphafold(uniprot_id, output_dir)
    if pdb_path:
        return FetchedStructure(
            pdb_path=pdb_path,
            source="alphafold",
            uniprot_id=uniprot_id
        )

    raise ValueError(
        f"Could not fetch structure for {query} (UniProt: {uniprot_id}). "
        "No experimental or AlphaFold structure available."
    )


def fetch_structure_sync(
    query: str,
    output_dir: Optional[Path] = None,
    prefer_experimental: bool = True,
) -> FetchedStructure:
    """Synchronous wrapper for fetch_structure."""
    return asyncio.run(fetch_structure(query, output_dir, prefer_experimental))
