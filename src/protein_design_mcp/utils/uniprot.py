"""
UniProt API integration.

Fetches protein annotations including binding sites, active sites,
and known protein interactors from the UniProt database.
"""

import aiohttp
from dataclasses import dataclass


@dataclass
class BindingSite:
    """Represents a binding site annotation from UniProt."""

    start: int
    end: int
    ligand: str

    def residue_range(self) -> list[int]:
        """Return list of residue positions in this binding site."""
        return list(range(self.start, self.end + 1))


@dataclass
class ActiveSite:
    """Represents an active site annotation from UniProt."""

    position: int
    description: str


@dataclass
class UniProtFeatures:
    """Container for UniProt feature annotations."""

    uniprot_id: str
    sequence: str
    binding_sites: list[BindingSite]
    active_sites: list[ActiveSite]
    known_interactors: list[str]


UNIPROT_API_BASE = "https://rest.uniprot.org/uniprotkb"


def parse_uniprot_response(uniprot_id: str, data: dict) -> UniProtFeatures:
    """
    Parse UniProt API response into UniProtFeatures.

    Args:
        uniprot_id: UniProt accession ID
        data: JSON response from UniProt API

    Returns:
        UniProtFeatures containing parsed annotations
    """
    # Extract sequence
    sequence = data.get("sequence", {}).get("value", "")

    # Extract binding sites and active sites from features
    binding_sites = []
    active_sites = []

    for feature in data.get("features", []):
        feature_type = feature.get("type", "")
        location = feature.get("location", {})
        start = location.get("start", {}).get("value")
        end = location.get("end", {}).get("value")

        if feature_type == "Binding site":
            ligand_info = feature.get("ligand", {})
            ligand = ligand_info.get("name", "Unknown")
            if start is not None and end is not None:
                binding_sites.append(BindingSite(start=start, end=end, ligand=ligand))

        elif feature_type == "Active site":
            description = feature.get("description", "")
            if start is not None:
                active_sites.append(ActiveSite(position=start, description=description))

    # Extract known interactors from comments
    known_interactors = []
    for comment in data.get("comments", []):
        if comment.get("commentType") == "INTERACTION":
            for interaction in comment.get("interactions", []):
                interactor_two = interaction.get("interactantTwo", {})
                interactor_id = interactor_two.get("uniProtKBAccession")
                if interactor_id and interactor_id != uniprot_id:
                    known_interactors.append(interactor_id)

    return UniProtFeatures(
        uniprot_id=uniprot_id,
        sequence=sequence,
        binding_sites=binding_sites,
        active_sites=active_sites,
        known_interactors=known_interactors,
    )


async def fetch_uniprot_features(uniprot_id: str) -> UniProtFeatures:
    """
    Fetch protein features from UniProt API.

    Args:
        uniprot_id: UniProt accession ID (e.g., "P12345")

    Returns:
        UniProtFeatures containing binding sites, active sites, and interactors

    Raises:
        ValueError: If UniProt ID is not found or invalid
    """
    url = f"{UNIPROT_API_BASE}/{uniprot_id}"

    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers={"Accept": "application/json"}) as response:
            if response.status == 404:
                raise ValueError(f"UniProt ID not found: {uniprot_id}")
            if response.status != 200:
                raise ValueError(f"UniProt API error: {response.status}")

            data = await response.json()

    return parse_uniprot_response(uniprot_id, data)
