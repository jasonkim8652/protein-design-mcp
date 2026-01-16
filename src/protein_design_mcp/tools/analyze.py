"""
Interface analysis tool.

Analyzes protein-protein interfaces to calculate:
- Buried surface area
- Hydrogen bonds
- Salt bridges
- Hydrophobic contacts
"""

from pathlib import Path
from typing import Any

from protein_design_mcp.utils.pdb import parse_pdb, get_interface_residues


# Amino acid properties
HYDROPHOBIC = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO"}
CHARGED_POS = {"ARG", "LYS", "HIS"}
CHARGED_NEG = {"ASP", "GLU"}
POLAR = {"SER", "THR", "ASN", "GLN", "TYR", "CYS"}


async def analyze_interface(
    complex_pdb: str,
    chain_a: str,
    chain_b: str,
    distance_cutoff: float = 8.0,
) -> dict[str, Any]:
    """
    Analyze protein-protein interface properties.

    Args:
        complex_pdb: Path to protein complex PDB file
        chain_a: Chain ID of first protein
        chain_b: Chain ID of second protein
        distance_cutoff: Distance cutoff for interface definition (Angstroms)

    Returns:
        Dictionary containing:
        - interface_residues: Residues at the interface for each chain
        - buried_surface_area: Estimated buried surface area
        - hydrogen_bonds: Count of hydrogen bond donor/acceptor pairs
        - salt_bridges: Count of opposite charge pairs
        - hydrophobic_contacts: Count of hydrophobic-hydrophobic contacts

    Raises:
        FileNotFoundError: If PDB file doesn't exist
        ValueError: If chain IDs are invalid
    """
    # Validate PDB path
    pdb_path = Path(complex_pdb)
    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB file not found: {complex_pdb}")

    # Parse structure
    structure = parse_pdb(complex_pdb)

    # Validate chains exist
    chain_ids = [c.chain_id for c in structure.chains]
    if chain_a not in chain_ids:
        raise ValueError(f"Chain {chain_a} not found in PDB. Available chains: {chain_ids}")
    if chain_b not in chain_ids:
        raise ValueError(f"Chain {chain_b} not found in PDB. Available chains: {chain_ids}")

    # Get interface residues
    residues_a, residues_b = get_interface_residues(
        complex_pdb, chain_a, chain_b, distance_cutoff
    )

    # Get chain objects
    chain_a_obj = next(c for c in structure.chains if c.chain_id == chain_a)
    chain_b_obj = next(c for c in structure.chains if c.chain_id == chain_b)

    # Calculate interface metrics
    hydrogen_bonds = _count_hydrogen_bonds(chain_a_obj, chain_b_obj, residues_a, residues_b)
    salt_bridges = _count_salt_bridges(chain_a_obj, chain_b_obj, residues_a, residues_b)
    hydrophobic_contacts = _count_hydrophobic_contacts(chain_a_obj, chain_b_obj, residues_a, residues_b)

    # Estimate buried surface area (simplified - ~100 Å² per interface residue)
    buried_surface_area = (len(residues_a) + len(residues_b)) * 100.0

    return {
        "interface_residues": {
            "chain_a": residues_a,
            "chain_b": residues_b,
        },
        "buried_surface_area": buried_surface_area,
        "hydrogen_bonds": hydrogen_bonds,
        "salt_bridges": salt_bridges,
        "hydrophobic_contacts": hydrophobic_contacts,
    }


def _get_residue_type(chain, res_id: str) -> str | None:
    """Get residue type (3-letter code) for a residue ID."""
    # res_id format is like "3" (residue number)
    try:
        res_num = int(res_id)
        for residue in chain.residues:
            if residue.residue_number == res_num:
                return residue.residue_name
    except (ValueError, AttributeError):
        pass
    return None


def _count_hydrogen_bonds(chain_a, chain_b, residues_a: list[str], residues_b: list[str]) -> int:
    """
    Count potential hydrogen bonds at interface.

    Simplified: count polar residue pairs at interface.
    """
    hbond_capable = POLAR | CHARGED_POS | CHARGED_NEG | {"SER", "THR", "ASN", "GLN", "TYR"}

    count = 0
    for res_a in residues_a:
        type_a = _get_residue_type(chain_a, res_a)
        if type_a and type_a in hbond_capable:
            for res_b in residues_b:
                type_b = _get_residue_type(chain_b, res_b)
                if type_b and type_b in hbond_capable:
                    count += 1
    return count


def _count_salt_bridges(chain_a, chain_b, residues_a: list[str], residues_b: list[str]) -> int:
    """
    Count potential salt bridges at interface.

    Salt bridges form between oppositely charged residues.
    """
    count = 0

    # Check chain A positive with chain B negative
    for res_a in residues_a:
        type_a = _get_residue_type(chain_a, res_a)
        if type_a in CHARGED_POS:
            for res_b in residues_b:
                type_b = _get_residue_type(chain_b, res_b)
                if type_b in CHARGED_NEG:
                    count += 1

    # Check chain A negative with chain B positive
    for res_a in residues_a:
        type_a = _get_residue_type(chain_a, res_a)
        if type_a in CHARGED_NEG:
            for res_b in residues_b:
                type_b = _get_residue_type(chain_b, res_b)
                if type_b in CHARGED_POS:
                    count += 1

    return count


def _count_hydrophobic_contacts(chain_a, chain_b, residues_a: list[str], residues_b: list[str]) -> int:
    """
    Count hydrophobic contacts at interface.

    Hydrophobic contacts form between hydrophobic residues.
    """
    count = 0
    for res_a in residues_a:
        type_a = _get_residue_type(chain_a, res_a)
        if type_a and type_a in HYDROPHOBIC:
            for res_b in residues_b:
                type_b = _get_residue_type(chain_b, res_b)
                if type_b and type_b in HYDROPHOBIC:
                    count += 1
    return count
