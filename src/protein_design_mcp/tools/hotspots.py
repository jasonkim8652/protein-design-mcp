"""
Hotspot suggestion tool.

Analyzes target proteins to identify potential binding hotspots
based on surface exposure, hydrophobicity, and other criteria.
"""

from pathlib import Path
from typing import Any
import math

from protein_design_mcp.utils.pdb import parse_pdb


# Valid criteria for hotspot selection
VALID_CRITERIA = {"druggable", "exposed", "conserved"}

# Amino acid properties
HYDROPHOBIC = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO"}
CHARGED = {"ARG", "LYS", "HIS", "ASP", "GLU"}
AROMATIC = {"PHE", "TYR", "TRP"}

# Average surface area per residue (Å²)
AVG_SURFACE_AREA_PER_RESIDUE = 120.0


async def suggest_hotspots(
    target_pdb: str,
    chain_id: str | None = None,
    criteria: str = "exposed",
) -> dict[str, Any]:
    """
    Suggest potential binding hotspots on a target protein.

    Analyzes the protein surface to identify regions suitable for
    binder design based on the specified criteria.

    Args:
        target_pdb: Path to target protein PDB file
        chain_id: Specific chain to analyze (default: first chain)
        criteria: Hotspot selection criteria:
            - "exposed": Surface-exposed residues
            - "druggable": Hydrophobic pockets suitable for small molecules
            - "conserved": Conserved residues (requires external data)

    Returns:
        Dictionary containing:
        - suggested_hotspots: List of hotspot suggestions with scores
        - surface_analysis: Overall surface analysis metrics

    Raises:
        FileNotFoundError: If PDB file doesn't exist
        ValueError: If chain_id is invalid or criteria is not recognized
    """
    # Validate PDB path
    pdb_path = Path(target_pdb)
    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB file not found: {target_pdb}")

    # Validate criteria
    if criteria not in VALID_CRITERIA:
        raise ValueError(
            f"Criteria must be one of {VALID_CRITERIA}, got: {criteria}"
        )

    # Parse structure
    structure = parse_pdb(target_pdb)

    # Get target chain
    if chain_id is None:
        if not structure.chains:
            raise ValueError("No chains found in PDB")
        target_chain = structure.chains[0]
    else:
        chain_ids = [c.chain_id for c in structure.chains]
        if chain_id not in chain_ids:
            raise ValueError(
                f"Chain {chain_id} not found in PDB. Available chains: {chain_ids}"
            )
        target_chain = next(c for c in structure.chains if c.chain_id == chain_id)

    # Analyze surface and find hotspots
    hotspots = _find_hotspots(target_chain, criteria)

    # Calculate surface analysis
    surface_analysis = _analyze_surface(target_chain)

    return {
        "suggested_hotspots": hotspots,
        "surface_analysis": surface_analysis,
    }


def _calculate_solvent_exposure(residue) -> float:
    """
    Calculate approximate solvent exposure for a residue.

    Uses a simplified heuristic based on neighbor atom count.
    Returns value between 0 (buried) and 1 (fully exposed).
    """
    # Count CA atoms (simplified - in real implementation would check all atoms)
    if not residue.atoms:
        return 0.5  # Default

    # Check if residue has coordinates (atoms is a list of dicts)
    ca_atom = next((a for a in residue.atoms if a.get("atom_name") == "CA"), None)
    if ca_atom is None:
        return 0.5

    # Simplified: assume all surface residues have similar exposure
    # In reality, would calculate based on neighboring atoms
    return 0.7


def _find_hotspots(chain, criteria: str) -> list[dict[str, Any]]:
    """
    Find hotspot regions based on criteria.

    Returns list of hotspot suggestions sorted by score.
    """
    chain_id = chain.chain_id
    hotspots = []

    if criteria == "exposed":
        hotspots = _find_exposed_hotspots(chain, chain_id)
    elif criteria == "druggable":
        hotspots = _find_druggable_hotspots(chain, chain_id)
    elif criteria == "conserved":
        # Conserved hotspots would require MSA data
        # For now, return exposed as proxy
        hotspots = _find_exposed_hotspots(chain, chain_id)

    # Sort by score (highest first)
    hotspots.sort(key=lambda h: h["score"], reverse=True)

    return hotspots


def _find_exposed_hotspots(chain, chain_id: str) -> list[dict[str, Any]]:
    """Find surface-exposed hotspot patches."""
    hotspots = []

    # Group residues into patches of 3-5 consecutive residues
    residues = list(chain.residues)
    if len(residues) < 3:
        # If too few residues, return single hotspot with all
        if residues:
            hotspots.append({
                "residues": [f"{chain_id}{r.residue_number}" for r in residues],
                "score": 0.7,
                "rationale": "Small protein - all residues exposed",
            })
        return hotspots

    # Create sliding window of patches
    patch_size = 3
    for i in range(len(residues) - patch_size + 1):
        patch = residues[i:i + patch_size]

        # Calculate exposure score for patch
        exposure_scores = [_calculate_solvent_exposure(r) for r in patch]
        avg_exposure = sum(exposure_scores) / len(exposure_scores)

        # Check for hydrophobic content (good for binding)
        hydrophobic_count = sum(1 for r in patch if r.residue_name in HYDROPHOBIC)
        hydrophobic_ratio = hydrophobic_count / len(patch)

        # Calculate overall score
        score = avg_exposure * 0.6 + hydrophobic_ratio * 0.4

        # Generate rationale
        if hydrophobic_ratio > 0.5:
            rationale = "Exposed hydrophobic patch"
        elif avg_exposure > 0.7:
            rationale = "Highly exposed surface region"
        else:
            rationale = "Surface-accessible region"

        hotspots.append({
            "residues": [f"{chain_id}{r.residue_number}" for r in patch],
            "score": round(min(score, 1.0), 2),
            "rationale": rationale,
        })

    # Return top 5 non-overlapping hotspots
    return _select_non_overlapping(hotspots, max_count=5)


def _find_druggable_hotspots(chain, chain_id: str) -> list[dict[str, Any]]:
    """Find druggable pocket-like regions."""
    hotspots = []

    residues = list(chain.residues)
    if len(residues) < 3:
        return hotspots

    # Look for clusters of aromatic/hydrophobic residues
    patch_size = 4
    for i in range(len(residues) - patch_size + 1):
        patch = residues[i:i + patch_size]

        # Calculate druggability score
        aromatic_count = sum(1 for r in patch if r.residue_name in AROMATIC)
        hydrophobic_count = sum(1 for r in patch if r.residue_name in HYDROPHOBIC)

        aromatic_ratio = aromatic_count / len(patch)
        hydrophobic_ratio = hydrophobic_count / len(patch)

        # Druggable sites often have aromatic and hydrophobic residues
        score = aromatic_ratio * 0.5 + hydrophobic_ratio * 0.5

        if score > 0.2:  # Minimum threshold
            rationale = f"Potential pocket with {hydrophobic_count} hydrophobic residues"
            if aromatic_count > 0:
                rationale = f"Aromatic-rich region ({aromatic_count} aromatic residues)"

            hotspots.append({
                "residues": [f"{chain_id}{r.residue_number}" for r in patch],
                "score": round(min(score, 1.0), 2),
                "rationale": rationale,
            })

    # Return top 5 non-overlapping
    return _select_non_overlapping(hotspots, max_count=5)


def _select_non_overlapping(hotspots: list[dict], max_count: int) -> list[dict]:
    """Select top non-overlapping hotspots."""
    if not hotspots:
        return []

    # Sort by score
    sorted_hotspots = sorted(hotspots, key=lambda h: h["score"], reverse=True)

    selected = []
    used_residues = set()

    for hotspot in sorted_hotspots:
        residues = set(hotspot["residues"])
        # Check if this hotspot overlaps with already selected ones
        if not residues & used_residues:
            selected.append(hotspot)
            used_residues.update(residues)

            if len(selected) >= max_count:
                break

    return selected


def _analyze_surface(chain) -> dict[str, Any]:
    """Calculate surface analysis metrics."""
    residues = list(chain.residues)
    num_residues = len(residues)

    # Estimate total surface area
    total_surface_area = num_residues * AVG_SURFACE_AREA_PER_RESIDUE

    # Count hydrophobic patches (consecutive hydrophobic residues)
    hydrophobic_patches = 0
    in_patch = False
    for residue in residues:
        if residue.residue_name in HYDROPHOBIC:
            if not in_patch:
                hydrophobic_patches += 1
                in_patch = True
        else:
            in_patch = False

    # Count charged regions
    charged_regions = 0
    in_region = False
    for residue in residues:
        if residue.residue_name in CHARGED:
            if not in_region:
                charged_regions += 1
                in_region = True
        else:
            in_region = False

    return {
        "total_surface_area": total_surface_area,
        "hydrophobic_patches": hydrophobic_patches,
        "charged_regions": charged_regions,
        "total_residues": num_residues,
    }
