"""
Hotspot suggestion tool.

Analyzes target proteins to identify potential binding hotspots
using multiple data sources:
- Structural analysis (SASA, pocket detection)
- UniProt annotations (binding sites, active sites)
- Conservation scoring (BLAST-based)
- Literature mining (PubMed search)
"""

from pathlib import Path
from typing import Any

from protein_design_mcp.utils.pdb import parse_pdb
from protein_design_mcp.utils.sasa import calculate_sasa, detect_pockets
from protein_design_mcp.utils.uniprot import fetch_uniprot_features
from protein_design_mcp.utils.conservation import calculate_conservation_scores
from protein_design_mcp.utils.pubmed import search_binding_partners
from protein_design_mcp.utils.fetch_structure import fetch_structure


# Valid criteria for hotspot selection
VALID_CRITERIA = {"druggable", "exposed", "conserved"}

# Amino acid properties
HYDROPHOBIC = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO"}
CHARGED = {"ARG", "LYS", "HIS", "ASP", "GLU"}
AROMATIC = {"PHE", "TYR", "TRP"}

# Average surface area per residue (Å²)
AVG_SURFACE_AREA_PER_RESIDUE = 120.0


async def suggest_hotspots(
    target: str,
    chain_id: str | None = None,
    criteria: str = "exposed",
    uniprot_id: str | None = None,
    include_literature: bool = False,
) -> dict[str, Any]:
    """
    Suggest potential binding hotspots on a target protein.

    Analyzes the protein surface using multiple data sources to identify
    regions suitable for binder design.

    Args:
        target: Target protein - can be:
            - Protein name (e.g., "EGFR", "insulin receptor")
            - UniProt ID (e.g., "P00533")
            - PDB ID (e.g., "1IVO")
            - Path to local PDB file
        chain_id: Specific chain to analyze (default: first chain)
        criteria: Hotspot selection criteria:
            - "exposed": Surface-exposed residues
            - "druggable": Hydrophobic pockets suitable for small molecules
            - "conserved": Conserved residues
        uniprot_id: UniProt ID for retrieving known binding sites (auto-detected if not provided)
        include_literature: Search PubMed for known binding partners

    Returns:
        Dictionary containing:
        - suggested_hotspots: List of hotspot suggestions with evidence
        - surface_analysis: Overall surface analysis metrics
        - conservation_profile: Per-residue conservation (if available)
        - uniprot_features: Binding sites from UniProt (if uniprot_id provided)
        - literature_insights: PubMed search results (if include_literature)
        - structure_source: Where the structure came from (local/rcsb/alphafold)

    Raises:
        FileNotFoundError: If PDB file doesn't exist
        ValueError: If chain_id is invalid or criteria is not recognized
    """
    # Fetch structure if needed (supports protein name, UniProt ID, PDB ID, or path)
    fetched = await fetch_structure(target)
    pdb_path = Path(fetched.pdb_path)

    # Auto-detect UniProt ID if not provided
    if uniprot_id is None and fetched.uniprot_id:
        uniprot_id = fetched.uniprot_id

    # Validate criteria
    if criteria not in VALID_CRITERIA:
        raise ValueError(
            f"Criteria must be one of {VALID_CRITERIA}, got: {criteria}"
        )

    # Parse structure
    structure = parse_pdb(str(pdb_path))

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

    # Get chain sequence for conservation analysis
    chain_sequence = "".join(
        _three_to_one(r.residue_name) for r in target_chain.residues
    )

    # Gather data from multiple sources
    sasa_result = None
    pockets = []
    uniprot_features = None
    conservation_profile = None
    literature_result = None

    # 1. Calculate SASA
    try:
        sasa_result = calculate_sasa(str(pdb_path))
    except Exception:
        sasa_result = None

    # 2. Detect pockets
    try:
        pockets = detect_pockets(str(pdb_path))
    except Exception:
        pockets = []

    # 3. Fetch UniProt features if ID provided
    if uniprot_id:
        try:
            uniprot_features = await fetch_uniprot_features(uniprot_id)
        except Exception:
            uniprot_features = None

    # 4. Calculate conservation scores
    if criteria == "conserved" and chain_sequence:
        try:
            conservation_profile = await calculate_conservation_scores(chain_sequence)
        except Exception:
            conservation_profile = None

    # 5. Search PubMed if requested
    if include_literature:
        try:
            protein_name = uniprot_id or pdb_path.stem
            literature_result = await search_binding_partners(protein_name)
        except Exception:
            literature_result = None

    # Analyze surface and find hotspots using all data sources
    hotspots = _find_hotspots_enhanced(
        target_chain,
        criteria,
        sasa_result=sasa_result,
        pockets=pockets,
        uniprot_features=uniprot_features,
        conservation_profile=conservation_profile,
    )

    # Calculate surface analysis
    surface_analysis = _analyze_surface_enhanced(
        target_chain,
        sasa_result=sasa_result,
        pockets=pockets,
    )

    # Build result
    result = {
        "suggested_hotspots": hotspots,
        "surface_analysis": surface_analysis,
        "structure_source": {
            "source": fetched.source,
            "pdb_path": fetched.pdb_path,
            "pdb_id": fetched.pdb_id,
            "uniprot_id": fetched.uniprot_id,
            "resolution": fetched.resolution,
        },
    }

    # Add conservation profile if available
    if conservation_profile:
        result["conservation_profile"] = {
            "average_conservation": conservation_profile.average_conservation,
            "highly_conserved_residues": [
                f"{target_chain.chain_id}{pos}" for pos in conservation_profile.highly_conserved
            ],
        }

    # Add UniProt features if available
    if uniprot_features:
        result["uniprot_features"] = {
            "binding_sites": [
                {"position": f"{bs.start}-{bs.end}", "ligand": bs.ligand}
                for bs in uniprot_features.binding_sites
            ],
            "active_sites": [
                {"position": str(ac.position), "type": ac.description}
                for ac in uniprot_features.active_sites
            ],
            "known_interactors": uniprot_features.known_interactors,
        }

    # Add literature insights if available
    if literature_result:
        result["literature_insights"] = {
            "known_binding_partners": literature_result.known_binding_partners,
            "relevant_publications": [
                {
                    "pmid": pub.pmid,
                    "title": pub.title,
                    "binding_residues": pub.binding_residues,
                }
                for pub in literature_result.publications[:5]  # Top 5
            ],
        }

    return result


def _three_to_one(three_letter: str) -> str:
    """Convert three-letter amino acid code to one-letter."""
    mapping = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    }
    return mapping.get(three_letter, "X")


def _find_hotspots_enhanced(
    chain,
    criteria: str,
    sasa_result=None,
    pockets=None,
    uniprot_features=None,
    conservation_profile=None,
) -> list[dict[str, Any]]:
    """
    Find hotspot regions using multiple data sources.

    Returns list of hotspot suggestions with evidence from each source.
    """
    chain_id = chain.chain_id
    hotspots = []

    # Get residues and create lookup
    residues = list(chain.residues)
    if not residues:
        return []

    res_id_to_idx = {
        f"{chain_id}{r.residue_number}": i for i, r in enumerate(residues)
    }

    # 1. Add hotspots from UniProt binding sites
    if uniprot_features:
        for bs in uniprot_features.binding_sites:
            hotspot_residues = [
                f"{chain_id}{pos}" for pos in bs.residue_range()
                if f"{chain_id}{pos}" in res_id_to_idx
            ]
            if hotspot_residues:
                evidence = _build_evidence(
                    hotspot_residues, sasa_result, conservation_profile, chain_id
                )
                evidence["uniprot_annotation"] = f"Binding site for {bs.ligand}"
                hotspots.append({
                    "residues": hotspot_residues,
                    "score": 0.95,  # High score for known sites
                    "rationale": f"Known binding site from UniProt ({bs.ligand})",
                    "evidence": evidence,
                })

        for ac in uniprot_features.active_sites:
            res_id = f"{chain_id}{ac.position}"
            if res_id in res_id_to_idx:
                evidence = _build_evidence(
                    [res_id], sasa_result, conservation_profile, chain_id
                )
                evidence["uniprot_annotation"] = ac.description
                hotspots.append({
                    "residues": [res_id],
                    "score": 0.90,
                    "rationale": f"Active site from UniProt ({ac.description})",
                    "evidence": evidence,
                })

    # 2. Add hotspots from detected pockets
    if pockets:
        for pocket in pockets[:3]:  # Top 3 pockets
            pocket_residues = [r for r in pocket.residues if r in res_id_to_idx]
            if pocket_residues:
                evidence = _build_evidence(
                    pocket_residues, sasa_result, conservation_profile, chain_id
                )
                evidence["structural"] = {
                    "pocket_score": pocket.druggability,
                    "volume": pocket.volume,
                }
                hotspots.append({
                    "residues": pocket_residues,
                    "score": round(pocket.druggability * 0.9, 2),
                    "rationale": f"Detected pocket (volume: {pocket.volume:.1f} Å³)",
                    "evidence": evidence,
                })

    # 3. Add criteria-based hotspots
    criteria_hotspots = _find_criteria_hotspots(
        chain, chain_id, criteria, sasa_result, conservation_profile
    )
    hotspots.extend(criteria_hotspots)

    # Sort by score and select non-overlapping
    hotspots.sort(key=lambda h: h["score"], reverse=True)
    return _select_non_overlapping(hotspots, max_count=5)


def _build_evidence(
    residue_ids: list[str],
    sasa_result,
    conservation_profile,
    chain_id: str,
) -> dict[str, Any]:
    """Build evidence dict for a hotspot from available data sources."""
    evidence = {}

    # SASA evidence
    if sasa_result:
        sasa_values = [sasa_result.per_residue.get(r, 0) for r in residue_ids]
        if sasa_values:
            evidence["structural"] = {
                "sasa": round(sum(sasa_values) / len(sasa_values), 1),
            }

    # Conservation evidence
    if conservation_profile:
        conservation_scores = []
        for res_id in residue_ids:
            # Extract position number from res_id (e.g., "A45" -> 45)
            try:
                pos = int(res_id.replace(chain_id, ""))
                if 1 <= pos <= len(conservation_profile.scores):
                    conservation_scores.append(conservation_profile.scores[pos - 1])
            except ValueError:
                pass
        if conservation_scores:
            evidence["conservation"] = round(
                sum(conservation_scores) / len(conservation_scores), 2
            )

    return evidence


def _find_criteria_hotspots(
    chain,
    chain_id: str,
    criteria: str,
    sasa_result,
    conservation_profile,
) -> list[dict[str, Any]]:
    """Find hotspots based on selection criteria."""
    hotspots = []
    residues = list(chain.residues)

    if len(residues) < 3:
        if residues:
            evidence = _build_evidence(
                [f"{chain_id}{r.residue_number}" for r in residues],
                sasa_result, conservation_profile, chain_id
            )
            hotspots.append({
                "residues": [f"{chain_id}{r.residue_number}" for r in residues],
                "score": 0.7,
                "rationale": "Small protein - all residues exposed",
                "evidence": evidence,
            })
        return hotspots

    patch_size = 3
    for i in range(len(residues) - patch_size + 1):
        patch = residues[i:i + patch_size]
        patch_ids = [f"{chain_id}{r.residue_number}" for r in patch]

        # Calculate score based on criteria
        if criteria == "exposed":
            score = _calculate_exposed_score(patch, sasa_result, patch_ids)
            rationale = "Exposed surface region"
        elif criteria == "druggable":
            score = _calculate_druggable_score(patch)
            rationale = "Druggable region"
        elif criteria == "conserved":
            score = _calculate_conserved_score(patch_ids, conservation_profile, chain_id)
            rationale = "Conserved region"
        else:
            score = 0.5
            rationale = "Default"

        if score > 0.3:  # Minimum threshold
            evidence = _build_evidence(
                patch_ids, sasa_result, conservation_profile, chain_id
            )
            hotspots.append({
                "residues": patch_ids,
                "score": round(min(score, 1.0), 2),
                "rationale": rationale,
                "evidence": evidence,
            })

    return hotspots


def _calculate_exposed_score(patch, sasa_result, patch_ids) -> float:
    """Calculate exposure score for a patch."""
    # Use SASA if available
    if sasa_result:
        sasa_values = [sasa_result.per_residue.get(r, 0) for r in patch_ids]
        avg_sasa = sum(sasa_values) / len(sasa_values) if sasa_values else 0
        sasa_score = min(avg_sasa / 100.0, 1.0)  # Normalize
    else:
        sasa_score = 0.7

    # Hydrophobic content
    hydrophobic_count = sum(1 for r in patch if r.residue_name in HYDROPHOBIC)
    hydrophobic_ratio = hydrophobic_count / len(patch)

    return sasa_score * 0.6 + hydrophobic_ratio * 0.4


def _calculate_druggable_score(patch) -> float:
    """Calculate druggability score for a patch."""
    aromatic_count = sum(1 for r in patch if r.residue_name in AROMATIC)
    hydrophobic_count = sum(1 for r in patch if r.residue_name in HYDROPHOBIC)

    aromatic_ratio = aromatic_count / len(patch)
    hydrophobic_ratio = hydrophobic_count / len(patch)

    return aromatic_ratio * 0.5 + hydrophobic_ratio * 0.5


def _calculate_conserved_score(patch_ids, conservation_profile, chain_id) -> float:
    """Calculate conservation score for a patch."""
    if not conservation_profile:
        return 0.5

    scores = []
    for res_id in patch_ids:
        try:
            pos = int(res_id.replace(chain_id, ""))
            if 1 <= pos <= len(conservation_profile.scores):
                scores.append(conservation_profile.scores[pos - 1])
        except ValueError:
            pass

    return sum(scores) / len(scores) if scores else 0.5


def _select_non_overlapping(hotspots: list[dict], max_count: int) -> list[dict]:
    """Select top non-overlapping hotspots."""
    if not hotspots:
        return []

    sorted_hotspots = sorted(hotspots, key=lambda h: h["score"], reverse=True)

    selected = []
    used_residues = set()

    for hotspot in sorted_hotspots:
        residues = set(hotspot["residues"])
        if not residues & used_residues:
            selected.append(hotspot)
            used_residues.update(residues)

            if len(selected) >= max_count:
                break

    return selected


def _analyze_surface_enhanced(
    chain,
    sasa_result=None,
    pockets=None,
) -> dict[str, Any]:
    """Calculate surface analysis metrics with enhanced data."""
    residues = list(chain.residues)
    num_residues = len(residues)

    # Use actual SASA if available
    if sasa_result:
        total_surface_area = sasa_result.total_sasa
    else:
        total_surface_area = num_residues * AVG_SURFACE_AREA_PER_RESIDUE

    # Count hydrophobic patches
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

    result = {
        "total_surface_area": total_surface_area,
        "hydrophobic_patches": hydrophobic_patches,
        "charged_regions": charged_regions,
        "total_residues": num_residues,
    }

    # Add detected pockets
    if pockets:
        result["detected_pockets"] = [
            {
                "center": list(p.center),
                "volume": p.volume,
                "druggability": p.druggability,
            }
            for p in pockets[:5]  # Top 5 pockets
        ]

    return result
