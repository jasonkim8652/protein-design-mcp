"""
Quality metrics calculation.

Functions for calculating protein design quality metrics including:
- pLDDT (predicted local distance difference test)
- pTM (predicted template modeling score)
- Interface metrics
"""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class DesignMetrics:
    """Quality metrics for a protein design."""

    plddt: float  # Mean pLDDT score (0-100)
    ptm: float  # Predicted TM score (0-1)
    interface_plddt: float | None = None  # Mean pLDDT at interface
    binding_energy: float | None = None  # Predicted binding energy (kcal/mol)
    rmsd_to_expected: float | None = None  # RMSD to expected structure


@dataclass
class InterfaceMetrics:
    """Metrics for a protein-protein interface."""

    buried_surface_area: float  # Angstroms^2
    hydrogen_bonds: int
    salt_bridges: int
    hydrophobic_contacts: int
    shape_complementarity: float  # 0-1


def calculate_metrics(
    plddt_per_residue: np.ndarray,
    ptm: float,
    interface_residues: list[int] | None = None,
) -> DesignMetrics:
    """
    Calculate design quality metrics.

    Args:
        plddt_per_residue: Array of per-residue pLDDT scores
        ptm: Predicted TM score
        interface_residues: Optional list of interface residue indices

    Returns:
        DesignMetrics object
    """
    mean_plddt = float(np.mean(plddt_per_residue))

    interface_plddt = None
    if interface_residues:
        interface_plddt = float(np.mean(plddt_per_residue[interface_residues]))

    return DesignMetrics(
        plddt=mean_plddt,
        ptm=ptm,
        interface_plddt=interface_plddt,
    )


def calculate_interface_metrics(
    complex_pdb: str,
    chain_a: str,
    chain_b: str,
) -> InterfaceMetrics:
    """
    Calculate interface metrics between two chains.

    Args:
        complex_pdb: Path to complex PDB file
        chain_a: First chain ID
        chain_b: Second chain ID

    Returns:
        InterfaceMetrics object
    """
    # TODO: Implement interface metrics calculation
    # 1. Calculate buried surface area
    # 2. Find hydrogen bonds
    # 3. Find salt bridges
    # 4. Count hydrophobic contacts
    # 5. Calculate shape complementarity

    raise NotImplementedError("calculate_interface_metrics not yet implemented")


def filter_designs(
    designs: list[dict[str, Any]],
    min_plddt: float = 70.0,
    min_ptm: float = 0.5,
    min_interface_plddt: float | None = None,
) -> list[dict[str, Any]]:
    """
    Filter designs based on quality thresholds.

    Args:
        designs: List of design dictionaries with metrics
        min_plddt: Minimum mean pLDDT score
        min_ptm: Minimum pTM score
        min_interface_plddt: Optional minimum interface pLDDT

    Returns:
        Filtered list of designs
    """
    filtered = []
    for design in designs:
        metrics = design.get("metrics", {})

        if metrics.get("plddt", 0) < min_plddt:
            continue
        if metrics.get("ptm", 0) < min_ptm:
            continue
        if min_interface_plddt and metrics.get("interface_plddt", 0) < min_interface_plddt:
            continue

        filtered.append(design)

    return filtered


def rank_designs(
    designs: list[dict[str, Any]],
    weights: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """
    Rank designs by quality score.

    Args:
        designs: List of design dictionaries with metrics
        weights: Optional custom weights for scoring

    Returns:
        Sorted list of designs (best first)
    """
    if weights is None:
        weights = {
            "plddt": 0.3,
            "ptm": 0.3,
            "interface_plddt": 0.4,
        }

    def score(design: dict[str, Any]) -> float:
        metrics = design.get("metrics", {})
        total = 0.0
        for key, weight in weights.items():
            if key in metrics:
                value = metrics[key]
                # Normalize pLDDT to 0-1 range
                if key == "plddt" or key == "interface_plddt":
                    value = value / 100.0
                total += value * weight
        return total

    return sorted(designs, key=score, reverse=True)
