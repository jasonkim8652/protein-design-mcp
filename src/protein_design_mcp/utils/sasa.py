"""
Solvent Accessible Surface Area (SASA) calculation and pocket detection.

Calculates SASA using the Shrake-Rupley algorithm and detects
potential binding pockets based on surface geometry.
"""

import math
from dataclasses import dataclass
from pathlib import Path

from protein_design_mcp.utils.pdb import parse_pdb


@dataclass
class SASAResult:
    """Container for SASA calculation results."""

    total_sasa: float
    per_residue: dict[str, float]  # Residue ID -> SASA
    exposed_residues: list[str]  # Residues with high SASA

    def get_residue_sasa(self, residue_id: str) -> float:
        """Get SASA for a specific residue."""
        return self.per_residue.get(residue_id, 0.0)


@dataclass
class Pocket:
    """Represents a detected binding pocket."""

    center: tuple[float, float, float]
    volume: float
    residues: list[str]
    druggability: float  # 0-1 score


# Van der Waals radii for common atoms (Angstroms)
VDW_RADII = {
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "S": 1.80,
    "H": 1.20,
    "P": 1.80,
}

# Probe radius for solvent (water)
PROBE_RADIUS = 1.4

# SASA thresholds
EXPOSED_THRESHOLD = 30.0  # Å² - residue is considered exposed if SASA > this

# Expected SASA for fully exposed residues (Å²)
MAX_SASA_PER_RESIDUE = {
    "ALA": 113, "ARG": 241, "ASN": 158, "ASP": 151, "CYS": 140,
    "GLN": 189, "GLU": 183, "GLY": 85, "HIS": 194, "ILE": 182,
    "LEU": 180, "LYS": 211, "MET": 204, "PHE": 218, "PRO": 143,
    "SER": 122, "THR": 146, "TRP": 259, "TYR": 229, "VAL": 160,
}


def _get_atom_radius(atom_name: str) -> float:
    """Get van der Waals radius for an atom."""
    # Get element from atom name (first letter, or first two for special cases)
    element = atom_name[0] if atom_name else "C"
    return VDW_RADII.get(element, 1.70)


def _calculate_atom_sasa(
    atom_coords: tuple[float, float, float],
    atom_radius: float,
    neighbor_coords: list[tuple[float, float, float]],
    neighbor_radii: list[float],
    n_points: int = 92,
) -> float:
    """
    Calculate SASA for a single atom using Shrake-Rupley algorithm.

    Args:
        atom_coords: (x, y, z) coordinates of the atom
        atom_radius: Van der Waals radius of the atom
        neighbor_coords: Coordinates of neighboring atoms
        neighbor_radii: Radii of neighboring atoms
        n_points: Number of test points on sphere

    Returns:
        SASA for this atom in Å²
    """
    ax, ay, az = atom_coords
    total_radius = atom_radius + PROBE_RADIUS

    # Generate points on a sphere (Fibonacci lattice)
    golden_ratio = (1 + math.sqrt(5)) / 2
    accessible_points = 0

    for i in range(n_points):
        theta = 2 * math.pi * i / golden_ratio
        phi = math.acos(1 - 2 * (i + 0.5) / n_points)

        # Point on sphere surface
        px = ax + total_radius * math.sin(phi) * math.cos(theta)
        py = ay + total_radius * math.sin(phi) * math.sin(theta)
        pz = az + total_radius * math.cos(phi)

        # Check if point is accessible (not inside any neighbor sphere)
        accessible = True
        for (nx, ny, nz), nr in zip(neighbor_coords, neighbor_radii):
            neighbor_total = nr + PROBE_RADIUS
            dist_sq = (px - nx)**2 + (py - ny)**2 + (pz - nz)**2
            if dist_sq < neighbor_total**2:
                accessible = False
                break

        if accessible:
            accessible_points += 1

    # Calculate SASA
    sphere_area = 4 * math.pi * total_radius**2
    return sphere_area * (accessible_points / n_points)


def calculate_sasa(pdb_path: str) -> SASAResult:
    """
    Calculate Solvent Accessible Surface Area for a protein.

    Uses the Shrake-Rupley algorithm with a probe radius of 1.4 Å.

    Args:
        pdb_path: Path to PDB file

    Returns:
        SASAResult with total and per-residue SASA values

    Raises:
        FileNotFoundError: If PDB file doesn't exist
    """
    path = Path(pdb_path)
    if not path.exists():
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")

    structure = parse_pdb(pdb_path)

    # Collect all atoms with coordinates
    all_atoms = []
    atom_to_residue = {}

    for chain in structure.chains:
        for residue in chain.residues:
            res_id = f"{chain.chain_id}{residue.residue_number}"
            for atom in residue.atoms:
                if "x" in atom and "y" in atom and "z" in atom:
                    coords = (atom["x"], atom["y"], atom["z"])
                    atom_name = atom.get("atom_name", "C")
                    radius = _get_atom_radius(atom_name)
                    all_atoms.append((coords, radius, res_id, atom_name))
                    atom_to_residue[len(all_atoms) - 1] = res_id

    # Calculate per-residue SASA
    per_residue_sasa = {}

    for i, (coords, radius, res_id, atom_name) in enumerate(all_atoms):
        # Get neighbors within cutoff distance
        neighbor_coords = []
        neighbor_radii = []
        cutoff = 10.0  # Å

        for j, (other_coords, other_radius, _, _) in enumerate(all_atoms):
            if i != j:
                dx = coords[0] - other_coords[0]
                dy = coords[1] - other_coords[1]
                dz = coords[2] - other_coords[2]
                dist_sq = dx*dx + dy*dy + dz*dz
                if dist_sq < cutoff**2:
                    neighbor_coords.append(other_coords)
                    neighbor_radii.append(other_radius)

        # Calculate atom SASA
        atom_sasa = _calculate_atom_sasa(
            coords, radius, neighbor_coords, neighbor_radii, n_points=50
        )

        # Add to residue total
        if res_id not in per_residue_sasa:
            per_residue_sasa[res_id] = 0.0
        per_residue_sasa[res_id] += atom_sasa

    # Identify exposed residues
    exposed_residues = [
        res_id for res_id, sasa in per_residue_sasa.items()
        if sasa > EXPOSED_THRESHOLD
    ]

    # Calculate total SASA
    total_sasa = sum(per_residue_sasa.values())

    return SASAResult(
        total_sasa=total_sasa,
        per_residue=per_residue_sasa,
        exposed_residues=exposed_residues,
    )


def detect_pockets(pdb_path: str, min_volume: float = 100.0) -> list[Pocket]:
    """
    Detect potential binding pockets in a protein structure.

    Uses a simple grid-based approach to identify concave surface regions.

    Args:
        pdb_path: Path to PDB file
        min_volume: Minimum pocket volume to report (Å³)

    Returns:
        List of detected Pocket objects

    Raises:
        FileNotFoundError: If PDB file doesn't exist
    """
    path = Path(pdb_path)
    if not path.exists():
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")

    structure = parse_pdb(pdb_path)

    # Collect atom coordinates and residue info
    atom_coords = []
    residue_coords = {}  # residue_id -> list of coords

    for chain in structure.chains:
        for residue in chain.residues:
            res_id = f"{chain.chain_id}{residue.residue_number}"
            residue_coords[res_id] = []
            for atom in residue.atoms:
                if "x" in atom and "y" in atom and "z" in atom:
                    coords = (atom["x"], atom["y"], atom["z"])
                    atom_coords.append(coords)
                    residue_coords[res_id].append(coords)

    if not atom_coords:
        return []

    # Simple pocket detection based on surface concavity
    # Find regions where atoms form a "bowl" shape
    pockets = []

    # Calculate center of mass
    cx = sum(c[0] for c in atom_coords) / len(atom_coords)
    cy = sum(c[1] for c in atom_coords) / len(atom_coords)
    cz = sum(c[2] for c in atom_coords) / len(atom_coords)

    # Group residues by spatial proximity to find pockets
    # Use a simple grid-based clustering
    grid_size = 8.0  # Å

    grid_cells = {}
    for res_id, coords_list in residue_coords.items():
        if not coords_list:
            continue
        # Use centroid of residue
        rx = sum(c[0] for c in coords_list) / len(coords_list)
        ry = sum(c[1] for c in coords_list) / len(coords_list)
        rz = sum(c[2] for c in coords_list) / len(coords_list)

        cell_key = (int(rx / grid_size), int(ry / grid_size), int(rz / grid_size))
        if cell_key not in grid_cells:
            grid_cells[cell_key] = []
        grid_cells[cell_key].append((res_id, (rx, ry, rz)))

    # Find cells with multiple residues (potential pocket)
    for cell_key, residues_in_cell in grid_cells.items():
        if len(residues_in_cell) >= 3:
            # Calculate pocket center
            pocket_cx = sum(r[1][0] for r in residues_in_cell) / len(residues_in_cell)
            pocket_cy = sum(r[1][1] for r in residues_in_cell) / len(residues_in_cell)
            pocket_cz = sum(r[1][2] for r in residues_in_cell) / len(residues_in_cell)

            # Estimate volume based on residue spread
            max_dist = 0
            for r in residues_in_cell:
                dist = math.sqrt(
                    (r[1][0] - pocket_cx)**2 +
                    (r[1][1] - pocket_cy)**2 +
                    (r[1][2] - pocket_cz)**2
                )
                max_dist = max(max_dist, dist)

            volume = (4/3) * math.pi * max_dist**3 if max_dist > 0 else 0

            if volume >= min_volume:
                # Calculate druggability based on residue composition
                pocket_residues = [r[0] for r in residues_in_cell]
                druggability = _calculate_druggability(structure, pocket_residues)

                pockets.append(Pocket(
                    center=(pocket_cx, pocket_cy, pocket_cz),
                    volume=volume,
                    residues=pocket_residues,
                    druggability=druggability,
                ))

    # Sort by druggability
    pockets.sort(key=lambda p: p.druggability, reverse=True)

    return pockets


def _calculate_druggability(structure, residue_ids: list[str]) -> float:
    """
    Calculate druggability score for a pocket.

    Based on composition of hydrophobic and aromatic residues.
    """
    hydrophobic = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO"}
    aromatic = {"PHE", "TYR", "TRP"}

    hydrophobic_count = 0
    aromatic_count = 0
    total = 0

    for chain in structure.chains:
        for residue in chain.residues:
            res_id = f"{chain.chain_id}{residue.residue_number}"
            if res_id in residue_ids:
                total += 1
                if residue.residue_name in hydrophobic:
                    hydrophobic_count += 1
                if residue.residue_name in aromatic:
                    aromatic_count += 1

    if total == 0:
        return 0.0

    # Druggability = weighted combination of hydrophobic and aromatic content
    hydrophobic_ratio = hydrophobic_count / total
    aromatic_ratio = aromatic_count / total

    return min(1.0, hydrophobic_ratio * 0.6 + aromatic_ratio * 0.4 + 0.2)
