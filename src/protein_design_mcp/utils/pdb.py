"""
PDB file utilities.

Functions for parsing, validating, and manipulating PDB files.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from Bio.PDB import PDBIO, PDBParser

from protein_design_mcp.exceptions import InvalidPDBError


# Standard amino acid 3-letter to 1-letter mapping
THREE_TO_ONE = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
}


@dataclass
class Residue:
    """Representation of a protein residue."""

    chain_id: str
    residue_number: int
    residue_name: str
    atoms: list[dict[str, Any]]


@dataclass
class Chain:
    """Representation of a protein chain."""

    chain_id: str
    residues: list[Residue]
    sequence: str


@dataclass
class Structure:
    """Representation of a protein structure."""

    name: str
    chains: list[Chain]


def parse_pdb(pdb_path: str | Path) -> Structure:
    """
    Parse a PDB file into a Structure object.

    Args:
        pdb_path: Path to PDB file

    Returns:
        Structure object with chains and residues

    Raises:
        InvalidPDBError: If file doesn't exist or is malformed
        FileNotFoundError: If file doesn't exist
    """
    path = Path(pdb_path)

    if not path.exists():
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")

    parser = PDBParser(QUIET=True)

    try:
        bio_structure = parser.get_structure(path.stem, str(path))
    except Exception as e:
        raise InvalidPDBError(f"Failed to parse PDB file {pdb_path}: {e}") from e

    chains = []

    # Get first model (most PDB files have only one)
    try:
        model = next(bio_structure.get_models())
    except StopIteration:
        raise InvalidPDBError(f"No models found in PDB file: {pdb_path}")

    for bio_chain in model.get_chains():
        chain_id = bio_chain.id
        residues = []
        sequence_chars = []

        for bio_residue in bio_chain.get_residues():
            # Skip heteroatoms (water, ligands, etc.)
            hetflag = bio_residue.id[0]
            if hetflag != " ":
                continue

            res_name = bio_residue.resname.strip()
            res_num = bio_residue.id[1]

            # Convert 3-letter code to 1-letter
            one_letter = THREE_TO_ONE.get(res_name, "X")
            sequence_chars.append(one_letter)

            # Extract atoms
            atoms = []
            for atom in bio_residue.get_atoms():
                coord = atom.get_coord()
                atoms.append({
                    "name": atom.name,
                    "element": atom.element,
                    "x": float(coord[0]),
                    "y": float(coord[1]),
                    "z": float(coord[2]),
                    "occupancy": float(atom.occupancy),
                    "bfactor": float(atom.bfactor),
                })

            residue = Residue(
                chain_id=chain_id,
                residue_number=res_num,
                residue_name=res_name,
                atoms=atoms,
            )
            residues.append(residue)

        if residues:  # Only add chains with residues
            chain = Chain(
                chain_id=chain_id,
                residues=residues,
                sequence="".join(sequence_chars),
            )
            chains.append(chain)

    if not chains:
        raise InvalidPDBError(f"No valid protein chains found in: {pdb_path}")

    return Structure(name=path.stem, chains=chains)


def write_pdb(
    structure: Structure,
    output_path: str | Path,
) -> None:
    """
    Write a Structure object to a PDB file.

    Args:
        structure: Structure to write
        output_path: Output file path
    """
    from Bio.PDB.StructureBuilder import StructureBuilder
    from Bio.PDB.Atom import Atom
    from Bio.PDB.Residue import Residue as BioResidue
    from Bio.PDB.Chain import Chain as BioChain
    from Bio.PDB.Model import Model
    from Bio.PDB.Structure import Structure as BioStructure

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Build BioPython structure
    bio_structure = BioStructure(structure.name)
    model = Model(0)
    bio_structure.add(model)

    for chain in structure.chains:
        bio_chain = BioChain(chain.chain_id)
        model.add(bio_chain)

        for residue in chain.residues:
            # Create residue ID tuple: (hetflag, resseq, icode)
            res_id = (" ", residue.residue_number, " ")
            bio_residue = BioResidue(res_id, residue.residue_name, "")
            bio_chain.add(bio_residue)

            for i, atom_data in enumerate(residue.atoms):
                atom = Atom(
                    name=atom_data["name"],
                    coord=[atom_data["x"], atom_data["y"], atom_data["z"]],
                    bfactor=atom_data.get("bfactor", 0.0),
                    occupancy=atom_data.get("occupancy", 1.0),
                    altloc=" ",
                    fullname=f" {atom_data['name']:3s}",
                    serial_number=i + 1,
                    element=atom_data.get("element", atom_data["name"][0]),
                )
                bio_residue.add(atom)

    # Write to file
    io = PDBIO()
    io.set_structure(bio_structure)
    io.save(str(path))


def validate_pdb(pdb_path: str | Path) -> tuple[bool, list[str]]:
    """
    Validate a PDB file for common issues.

    Args:
        pdb_path: Path to PDB file

    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    path = Path(pdb_path)

    # Check file exists
    if not path.exists():
        return False, [f"File not found: {pdb_path}"]

    # Check file extension
    if path.suffix.lower() not in [".pdb", ".ent"]:
        issues.append(f"Unusual file extension: {path.suffix}")

    # Check for ATOM records
    has_atom_records = False
    try:
        with open(path, "r") as f:
            for line in f:
                if line.startswith("ATOM"):
                    has_atom_records = True
                    break
    except Exception as e:
        return False, [f"Cannot read file: {e}"]

    if not has_atom_records:
        issues.append("No ATOM records found in PDB file")
        return False, issues

    # Try to parse to check for structural issues
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(path.stem, str(path))

        # Check for chains
        model = next(structure.get_models())
        chain_count = len(list(model.get_chains()))
        if chain_count == 0:
            issues.append("No chains found in structure")
    except StopIteration:
        issues.append("No models found in PDB file")
    except Exception as e:
        issues.append(f"Parse error: {e}")

    return len(issues) == 0, issues


def extract_sequence(pdb_path: str | Path, chain_id: str | None = None) -> str:
    """
    Extract amino acid sequence from a PDB file.

    Args:
        pdb_path: Path to PDB file
        chain_id: Specific chain to extract (None for all chains)

    Returns:
        Amino acid sequence string

    Raises:
        InvalidPDBError: If chain_id not found
    """
    structure = parse_pdb(pdb_path)

    if chain_id is None:
        # Return sequence from all chains concatenated
        return "".join(chain.sequence for chain in structure.chains)

    # Find specific chain
    for chain in structure.chains:
        if chain.chain_id == chain_id:
            return chain.sequence

    raise InvalidPDBError(f"Chain {chain_id} not found in {pdb_path}")


def get_interface_residues(
    pdb_path: str | Path,
    chain_a: str,
    chain_b: str,
    distance_cutoff: float = 8.0,
) -> tuple[list[str], list[str]]:
    """
    Find interface residues between two chains.

    Args:
        pdb_path: Path to PDB file
        chain_a: First chain ID
        chain_b: Second chain ID
        distance_cutoff: Distance cutoff in Angstroms

    Returns:
        Tuple of (chain_a_residues, chain_b_residues)

    Raises:
        InvalidPDBError: If chain not found
    """
    import numpy as np

    structure = parse_pdb(pdb_path)

    # Find chains
    chain_a_obj = None
    chain_b_obj = None
    for chain in structure.chains:
        if chain.chain_id == chain_a:
            chain_a_obj = chain
        if chain.chain_id == chain_b:
            chain_b_obj = chain

    if chain_a_obj is None:
        raise InvalidPDBError(f"Chain {chain_a} not found")
    if chain_b_obj is None:
        raise InvalidPDBError(f"Chain {chain_b} not found")

    # Get CA atom coordinates for each residue
    def get_ca_coords(chain: Chain) -> dict[int, np.ndarray]:
        coords = {}
        for residue in chain.residues:
            for atom in residue.atoms:
                if atom["name"] == "CA":
                    coords[residue.residue_number] = np.array([
                        atom["x"], atom["y"], atom["z"]
                    ])
                    break
        return coords

    coords_a = get_ca_coords(chain_a_obj)
    coords_b = get_ca_coords(chain_b_obj)

    interface_a = set()
    interface_b = set()

    # Find residues within distance cutoff
    for res_a, coord_a in coords_a.items():
        for res_b, coord_b in coords_b.items():
            distance = np.linalg.norm(coord_a - coord_b)
            if distance <= distance_cutoff:
                interface_a.add(str(res_a))
                interface_b.add(str(res_b))

    return sorted(list(interface_a), key=int), sorted(list(interface_b), key=int)
