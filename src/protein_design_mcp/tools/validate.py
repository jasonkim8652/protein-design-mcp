"""
Validation tool using ESMFold or AlphaFold2.

Validates designed protein sequences by predicting their structure
and calculating quality metrics.
"""

import tempfile
from pathlib import Path
from typing import Any, Literal

from protein_design_mcp.pipelines.esmfold import ESMFoldRunner, PredictionResult
from protein_design_mcp.pipelines.alphafold2 import AlphaFold2Runner


# Valid amino acid characters
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

# Supported structure predictors
SUPPORTED_PREDICTORS = ("esmfold", "alphafold2")


def _validate_sequence(sequence: str) -> bool:
    """
    Validate amino acid sequence.

    Args:
        sequence: Amino acid sequence

    Returns:
        True if valid, False otherwise
    """
    if not sequence:
        return False

    # Convert to uppercase for validation
    seq_upper = sequence.upper()

    # Check all characters are valid amino acids
    return all(aa in VALID_AA for aa in seq_upper)


async def validate_design(
    sequence: str,
    expected_structure: str | None = None,
    predictor: str = "esmfold",
) -> dict[str, Any]:
    """
    Validate a designed protein sequence.

    Uses ESMFold or AlphaFold2 to predict the structure and calculate quality metrics.

    Args:
        sequence: Amino acid sequence to validate
        expected_structure: Optional path to expected structure for RMSD comparison
        predictor: Structure predictor to use ("esmfold" or "alphafold2").
            Default is "esmfold" for faster predictions.
            Use "alphafold2" for potentially higher accuracy (requires ColabFold).

    Returns:
        Dictionary containing validation results and metrics:
        - predicted_structure_pdb: PDB string of predicted structure
        - plddt: Mean pLDDT score (0-100)
        - ptm: Predicted TM score (0-1)
        - plddt_per_residue: List of per-residue pLDDT scores
        - pae_matrix: Optional predicted aligned error matrix
        - rmsd_to_expected: Optional RMSD to expected structure
        - secondary_structure: Optional secondary structure string
        - predictor: The predictor used for this validation

    Raises:
        ValueError: If sequence is invalid or predictor is not supported
    """
    # Validate predictor
    if predictor not in SUPPORTED_PREDICTORS:
        raise ValueError(
            f"Unknown predictor: {predictor}. "
            f"Supported predictors: {SUPPORTED_PREDICTORS}"
        )

    # Validate sequence
    if not _validate_sequence(sequence):
        raise ValueError(
            f"Invalid sequence. Must contain only valid amino acids: "
            f"{sorted(VALID_AA)}"
        )

    # Create appropriate runner based on predictor
    if predictor == "esmfold":
        runner = ESMFoldRunner()
    elif predictor == "alphafold2":
        runner = AlphaFold2Runner()

    result = await runner.predict_structure(sequence.upper())

    # Write PDB to a persistent temp file instead of embedding inline
    import os

    pdb_file = tempfile.NamedTemporaryFile(
        suffix=".pdb", prefix="validate_", delete=False, mode="w"
    )
    pdb_file.write(result.pdb_string)
    pdb_file.close()

    # Build response dictionary
    response = {
        "predicted_structure_pdb": pdb_file.name,
        "plddt": result.plddt,
        "ptm": result.ptm,
        "plddt_per_residue": result.plddt_per_residue.tolist(),
        "predictor": predictor,
    }

    # Write PAE matrix to file if available (N x N can be huge)
    if result.pae_matrix is not None:
        import json as _json

        pae_path = pdb_file.name.replace(".pdb", "_pae.json")
        with open(pae_path, "w") as pf:
            _json.dump(result.pae_matrix.tolist(), pf)
        response["pae_matrix_path"] = pae_path

    # Calculate RMSD if expected structure provided
    if expected_structure:
        # Write predicted structure to temp file for RMSD calculation
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w") as f:
            f.write(result.pdb_string)
            predicted_pdb = f.name

        try:
            rmsd = runner.calculate_rmsd(predicted_pdb, expected_structure)
            response["rmsd_to_expected"] = rmsd
        finally:
            # Cleanup temp file
            Path(predicted_pdb).unlink(missing_ok=True)

    # Add secondary structure prediction (simple estimation from pLDDT)
    # High pLDDT regions are typically structured (helix/sheet)
    response["secondary_structure"] = _estimate_secondary_structure(
        result.plddt_per_residue
    )

    return response


def _estimate_secondary_structure(plddt_per_residue) -> str:
    """
    Estimate secondary structure from pLDDT values.

    This is a rough approximation. High pLDDT typically indicates
    structured regions (H=helix, E=sheet), while low pLDDT suggests
    flexible loops (C=coil).

    Args:
        plddt_per_residue: Array of per-residue pLDDT scores

    Returns:
        String of secondary structure characters (H/E/C)
    """
    ss = []
    for plddt in plddt_per_residue:
        if plddt >= 90:
            ss.append("H")  # Likely helix (very confident)
        elif plddt >= 70:
            ss.append("E")  # Likely sheet (confident)
        else:
            ss.append("C")  # Likely coil/loop (less confident)
    return "".join(ss)
