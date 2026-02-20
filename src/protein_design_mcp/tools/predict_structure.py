"""
Single-chain structure prediction tool.

Simpler than ``validate_design`` — predicts a structure and returns quality
metrics without RMSD comparison or secondary structure estimation.
"""

import tempfile
from typing import Any

from protein_design_mcp.pipelines.esmfold import ESMFoldRunner


VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")
SUPPORTED_PREDICTORS = ("esmfold", "alphafold2")


async def predict_structure(
    sequence: str,
    predictor: str = "esmfold",
) -> dict[str, Any]:
    """Predict the 3D structure of a single protein chain.

    Args:
        sequence: Amino acid sequence (standard 20 AAs).
        predictor: ``"esmfold"`` (fast) or ``"alphafold2"`` (potentially more
            accurate, requires ColabFold).

    Returns:
        Dict with ``predicted_structure_pdb`` (file path), ``plddt``, ``ptm``,
        ``plddt_per_residue``, and ``predictor``.
    """
    if predictor not in SUPPORTED_PREDICTORS:
        raise ValueError(
            f"Unknown predictor: {predictor}. "
            f"Supported: {SUPPORTED_PREDICTORS}"
        )

    # Clean and validate sequence
    sequence = sequence.upper().strip()
    sequence = "".join(c for c in sequence if c in VALID_AA)
    if not sequence:
        raise ValueError("Sequence must contain valid amino acids")

    # Create runner
    if predictor == "esmfold":
        runner = ESMFoldRunner()
    elif predictor == "alphafold2":
        from protein_design_mcp.pipelines.alphafold2 import AlphaFold2Runner
        runner = AlphaFold2Runner()

    result = await runner.predict_structure(sequence)

    # Write PDB to file
    pdb_file = tempfile.NamedTemporaryFile(
        suffix=".pdb", prefix="predict_", delete=False, mode="w"
    )
    pdb_file.write(result.pdb_string)
    pdb_file.close()

    response: dict[str, Any] = {
        "predicted_structure_pdb": pdb_file.name,
        "plddt": result.plddt,
        "ptm": result.ptm,
        "plddt_per_residue": result.plddt_per_residue.tolist(),
        "predictor": predictor,
        "sequence_length": len(sequence),
    }

    # Write PAE matrix if available
    if result.pae_matrix is not None:
        import json as _json

        pae_path = pdb_file.name.replace(".pdb", "_pae.json")
        with open(pae_path, "w") as pf:
            _json.dump(result.pae_matrix.tolist(), pf)
        response["pae_matrix_path"] = pae_path

    return response
