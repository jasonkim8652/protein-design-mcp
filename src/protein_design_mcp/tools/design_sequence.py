"""
Sequence design tool - design sequences for a given backbone using ProteinMPNN.

Unlike optimize_sequence (which refines existing sequences with ESM-2),
this tool uses ProteinMPNN to design entirely new sequences from scratch
given a backbone structure. This is the correct tool for de novo design
workflows where you have a backbone (e.g., from RFdiffusion) but no sequence.
"""

import tempfile
import uuid
from pathlib import Path
from typing import Any

from protein_design_mcp.pipelines.proteinmpnn import ProteinMPNNConfig, ProteinMPNNRunner
from protein_design_mcp.pipelines.esmfold import ESMFoldRunner
from protein_design_mcp.utils.metrics import calculate_metrics


async def design_sequence(
    backbone_pdb: str,
    num_sequences: int = 8,
    sampling_temp: float = 0.1,
    fixed_positions: list[int] | None = None,
    validate: bool = True,
) -> dict[str, Any]:
    """
    Design amino acid sequences for a protein backbone using ProteinMPNN.

    Given a backbone PDB (e.g., from RFdiffusion's generate_backbone),
    this tool uses ProteinMPNN to design sequences that fold into that
    structure. Optionally validates each design with ESMFold.

    Args:
        backbone_pdb: Path to backbone PDB file
        num_sequences: Number of sequences to design (default: 8)
        sampling_temp: ProteinMPNN sampling temperature (default: 0.1).
            Lower = more conservative, higher = more diverse.
        fixed_positions: Residue positions to keep fixed (1-indexed)
        validate: Whether to validate designs with ESMFold (default: True)

    Returns:
        Dictionary containing:
        - designs: List of designed sequences with metrics
        - summary: Statistics about the design run

    Raises:
        FileNotFoundError: If backbone PDB doesn't exist
    """
    backbone_path = Path(backbone_pdb)
    if not backbone_path.exists():
        raise FileNotFoundError(f"Backbone PDB not found: {backbone_pdb}")

    job_id = str(uuid.uuid4())[:8]
    work_dir = Path(tempfile.mkdtemp(prefix=f"designseq_{job_id}_"))

    # Configure ProteinMPNN with the requested temperature
    config = ProteinMPNNConfig(
        num_sequences=num_sequences,
        sampling_temp=sampling_temp,
    )
    mpnn = ProteinMPNNRunner(config=config)

    # Run ProteinMPNN
    mpnn_dir = work_dir / "sequences"
    raw_sequences = await mpnn.design_sequences(
        backbone_pdb=str(backbone_path),
        output_dir=str(mpnn_dir),
        fixed_positions=fixed_positions,
        num_sequences=num_sequences,
    )

    # Filter out the "native" reference sequence (recovery=1.0 typically)
    designed = [s for s in raw_sequences if s.get("recovery", 0) != 1.0]
    if not designed:
        designed = raw_sequences  # fallback: keep all

    # Optionally validate with ESMFold
    designs = []
    esmfold = ESMFoldRunner() if validate else None

    for seq_info in designed:
        sequence = seq_info["sequence"]
        # For complex designs, extract the designed chain
        if "/" in sequence:
            sequence = sequence.split("/")[-1]

        entry: dict[str, Any] = {
            "id": seq_info.get("id", f"seq_{len(designs)}"),
            "sequence": sequence,
            "length": len(sequence),
            "mpnn_score": seq_info.get("score"),
            "recovery": seq_info.get("recovery"),
        }

        if validate and esmfold is not None:
            prediction = await esmfold.predict_structure(sequence)
            metrics = calculate_metrics(
                plddt_per_residue=prediction.plddt_per_residue,
                ptm=prediction.ptm,
            )
            entry["plddt"] = metrics.plddt
            entry["ptm"] = metrics.ptm

            # Write PDB
            pdb_dir = work_dir / "predicted_structures"
            pdb_dir.mkdir(parents=True, exist_ok=True)
            pdb_path = pdb_dir / f"{entry['id']}.pdb"
            pdb_path.write_text(prediction.pdb_string)
            entry["structure_pdb_path"] = str(pdb_path)

        designs.append(entry)

    return {
        "designs": designs,
        "summary": {
            "total_designed": len(designs),
            "backbone_pdb": str(backbone_path),
            "sampling_temp": sampling_temp,
            "job_id": job_id,
        },
    }
