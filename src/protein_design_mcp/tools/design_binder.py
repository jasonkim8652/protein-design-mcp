"""
Design binder tool - main end-to-end binder design pipeline.

This tool orchestrates:
1. RFdiffusion for backbone generation
2. ProteinMPNN for sequence design
3. ESMFold for structure validation
"""

import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from protein_design_mcp.pipelines.rfdiffusion import RFdiffusionRunner
from protein_design_mcp.pipelines.proteinmpnn import ProteinMPNNRunner
from protein_design_mcp.pipelines.esmfold import ESMFoldRunner
from protein_design_mcp.utils.metrics import calculate_metrics, filter_designs, rank_designs


@dataclass
class DesignResult:
    """Result from the binder design pipeline."""

    id: str
    sequence: str
    structure_pdb: str
    metrics: dict[str, float]


async def design_binder(
    target_pdb: str,
    hotspot_residues: list[str],
    num_designs: int = 10,
    binder_length: int = 80,
) -> dict[str, Any]:
    """
    Design protein binders for a target protein.

    This is the main end-to-end pipeline that orchestrates:
    1. RFdiffusion - Generate backbone structures for binders
    2. ProteinMPNN - Design sequences for each backbone
    3. ESMFold - Validate designed sequences by structure prediction

    Args:
        target_pdb: Path to target protein PDB file
        hotspot_residues: Residues on target for binder interface (e.g., ["A45", "A46"])
        num_designs: Number of backbone designs to generate
        binder_length: Length of binder in residues

    Returns:
        Dictionary containing:
        - designs: List of design results with sequences and metrics
        - summary: Statistics about the design run

    Raises:
        FileNotFoundError: If target PDB doesn't exist
        ValueError: If hotspot_residues is empty
    """
    # Validate inputs
    target_path = Path(target_pdb)
    if not target_path.exists():
        raise FileNotFoundError(f"Target PDB not found: {target_pdb}")

    if not hotspot_residues:
        raise ValueError("Hotspot residues cannot be empty")

    # Create temporary working directory
    job_id = str(uuid.uuid4())[:8]
    work_dir = Path(tempfile.mkdtemp(prefix=f"design_{job_id}_"))

    # Initialize runners
    rfdiffusion = RFdiffusionRunner()
    proteinmpnn = ProteinMPNNRunner()
    esmfold = ESMFoldRunner()

    all_designs = []

    # Step 1: Generate backbones with RFdiffusion
    backbone_dir = work_dir / "backbones"
    backbones = await rfdiffusion.generate_backbones(
        target_pdb=str(target_path),
        hotspot_residues=hotspot_residues,
        output_dir=str(backbone_dir),
        num_designs=num_designs,
        binder_length=binder_length,
    )

    # Step 2: Design sequences for each backbone with ProteinMPNN
    for backbone in backbones:
        backbone_id = backbone["id"]
        backbone_pdb = backbone["pdb_path"]

        # Skip if backbone file doesn't exist (e.g., mock scenario)
        if not Path(backbone_pdb).exists():
            continue

        mpnn_dir = work_dir / "sequences" / backbone_id
        sequences = await proteinmpnn.design_sequences(
            backbone_pdb=backbone_pdb,
            output_dir=str(mpnn_dir),
        )

        # Step 3: Validate each sequence with ESMFold
        for seq_info in sequences:
            sequence = seq_info["sequence"]
            seq_id = seq_info["id"]

            # For binder designs, ProteinMPNN outputs "TARGET/BINDER" —
            # extract the binder chain (last segment after "/")
            if "/" in sequence:
                sequence = sequence.split("/")[-1]

            # Predict structure
            prediction = await esmfold.predict_structure(sequence)

            # Calculate metrics
            metrics = calculate_metrics(
                plddt_per_residue=prediction.plddt_per_residue,
                ptm=prediction.ptm,
            )

            # Write PDB to disk instead of embedding inline (avoids multi-MB responses)
            design_id = f"{backbone_id}_{seq_id}"
            pdb_dir = work_dir / "predicted_structures"
            pdb_dir.mkdir(parents=True, exist_ok=True)
            pdb_path = pdb_dir / f"{design_id}.pdb"
            pdb_path.write_text(prediction.pdb_string)

            design = {
                "id": design_id,
                "sequence": sequence,
                "structure_pdb_path": str(pdb_path),
                "backbone_id": backbone_id,
                "metrics": {
                    "plddt": metrics.plddt,
                    "ptm": metrics.ptm,
                    "mpnn_score": seq_info.get("score"),
                },
            }

            all_designs.append(design)

    # Filter and rank designs
    filtered = filter_designs(all_designs, min_plddt=70.0, min_ptm=0.5)
    ranked = rank_designs(filtered)

    # Build summary
    summary = {
        "total_generated": len(all_designs),
        "passed_filters": len(filtered),
        "best_design_id": ranked[0]["id"] if ranked else None,
        "job_id": job_id,
    }

    return {
        "designs": ranked,
        "summary": summary,
    }
