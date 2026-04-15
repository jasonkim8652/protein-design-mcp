"""
Design fold tool - end-to-end de novo fold design pipeline.

This tool orchestrates:
1. RFdiffusion (unconditional) for backbone generation
2. ProteinMPNN for sequence design
3. AlphaFold2 for structure validation
"""

import tempfile
import uuid
from pathlib import Path
from typing import Any

from protein_design_mcp.pipelines.rfdiffusion import run_unconditional
from protein_design_mcp.pipelines.proteinmpnn import ProteinMPNNRunner
from protein_design_mcp.utils.metrics import calculate_metrics, filter_designs, rank_designs


async def design_fold(
    length: int,
    num_designs: int = 10,
    num_sequences_per_backbone: int = 4,
    sampling_temp: float = 0.1,
) -> dict[str, Any]:
    """
    End-to-end de novo fold design pipeline.

    Orchestrates:
    1. RFdiffusion — unconditional backbone generation
    2. ProteinMPNN — sequence design for each backbone
    3. AlphaFold2 — structure validation (pLDDT, pTM)

    Args:
        length: Backbone length in residues.
        num_designs: Number of backbone designs to generate.
        num_sequences_per_backbone: ProteinMPNN sequences per backbone.
        sampling_temp: ProteinMPNN sampling temperature.

    Returns:
        Dictionary with designs list and summary.
    """
    job_id = str(uuid.uuid4())[:8]
    work_dir = Path(tempfile.mkdtemp(prefix=f"fold_{job_id}_"))

    # Step 1: Generate unconditional backbones with RFdiffusion
    rfd_result = await run_unconditional(
        length=length,
        num_designs=num_designs,
        output_dir=str(work_dir / "backbones"),
    )

    backbones = rfd_result.get("designs", [])

    # Step 2: Design sequences for each backbone with ProteinMPNN
    proteinmpnn = ProteinMPNNRunner()
    all_designs = []

    for backbone in backbones:
        backbone_id = backbone.get("id", "unk")
        backbone_pdb = backbone.get("pdb_path", "")

        if not backbone_pdb or not Path(backbone_pdb).exists():
            continue

        mpnn_dir = work_dir / "sequences" / backbone_id
        sequences = await proteinmpnn.design_sequences(
            backbone_pdb=backbone_pdb,
            output_dir=str(mpnn_dir),
            num_sequences=num_sequences_per_backbone,
            sampling_temp=sampling_temp,
        )

        # Step 3: Validate each sequence with AlphaFold2
        # Fall back to ESMFold if AF2 is unavailable
        try:
            from protein_design_mcp.pipelines.alphafold2 import AlphaFold2Runner
            predictor = AlphaFold2Runner()
            predictor_name = "alphafold2"
        except (ImportError, Exception):
            from protein_design_mcp.pipelines.esmfold import ESMFoldRunner
            predictor = ESMFoldRunner()
            predictor_name = "esmfold"

        for seq_info in sequences:
            sequence = seq_info["sequence"]
            seq_id = seq_info["id"]

            prediction = await predictor.predict_structure(sequence)

            metrics = calculate_metrics(
                plddt_per_residue=prediction.plddt_per_residue,
                ptm=prediction.ptm,
            )

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
                "predictor": predictor_name,
            }
            all_designs.append(design)

    filtered = filter_designs(all_designs, min_plddt=70.0, min_ptm=0.5)
    ranked = rank_designs(filtered)

    summary = {
        "total_backbones": len(backbones),
        "total_designs": len(all_designs),
        "passed_filters": len(filtered),
        "best_design_id": ranked[0]["id"] if ranked else None,
        "job_id": job_id,
        "predictor": predictor_name,
    }

    return {
        "designs": ranked,
        "summary": summary,
    }
