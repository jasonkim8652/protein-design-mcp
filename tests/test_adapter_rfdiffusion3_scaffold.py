import json
from pathlib import Path

import pytest

from protein_design_mcp.adapters.rfdiffusion3_scaffold import build_args, parse_output
from protein_design_mcp.app import manifest_dir
from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.loader import load_manifests
from protein_design_mcp.validation import ToolInputError, validate_and_fill

BASE_PARAMS = {
    "length": "80",
    "motif_pdb": None,
    "motif_contig": None,
    "redesign_motif_sidechains": False,
    "diffusion_batch_size": 8,
    "num_timesteps": 200,
    "step_scale": 1.5,
    "seed": None,
}


def _manifest():
    return next(
        m for m in load_manifests(manifest_dir()) if m.name == "run_rfdiffusion3_scaffold"
    )


def _completed_run(tmp_path: Path, metrics: dict, num_structures: int = 1) -> CompletedRun:
    structures = []
    metadata = []
    for i in range(num_structures):
        cif = tmp_path / f"input_job_0_model_{i}.cif.gz"
        cif.write_bytes(b"\x1f\x8b")
        structures.append(str(cif))
        meta_path = tmp_path / f"input_job_0_model_{i}.json"
        meta_path.write_text(json.dumps({
            "metrics": metrics,
            "ckpt_path": "/home/jk661/.foundry/checkpoints/rfd3_latest.ckpt",
        }))
        metadata.append(str(meta_path))
    return CompletedRun(
        returncode=0, stdout="", stderr="", workdir=tmp_path,
        outputs={"structure_cif": structures, "metadata_json": metadata},
    )


def test_manifest_loads_and_requires_gpu():
    m = _manifest()
    assert m.category == "monomer_generation"
    assert m.requires.gpu is True


def test_build_args_unconditional_job_has_no_input():
    args = build_args(_manifest(), BASE_PARAMS)
    payload = json.loads(args[0])
    assert payload["job"]["length"] == "80"
    assert "input" not in payload["job"]
    assert "contig" not in payload["job"]


def test_build_args_motif_scaffolding_includes_input_and_contig():
    params = dict(BASE_PARAMS, motif_pdb="/tmp/motif.pdb", motif_contig="20-20,A10-15,24-24")
    args = build_args(_manifest(), params)
    payload = json.loads(args[0])
    assert payload["job"]["input"] == "/tmp/motif.pdb"
    assert payload["job"]["contig"] == "20-20,A10-15,24-24"


def test_build_args_rejects_motif_pdb_without_contig():
    params = dict(BASE_PARAMS, motif_pdb="/tmp/motif.pdb", motif_contig=None)
    with pytest.raises(ValueError, match="motif_contig"):
        build_args(_manifest(), params)


def test_build_args_rejects_motif_contig_without_pdb():
    params = dict(BASE_PARAMS, motif_pdb=None, motif_contig="20-20,A10-15,24-24")
    with pytest.raises(ValueError, match="motif_pdb"):
        build_args(_manifest(), params)


def test_parse_output_reads_rank0_metrics(tmp_path):
    metrics = {"radius_of_gyration": 10.5, "n_clashing.interresidue_clashes_w_sidechain": 0}
    run = _completed_run(tmp_path, metrics, num_structures=2)
    result = parse_output(_manifest(), run)
    assert result["num_structures"] == 2
    assert result["metrics"] == metrics


def test_parse_output_raises_when_metadata_missing(tmp_path):
    run = CompletedRun(returncode=0, stdout="", stderr="", workdir=tmp_path, outputs={})
    with pytest.raises(ValueError, match="metadata_json"):
        parse_output(_manifest(), run)


def test_validation_requires_length():
    with pytest.raises(ToolInputError, match="length"):
        validate_and_fill(_manifest(), {})


def test_validation_accepts_length_range():
    params = validate_and_fill(_manifest(), {"length": "60-100"})
    assert params["length"] == "60-100"


def test_validation_rejects_malformed_length():
    with pytest.raises(ToolInputError, match="length"):
        validate_and_fill(_manifest(), {"length": "sixty"})


def test_validation_fills_diffusion_defaults():
    params = validate_and_fill(_manifest(), {"length": "80"})
    assert params["diffusion_batch_size"] == 8
    assert params["num_timesteps"] == 200
    assert params["step_scale"] == 1.5
