import json
from pathlib import Path

import pytest

from protein_design_mcp.adapters.rfdiffusion2 import build_args, parse_output
from protein_design_mcp.app import manifest_dir
from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.loader import load_manifests
from protein_design_mcp.validation import ToolInputError, validate_and_fill

BASE_PARAMS = {
    "target_pdb": "/tmp/target.pdb",
    "contig": "A1-150_100-100",
    "backend": "conda",
    "num_designs": 10,
    "diffusion_steps": 100,
    "noise_scale_ca": 1.0,
    "noise_scale_frame": 1.0,
    "ckpt_variant": "140",
}


def _manifest():
    return next(m for m in load_manifests(manifest_dir()) if m.name == "run_rfdiffusion2")


def _completed_run(tmp_path: Path, num_structures: int = 1) -> CompletedRun:
    structures = []
    for i in range(num_structures):
        p = tmp_path / f"design_{i}.pdb"
        p.write_text("ATOM\n")
        structures.append(str(p))
    return CompletedRun(
        returncode=0, stdout="", stderr="", workdir=tmp_path,
        outputs={"structures": structures},
    )


# ---------------------------------------------------------------------------
# Manifest shape
# ---------------------------------------------------------------------------


def test_manifest_loads_and_requires_gpu():
    m = _manifest()
    assert m.category == "binder_generation"
    assert m.requires.gpu is True


def test_manifest_has_no_hotspot_parameter():
    """Confirmed live: the only checkpoints on this host were never trained
    with hotspot conditioning, so this tool must not expose one that always
    fails."""
    assert "hotspot_res" not in _manifest().schema


def test_manifest_declares_a_timeout():
    assert _manifest().timeout_s > 3600


def test_manifest_dispatches_through_conda_prefix_by_default():
    """Security ruling: this tool must dispatch like every other engine
    (a mounted conda prefix, no Docker socket needed) -- Docker is an
    opt-in *parameter* (`backend`), never the manifest's own engine.prefix.
    """
    m = _manifest()
    assert m.engine.prefix.endswith("/rfd2_fixed")
    assert any(mount.endswith("/RFdiffusion2") for mount in m.engine.mounts)
    assert m.engine.env_vars.get("PYTHONPATH", "").endswith("/RFdiffusion2")


def test_manifest_backend_defaults_to_conda():
    spec = _manifest().schema["backend"]
    assert spec["default"] == "conda"
    assert set(spec["enum"]) == {"conda", "docker"}


# ---------------------------------------------------------------------------
# build_args
# ---------------------------------------------------------------------------


def test_build_args_serializes_job():
    args = build_args(_manifest(), BASE_PARAMS)
    assert len(args) == 1
    job = json.loads(args[0])
    assert job["target_pdb"] == "/tmp/target.pdb"
    assert job["contig"] == "A1-150_100-100"
    assert job["ckpt_variant"] == "140"


def test_build_args_includes_diffusion_steps_and_num_designs():
    args = build_args(_manifest(), BASE_PARAMS)
    job = json.loads(args[0])
    assert job["num_designs"] == 10
    assert job["diffusion_steps"] == 100


def test_build_args_passes_backend_through():
    args = build_args(_manifest(), BASE_PARAMS)
    job = json.loads(args[0])
    assert job["backend"] == "conda"

    docker_params = dict(BASE_PARAMS, backend="docker")
    args = build_args(_manifest(), docker_params)
    job = json.loads(args[0])
    assert job["backend"] == "docker"


# ---------------------------------------------------------------------------
# parse_output
# ---------------------------------------------------------------------------


def test_parse_output_reports_structure_count(tmp_path):
    run = _completed_run(tmp_path, num_structures=3)
    result = parse_output(_manifest(), run)
    assert result["num_structures"] == 3
    assert "poly-glycine" not in result["sequence_caveat"]
    assert "run_mpnn" in result["sequence_caveat"]


def test_parse_output_handles_single_structure_not_a_list(tmp_path):
    """Corner case: a single item is n=1, not the falsy-empty case."""
    p = tmp_path / "design_0.pdb"
    p.write_text("ATOM\n")
    run = CompletedRun(
        returncode=0, stdout="", stderr="", workdir=tmp_path,
        outputs={"structures": str(p)},
    )
    result = parse_output(_manifest(), run)
    assert result["num_structures"] == 1


def test_parse_output_raises_when_structures_missing(tmp_path):
    run = CompletedRun(returncode=0, stdout="", stderr="", workdir=tmp_path, outputs={})
    with pytest.raises(ValueError, match="structures"):
        parse_output(_manifest(), run)


# ---------------------------------------------------------------------------
# Schema validation -- RFdiffusion2's own contig grammar (underscore chains,
# comma sub-ranges), distinct from the other two generations'
# ---------------------------------------------------------------------------


def test_validation_accepts_wellformed_contig():
    params = validate_and_fill(
        _manifest(),
        {"target_pdb": "/tmp/target.pdb", "contig": "A1-150_100-100"},
    )
    assert params["contig"] == "A1-150_100-100"


def test_validation_rejects_rfdiffusion1_style_contig_with_spaces():
    """Corner case: a contig string written for run_rfdiffusion_binder
    (space-separated, "/0") must not silently pass here."""
    with pytest.raises(ToolInputError, match="contig"):
        validate_and_fill(
            _manifest(),
            {"target_pdb": "/tmp/target.pdb", "contig": "A1-150/0 100-100"},
        )


def test_validation_rejects_rfdiffusion3_style_contig_with_leading_slash_break():
    """Corner case: run_rfdiffusion3_binder's comma-separated "/0" break
    token must not silently pass here either -- RFdiffusion2 uses
    underscores, not "/0", for chain breaks."""
    with pytest.raises(ToolInputError, match="contig"):
        validate_and_fill(
            _manifest(),
            {"target_pdb": "/tmp/target.pdb", "contig": "50-50,/0,A1-150"},
        )


def test_validation_accepts_multi_subrange_chain():
    """Corner case: comma-joined sub-ranges within one chain segment."""
    params = validate_and_fill(
        _manifest(),
        {"target_pdb": "/tmp/target.pdb", "contig": "10,A20-25,10_A1-150"},
    )
    assert params["contig"] == "10,A20-25,10_A1-150"


def test_validation_rejects_ckpt_variant_not_in_enum():
    with pytest.raises(ToolInputError, match="ckpt_variant"):
        validate_and_fill(
            _manifest(),
            {
                "target_pdb": "/tmp/target.pdb",
                "contig": "A1-150_100-100",
                "ckpt_variant": "7",
            },
        )


def test_validation_fills_defaults():
    params = validate_and_fill(
        _manifest(),
        {"target_pdb": "/tmp/target.pdb", "contig": "A1-150_100-100"},
    )
    assert params["num_designs"] == 10
    assert params["diffusion_steps"] == 100
    assert params["ckpt_variant"] == "140"
    assert params["backend"] == "conda"


def test_validation_rejects_backend_not_in_enum():
    with pytest.raises(ToolInputError, match="backend"):
        validate_and_fill(
            _manifest(),
            {
                "target_pdb": "/tmp/target.pdb",
                "contig": "A1-150_100-100",
                "backend": "apptainer",
            },
        )


def test_validation_accepts_explicit_docker_backend():
    params = validate_and_fill(
        _manifest(),
        {
            "target_pdb": "/tmp/target.pdb",
            "contig": "A1-150_100-100",
            "backend": "docker",
        },
    )
    assert params["backend"] == "docker"
