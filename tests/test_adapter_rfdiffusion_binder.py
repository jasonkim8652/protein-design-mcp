from pathlib import Path

import pytest

from protein_design_mcp.adapters.rfdiffusion_binder import build_args, parse_output
from protein_design_mcp.app import manifest_dir
from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.loader import load_manifests
from protein_design_mcp.validation import ToolInputError, validate_and_fill

BASE_PARAMS = {
    "target_pdb": "/tmp/target.pdb",
    "contig": "B1-100/0 100-100",
    "hotspot_res": ["A30", "A33", "A34"],
    "num_designs": 10,
    "diffusion_steps": 50,
    "partial_t": None,
    "provide_seq": [],
    "noise_scale_ca": 1.0,
    "noise_scale_frame": 1.0,
    "deterministic": False,
    "ckpt_variant": "auto",
}


def _manifest():
    return next(m for m in load_manifests(manifest_dir()) if m.name == "run_rfdiffusion_binder")


def _completed_run(tmp_path: Path, *, stdout: str, num_structures: int = 1) -> CompletedRun:
    structures = []
    for i in range(num_structures):
        p = tmp_path / f"design_{i}.pdb"
        p.write_text("ATOM\n")
        structures.append(str(p))
    return CompletedRun(
        returncode=0,
        stdout=stdout,
        stderr="",
        workdir=tmp_path,
        outputs={"structures": structures},
    )


# ---------------------------------------------------------------------------
# Manifest shape
# ---------------------------------------------------------------------------


def test_manifest_loads_and_requires_gpu():
    m = _manifest()
    assert m.category == "binder_generation"
    assert m.requires.gpu is True


def test_manifest_declares_a_timeout():
    assert _manifest().timeout_s > 3600


def test_manifest_uses_a_mounted_prefix_and_pythonpath_workaround():
    m = _manifest()
    assert m.engine.prefix.endswith("/SE3nv")
    assert "PYTHONPATH" in m.engine.env_vars
    assert m.engine.env_vars["PYTHONPATH"].endswith("/RFdiffusion")


def test_manifest_states_diffusion_steps_floor_of_15():
    spec = _manifest().schema["diffusion_steps"]
    assert spec["minimum"] == 15


# ---------------------------------------------------------------------------
# build_args
# ---------------------------------------------------------------------------


def test_build_args_includes_contig_hotspots_and_target():
    args = build_args(_manifest(), BASE_PARAMS)
    assert "inference.input_pdb=/tmp/target.pdb" in args
    assert "contigmap.contigs=[B1-100/0 100-100]" in args
    assert "ppi.hotspot_res=[A30,A33,A34]" in args
    assert "diffuser.T=50" in args


def test_build_args_handles_empty_hotspot_list():
    """Corner case: empty collection, not conflated with the field being absent."""
    params = dict(BASE_PARAMS, hotspot_res=[])
    args = build_args(_manifest(), params)
    assert not any(a.startswith("ppi.hotspot_res=") for a in args)


def test_build_args_omits_partial_t_when_none():
    args = build_args(_manifest(), BASE_PARAMS)
    assert not any(a.startswith("diffuser.partial_T=") for a in args)


def test_build_args_includes_partial_t_when_set():
    params = dict(BASE_PARAMS, partial_t=20)
    args = build_args(_manifest(), params)
    assert "diffuser.partial_T=20" in args


def test_build_args_provide_seq_requires_partial_t():
    params = dict(BASE_PARAMS, partial_t=None, provide_seq=["100-119"])
    with pytest.raises(ValueError, match="partial_t"):
        build_args(_manifest(), params)


def test_build_args_provide_seq_with_partial_t_included():
    params = dict(BASE_PARAMS, partial_t=20, provide_seq=["100-119"])
    args = build_args(_manifest(), params)
    assert "contigmap.provide_seq=[100-119]" in args


def test_build_args_beta_ckpt_variant():
    params = dict(BASE_PARAMS, ckpt_variant="beta")
    args = build_args(_manifest(), params)
    assert "inference.ckpt_override_path=models/Complex_beta_ckpt.pt" in args


def test_build_args_beta_rejects_provide_seq_combo():
    params = dict(BASE_PARAMS, ckpt_variant="beta", partial_t=20, provide_seq=["100-119"])
    with pytest.raises(ValueError, match="beta"):
        build_args(_manifest(), params)


def test_build_args_auto_ckpt_variant_adds_no_override():
    args = build_args(_manifest(), BASE_PARAMS)
    assert not any(a.startswith("inference.ckpt_override_path=") for a in args)


# ---------------------------------------------------------------------------
# parse_output
# ---------------------------------------------------------------------------

STDOUT_SAMPLE = (
    "Reading checkpoint from /file_server/data/jk661/pioneer/RFdiffusion/"
    "rfdiffusion/inference/../../models/Complex_base_ckpt.pt\n"
    "Using contig: ['B1-100/0 100-100']\n"
)


def test_parse_output_reports_checkpoint_and_contig(tmp_path):
    run = _completed_run(tmp_path, stdout=STDOUT_SAMPLE, num_structures=3)
    result = parse_output(_manifest(), run)
    assert result["num_structures"] == 3
    assert result["checkpoint_used"].endswith("Complex_base_ckpt.pt")
    assert result["contig_used"] == "['B1-100/0 100-100']"
    assert "poly-glycine" in result["sequence_caveat"]


def test_parse_output_handles_single_structure_not_a_list(tmp_path):
    """Corner case: a single item is n=1, not the falsy-empty case."""
    p = tmp_path / "design_0.pdb"
    p.write_text("ATOM\n")
    run = CompletedRun(
        returncode=0, stdout=STDOUT_SAMPLE, stderr="", workdir=tmp_path,
        outputs={"structures": str(p)},
    )
    result = parse_output(_manifest(), run)
    assert result["num_structures"] == 1


def test_parse_output_raises_when_structures_missing(tmp_path):
    run = CompletedRun(
        returncode=0, stdout="", stderr="", workdir=tmp_path, outputs={}
    )
    with pytest.raises(ValueError, match="structures"):
        parse_output(_manifest(), run)


def test_parse_output_tolerates_missing_log_lines(tmp_path):
    run = _completed_run(tmp_path, stdout="no useful log lines here\n", num_structures=1)
    result = parse_output(_manifest(), run)
    assert result["checkpoint_used"] is None
    assert result["contig_used"] is None


# ---------------------------------------------------------------------------
# Schema validation -- the exact bug class this wave must not reproduce
# ---------------------------------------------------------------------------


def test_validation_rejects_hotspot_missing_chain_letter():
    with pytest.raises(ToolInputError, match="hotspot_res"):
        validate_and_fill(
            _manifest(),
            {
                "target_pdb": "/tmp/target.pdb",
                "contig": "B1-100/0 100-100",
                "hotspot_res": ["45"],
            },
        )


def test_validation_rejects_hotspot_with_colon_separator():
    with pytest.raises(ToolInputError, match="hotspot_res"):
        validate_and_fill(
            _manifest(),
            {
                "target_pdb": "/tmp/target.pdb",
                "contig": "B1-100/0 100-100",
                "hotspot_res": ["A:45"],
            },
        )


def test_validation_accepts_well_formed_hotspot():
    params = validate_and_fill(
        _manifest(),
        {
            "target_pdb": "/tmp/target.pdb",
            "contig": "B1-100/0 100-100",
            "hotspot_res": ["A45"],
        },
    )
    assert params["hotspot_res"] == ["A45"]


def test_validation_defaults_hotspot_res_to_empty_list():
    params = validate_and_fill(
        _manifest(),
        {"target_pdb": "/tmp/target.pdb", "contig": "B1-100/0 100-100"},
    )
    assert params["hotspot_res"] == []


def test_validation_rejects_malformed_contig_with_bad_separator():
    with pytest.raises(ToolInputError, match="contig"):
        validate_and_fill(
            _manifest(),
            {
                "target_pdb": "/tmp/target.pdb",
                "contig": "B1-100,100-100",
            },
        )


def test_validation_requires_diffusion_steps_at_least_15():
    with pytest.raises(ToolInputError, match="diffusion_steps"):
        validate_and_fill(
            _manifest(),
            {
                "target_pdb": "/tmp/target.pdb",
                "contig": "B1-100/0 100-100",
                "diffusion_steps": 10,
            },
        )


def test_validation_fills_defaults():
    params = validate_and_fill(
        _manifest(),
        {"target_pdb": "/tmp/target.pdb", "contig": "B1-100/0 100-100"},
    )
    assert params["num_designs"] == 10
    assert params["diffusion_steps"] == 50
    assert params["ckpt_variant"] == "auto"
