from pathlib import Path

import numpy as np
import pytest

from protein_design_mcp.adapters.boltzgen_design_fold import build_args, parse_output
from protein_design_mcp.app import manifest_dir
from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.loader import load_manifests
from protein_design_mcp.validation import ToolInputError, validate_and_fill

MANIFEST_DIR = manifest_dir()


def _manifest():
    return next(
        m for m in load_manifests(MANIFEST_DIR) if m.name == "run_boltzgen_design_fold"
    )


def _base_params(**overrides):
    return validate_and_fill(
        _manifest(),
        {
            "design_spec": "design.yaml",
            "generated_files": ["a.cif", "a.npz"],
            **overrides,
        },
    )


def test_manifest_loads_and_is_gpu_structure_prediction():
    m = _manifest()
    assert m.category == "structure_prediction"
    assert m.requires.gpu is True


def test_manifest_uses_prefix_and_stages_generated_files():
    engine = _manifest().engine
    assert engine.prefix == "/home/jk661/miniforge3/envs/boltzgen"
    assert engine.stage == ("generated_files",)


def test_outputs_use_design_specific_subdirectories():
    outputs = {o.name: o.pattern for o in _manifest().outputs}
    assert outputs["refolded_structures"] == "generated_files/refold_design_cif/*.cif"
    assert outputs["refold_metrics"] == "generated_files/fold_out_design_npz/*.npz"


def test_validation_fills_defaults():
    params = _base_params()
    assert params["recycling_steps"] == 3
    assert params["diffusion_samples"] == 5


def test_validation_rejects_single_item_generated_files():
    with pytest.raises(ToolInputError):
        validate_and_fill(
            _manifest(), {"design_spec": "d.yaml", "generated_files": ["a.cif"]}
        )


def test_build_args_wraps_boltzgen_run_with_design_folding_step():
    params = _base_params(generated_files=[
        "/scratch/generated_files/a.cif", "/scratch/generated_files/a.npz",
    ])
    args = build_args(_manifest(), params)
    assert args[args.index("--steps") + 1] == "design_folding"
    assert "--protocol" not in args


def test_build_args_sets_designfolding_writer_flags():
    params = _base_params(generated_files=["/s/a.cif", "/s/a.npz"])
    args = build_args(_manifest(), params)
    joined = " ".join(args)
    assert "writer.designfolding=true" in joined
    assert "data.cfg.return_designfolding=true" in joined


def test_build_args_derives_design_dir_from_staged_files_parent():
    params = _base_params(generated_files=[
        "/scratch/generated_files/a.cif", "/scratch/generated_files/a.npz",
    ])
    args = build_args(_manifest(), params)
    joined = " ".join(args)
    assert "data.design_dir=/scratch/generated_files" in joined


def _write_design_fold_npz(path: Path, iptm: list[float], ptm: list[float]) -> None:
    np.savez_compressed(
        path,
        iptm=np.array(iptm, dtype=np.float32),
        ptm=np.array(ptm, dtype=np.float32),
        design_ptm=np.array(ptm, dtype=np.float32),
        design_iptm=np.array(iptm, dtype=np.float32),
        # No target-relative keys at all -- this mode never has a target.
    )


def test_parse_output_omits_target_relative_fields(tmp_path):
    npz = tmp_path / "design_0.npz"
    _write_design_fold_npz(npz, iptm=[0.4, 0.8], ptm=[0.5, 0.6])
    run = CompletedRun(
        returncode=0, stdout="", stderr="", workdir=tmp_path,
        outputs={"refold_metrics": [str(npz)]},
    )
    result = parse_output(_manifest(), run)
    refold = result["refolds"][0]
    assert refold["best_sample_index"] == 1
    assert "design_to_target_iptm" not in refold
    assert "protein_iptm" not in refold
    assert "min_interaction_pae" not in refold


def test_parse_output_handles_a_single_sample(tmp_path):
    npz = tmp_path / "design_0.npz"
    _write_design_fold_npz(npz, iptm=[0.42], ptm=[0.5])
    run = CompletedRun(
        returncode=0, stdout="", stderr="", workdir=tmp_path,
        outputs={"refold_metrics": str(npz)},
    )
    result = parse_output(_manifest(), run)
    assert result["refolds"][0]["best_sample_index"] == 0
    assert result["refolds"][0]["num_samples"] == 1


def test_parse_output_raises_when_refold_metrics_missing(tmp_path):
    run = CompletedRun(returncode=0, stdout="", stderr="", workdir=tmp_path, outputs={})
    with pytest.raises(ValueError, match="refold_metrics"):
        parse_output(_manifest(), run)
