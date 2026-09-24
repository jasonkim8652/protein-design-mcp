from pathlib import Path

import numpy as np
import pytest

from protein_design_mcp.adapters.boltzgen_fold import build_args, parse_output
from protein_design_mcp.app import manifest_dir
from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.loader import load_manifests
from protein_design_mcp.validation import ToolInputError, validate_and_fill

MANIFEST_DIR = manifest_dir()


def _manifest():
    return next(m for m in load_manifests(MANIFEST_DIR) if m.name == "run_boltzgen_fold")


def _base_params(**overrides):
    return validate_and_fill(
        _manifest(),
        {
            "design_spec": "design.yaml",
            "generated_files": ["a.cif", "a.npz"],
            # Required and undefaulted: refolding in complex and refolding the
            # design alone are different experiments, so the mode is stated.
            "with_target": True,
            **overrides,
        },
    )


# --- manifest shape ---


def test_manifest_loads_and_is_gpu_structure_prediction():
    m = _manifest()
    assert m.category == "structure_prediction"
    assert m.composite is False
    assert m.requires.gpu is True


def test_manifest_uses_prefix_and_stages_generated_files():
    engine = _manifest().engine
    assert engine.prefix == "/home/jk661/miniforge3/envs/boltzgen"
    assert engine.stage == ("generated_files",)


def test_generated_files_requires_at_least_two_items():
    schema = _manifest().schema
    assert schema["generated_files"]["minItems"] == 2


# --- validation ---


def test_validation_fills_defaults():
    params = _base_params()
    assert params["recycling_steps"] == 3
    assert params["sampling_steps"] == 200
    assert params["diffusion_samples"] == 5
    assert params["use_kernels"] == "auto"


def test_validation_rejects_single_item_generated_files():
    with pytest.raises(ToolInputError):
        validate_and_fill(
            _manifest(), {"design_spec": "d.yaml", "generated_files": ["a.cif"]}
        )


def test_validation_rejects_empty_generated_files():
    with pytest.raises(ToolInputError):
        validate_and_fill(
            _manifest(), {"design_spec": "d.yaml", "generated_files": []}
        )


def test_validation_accepts_boundary_diffusion_samples():
    params = _base_params(diffusion_samples=25)
    assert params["diffusion_samples"] == 25
    with pytest.raises(ToolInputError):
        _base_params(diffusion_samples=26)


# --- build_args ---


def test_build_args_wraps_boltzgen_run_with_folding_step_only():
    params = _base_params(generated_files=[
        "/scratch/generated_files/a.cif", "/scratch/generated_files/a.npz",
    ])
    args = build_args(_manifest(), params)
    assert args[0] == str(Path("design.yaml"))
    assert args[args.index("--steps") + 1] == "folding"
    assert "--protocol" not in args


def test_build_args_derives_design_dir_from_staged_files_parent():
    params = _base_params(generated_files=[
        "/scratch/generated_files/a.cif", "/scratch/generated_files/a.npz",
    ])
    args = build_args(_manifest(), params)
    joined = " ".join(args)
    assert "data.design_dir=/scratch/generated_files" in joined
    assert "output=/scratch/generated_files" in joined


def test_build_args_includes_checkpoint_and_sampling_knobs():
    params = _base_params(
        generated_files=["/s/a.cif", "/s/a.npz"],
        recycling_steps=5, sampling_steps=50, diffusion_samples=2,
    )
    args = build_args(_manifest(), params)
    joined = " ".join(args)
    assert "recycling_steps=5" in joined
    assert "sampling_steps=50" in joined
    assert "diffusion_samples=2" in joined
    assert "checkpoint=" not in joined  # not via --config, see next test


def test_build_args_passes_checkpoint_moldir_use_kernels_as_top_level_flags():
    """These three MUST be top-level CLI flags, not --config overrides:
    BoltzGen's own CLI resolves a huggingface:repo:file reference to a real
    local path (and 'auto' use_kernels to an actual bool) before embedding
    it into the step's args -- a --config override bypasses that and hands
    the engine the raw unresolved string, which fails hard (verified live:
    "Invalid moldir. Expected directory or zip file: huggingface:...")."""
    params = _base_params(generated_files=["/s/a.cif", "/s/a.npz"])
    args = build_args(_manifest(), params)
    assert args[args.index("--folding_checkpoint") + 1] == (
        "huggingface:boltzgen/boltzgen-1:boltz2_conf_final.ckpt"
    )
    assert args[args.index("--moldir") + 1] == (
        "huggingface:boltzgen/inference-data:mols.zip"
    )
    assert args[args.index("--use_kernels") + 1] == "auto"


# --- parse_output ---


def _write_fold_npz(path: Path, iptm: list[float], ptm: list[float]) -> None:
    np.savez_compressed(
        path,
        iptm=np.array(iptm, dtype=np.float32),
        ptm=np.array(ptm, dtype=np.float32),
        design_ptm=np.array(ptm, dtype=np.float32),
        design_iptm=np.array(iptm, dtype=np.float32),
        design_to_target_iptm=np.array(iptm, dtype=np.float32),
        min_interaction_pae=np.array([5.0] * len(iptm), dtype=np.float32),
    )


def test_parse_output_picks_the_highest_confidence_sample(tmp_path):
    """confidence = 0.8*iptm + 0.2*ptm -- sample index 1 must win here even
    though it is not the first or the highest-iptm-alone sample."""
    npz = tmp_path / "design_0.npz"
    _write_fold_npz(npz, iptm=[0.5, 0.9, 0.6], ptm=[0.9, 0.7, 0.5])
    run = CompletedRun(
        returncode=0, stdout="", stderr="", workdir=tmp_path,
        outputs={"refold_metrics": [str(npz)]},
    )
    result = parse_output(_manifest(), run)
    assert result["num_refolds"] == 1
    refold = result["refolds"][0]
    assert refold["id"] == "design_0"
    assert refold["best_sample_index"] == 1
    assert refold["num_samples"] == 3
    assert refold["design_iptm"] == pytest.approx(0.9)


def test_parse_output_handles_a_single_sample(tmp_path):
    """Corner case: diffusion_samples=1 -- argmax over a length-1 array
    must not crash and must pick index 0."""
    npz = tmp_path / "design_0.npz"
    _write_fold_npz(npz, iptm=[0.42], ptm=[0.5])
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
