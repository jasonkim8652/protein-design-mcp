from pathlib import Path

import pytest

from protein_design_mcp.adapters.frameflow import build_args, parse_output
from protein_design_mcp.app import manifest_dir
from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.loader import load_manifests
from protein_design_mcp.validation import ToolInputError, validate_and_fill

MINIMAL_PDB = (
    "ATOM      1  N   GLY A   1      11.104  13.207   2.145  1.00  0.00           N\n"
    "ATOM      2  CA  GLY A   1      12.560  13.207   2.145  1.00  0.00           C\n"
    "ATOM      3  C   GLY A   1      13.090  14.630   2.145  1.00  0.00           C\n"
    "ATOM      4  N   GLY A   2      14.400  14.700   2.200  1.00  0.00           N\n"
    "ATOM      5  CA  GLY A   2      15.100  16.000   2.300  1.00  0.00           C\n"
    "TER\n"
)

DEFAULT_PARAMS = {
    "min_length": 70,
    "max_length": 70,
    "length_step": 10,
    "samples_per_length": 1,
    "num_timesteps": 100,
    "min_t": 0.01,
    "self_condition": True,
    "checkpoint_variant": "pdb",
    "seed": 123,
}


def _manifest():
    return next(m for m in load_manifests(manifest_dir()) if m.name == "run_frameflow")


def test_manifest_loads_and_requires_gpu():
    m = _manifest()
    assert m.category == "monomer_generation"
    assert m.requires.gpu is True


def test_manifest_mounts_the_repo_root_not_just_openfold_subdir():
    assert _manifest().engine.mounts == ("/home/jk661/projects/frameflow",)


def test_manifest_sets_pythonpath_env_var():
    assert (
        _manifest().engine.env_vars.get("PYTHONPATH")
        == "/home/jk661/projects/frameflow"
    )


def test_build_args_never_sets_output_dir():
    args = build_args(_manifest(), DEFAULT_PARAMS)
    assert not any("output_dir" in a for a in args)


def test_build_args_nulls_length_subset_so_min_max_take_effect():
    # Regression: inference_unconditional.yaml's own default
    # length_subset=[70,100,200,300] silently overrode min/max_length until
    # this was nulled explicitly -- confirmed live on GPU 7.
    args = build_args(_manifest(), DEFAULT_PARAMS)
    assert "inference.samples.length_subset=null" in args


def test_build_args_resolves_checkpoint_variant_to_full_path():
    args = build_args(_manifest(), DEFAULT_PARAMS)
    ckpt_arg = next(a for a in args if a.startswith("inference.ckpt_path="))
    assert ckpt_arg == "inference.ckpt_path=/home/jk661/projects/frameflow/weights/pdb/published.ckpt"


def test_build_args_uses_scope_checkpoint_when_selected():
    params = {**DEFAULT_PARAMS, "checkpoint_variant": "scope"}
    args = build_args(_manifest(), params)
    ckpt_arg = next(a for a in args if a.startswith("inference.ckpt_path="))
    assert "scope/published.ckpt" in ckpt_arg


def test_build_args_encodes_self_condition_as_hydra_bool():
    args = build_args(_manifest(), {**DEFAULT_PARAMS, "self_condition": False})
    assert "inference.interpolant.self_condition=false" in args
    args_true = build_args(_manifest(), {**DEFAULT_PARAMS, "self_condition": True})
    assert "inference.interpolant.self_condition=true" in args_true


def test_parse_output_uses_parent_sample_dir_name_as_id(tmp_path: Path):
    sample_dir = tmp_path / "inference_outputs" / "length_70" / "sample_0"
    sample_dir.mkdir(parents=True)
    pdb_path = sample_dir / "sample.pdb"
    pdb_path.write_text(MINIMAL_PDB)
    run = CompletedRun(
        returncode=0, stdout="", stderr="", workdir=tmp_path,
        outputs={"backbones": [str(pdb_path)]},
    )
    result = parse_output(_manifest(), run)
    assert result["num_backbones"] == 1
    assert result["backbones"][0] == {"id": "sample_0", "length": 2}


def test_parse_output_raises_when_no_backbones_collected():
    run = CompletedRun(returncode=0, stdout="", stderr="", workdir=Path("/tmp"), outputs={})
    with pytest.raises(ValueError, match="backbones"):
        parse_output(_manifest(), run)


def test_validation_requires_min_and_max_length():
    with pytest.raises(ToolInputError, match="min_length"):
        validate_and_fill(_manifest(), {"max_length": 70})


def test_validation_fills_defaults():
    params = validate_and_fill(_manifest(), {"min_length": 70, "max_length": 70})
    assert params["samples_per_length"] == 1
    assert params["num_timesteps"] == 100
    assert params["checkpoint_variant"] == "pdb"
    assert params["self_condition"] is True


def test_validation_rejects_unknown_checkpoint_variant():
    with pytest.raises(ToolInputError, match="checkpoint_variant"):
        validate_and_fill(
            _manifest(),
            {"min_length": 70, "max_length": 70, "checkpoint_variant": "bogus"},
        )


def test_validation_allows_self_condition_false_not_treated_as_missing():
    # Corner case: False must not be swallowed by the "if key in arguments"
    # check being confused with falsiness.
    params = validate_and_fill(
        _manifest(), {"min_length": 70, "max_length": 70, "self_condition": False}
    )
    assert params["self_condition"] is False


def test_validation_rejects_num_timesteps_of_zero():
    with pytest.raises(ToolInputError, match="num_timesteps"):
        validate_and_fill(
            _manifest(),
            {"min_length": 70, "max_length": 70, "num_timesteps": 0},
        )
