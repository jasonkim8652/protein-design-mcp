from pathlib import Path

import pytest

from protein_design_mcp.adapters.genie3_scaffold import build_args, parse_output
from protein_design_mcp.app import manifest_dir
from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.loader import load_manifests
from protein_design_mcp.validation import ToolInputError, validate_and_fill

MINIMAL_PDB = (
    "ATOM      1  N   GLY A   1      11.104  13.207   2.145  1.00  0.00           N\n"
    "ATOM      2  CA  GLY A   1      12.560  13.207   2.145  1.00  0.00           C\n"
    "ATOM      3  C   GLY A   1      13.090  14.630   2.145  1.00  0.00           C\n"
    "TER\n"
)

DEFAULT_PARAMS = {
    "min_length": 100,
    "max_length": 100,
    "length_step": 50,
    "num_samples": 2,
    "batch_size": 4,
    "model_variant": "v1",
    "direction_scale": 1.0,
    "eta": 1.0,
    "n_sample_step": 100,
    "noise_scale": 1.0,
    "predict_sidechain": False,
    "seed": 0,
}


def _manifest():
    return next(m for m in load_manifests(manifest_dir()) if m.name == "run_genie3_scaffold")


def test_manifest_loads_and_requires_gpu():
    m = _manifest()
    assert m.category == "monomer_generation"
    assert m.requires.gpu is True


def test_manifest_uses_genie2_prefix():
    assert _manifest().engine.prefix.endswith("/genie2")


def test_manifest_sets_pythonpath_for_deps_and_src():
    pythonpath = _manifest().engine.env_vars.get("PYTHONPATH", "")
    assert "/home/jk661/projects/genie3/.deps" in pythonpath
    assert "/home/jk661/projects/genie3/src" in pythonpath


def test_build_args_passes_model_variant():
    args = build_args(_manifest(), DEFAULT_PARAMS)
    assert args[args.index("--model-variant") + 1] == "v1"


def test_build_args_encodes_predict_sidechain_as_lowercase_string():
    args = build_args(_manifest(), {**DEFAULT_PARAMS, "predict_sidechain": True})
    assert args[args.index("--predict-sidechain") + 1] == "true"


def test_parse_output_counts_ca_atoms_per_backbone(tmp_path: Path):
    pdb_path = tmp_path / "sample_0.pdb"
    pdb_path.write_text(MINIMAL_PDB)
    run = CompletedRun(
        returncode=0, stdout="", stderr="", workdir=tmp_path,
        outputs={"backbones": [str(pdb_path)]},
    )
    result = parse_output(_manifest(), run)
    assert result["num_backbones"] == 1
    assert result["backbones"][0]["length"] == 1


def test_parse_output_raises_when_no_backbones_collected():
    run = CompletedRun(returncode=0, stdout="", stderr="", workdir=Path("/tmp"), outputs={})
    with pytest.raises(ValueError, match="backbones"):
        parse_output(_manifest(), run)


def test_validation_requires_min_and_max_length():
    with pytest.raises(ToolInputError, match="min_length"):
        validate_and_fill(_manifest(), {"max_length": 100})


def test_validation_fills_defaults():
    params = validate_and_fill(_manifest(), {"min_length": 100, "max_length": 100})
    assert params["model_variant"] == "v1"
    assert params["n_sample_step"] == 100
    assert params["predict_sidechain"] is False


def test_validation_rejects_unknown_model_variant():
    with pytest.raises(ToolInputError, match="model_variant"):
        validate_and_fill(
            _manifest(),
            {"min_length": 100, "max_length": 100, "model_variant": "v2"},
        )


def test_validation_allows_eta_of_zero():
    # Corner case: 0.0 (fully deterministic DDIM) must not be treated as
    # missing/falsy.
    params = validate_and_fill(
        _manifest(), {"min_length": 100, "max_length": 100, "eta": 0.0}
    )
    assert params["eta"] == 0.0
