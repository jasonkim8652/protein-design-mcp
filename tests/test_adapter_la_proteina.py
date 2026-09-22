from pathlib import Path

import pytest

from protein_design_mcp.adapters.la_proteina import build_args, parse_output
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
    "lengths": [50],
    "num_samples": 2,
    "max_nsamples_per_batch": 2,
    "nsteps": 400,
    "self_cond": True,
    "sc_scale_noise": 0.1,
    "sc_scale_score": 1.0,
    "guidance_w": 1.0,
    "seed": 5,
}


def _manifest():
    return next(m for m in load_manifests(manifest_dir()) if m.name == "run_la_proteina")


def test_manifest_loads_and_requires_gpu():
    m = _manifest()
    assert m.category == "monomer_generation"
    assert m.requires.gpu is True


def test_manifest_hand_sets_mounts_to_repo_root():
    mounts = _manifest().engine.mounts
    assert "/home/jk661/projects/la-proteina" in mounts


def test_build_args_serializes_lengths_as_json():
    args = build_args(_manifest(), DEFAULT_PARAMS)
    idx = args.index("--lengths")
    assert args[idx + 1] == "[50]"


def test_build_args_encodes_self_cond_as_lowercase_string():
    args_true = build_args(_manifest(), {**DEFAULT_PARAMS, "self_cond": True})
    assert "--self-cond" in args_true
    assert args_true[args_true.index("--self-cond") + 1] == "true"
    args_false = build_args(_manifest(), {**DEFAULT_PARAMS, "self_cond": False})
    assert args_false[args_false.index("--self-cond") + 1] == "false"


def test_build_args_includes_every_numeric_knob():
    args = build_args(_manifest(), DEFAULT_PARAMS)
    for flag in (
        "--num-samples", "--max-nsamples-per-batch", "--nsteps",
        "--sc-scale-noise", "--sc-scale-score", "--guidance-w", "--seed",
    ):
        assert flag in args


def test_parse_output_counts_ca_atoms_per_backbone(tmp_path: Path):
    pdb_path = tmp_path / "job_0_n_50_id_0.pdb"
    pdb_path.write_text(MINIMAL_PDB)
    run = CompletedRun(
        returncode=0, stdout="", stderr="", workdir=tmp_path,
        outputs={"backbones": [str(pdb_path)]},
    )
    result = parse_output(_manifest(), run)
    assert result["num_backbones"] == 1
    assert result["backbones"][0] == {"id": "job_0_n_50_id_0", "length": 2}


def test_parse_output_raises_when_no_backbones_collected():
    run = CompletedRun(returncode=0, stdout="", stderr="", workdir=Path("/tmp"), outputs={})
    with pytest.raises(ValueError, match="backbones"):
        parse_output(_manifest(), run)


def test_validation_requires_lengths():
    with pytest.raises(ToolInputError, match="lengths"):
        validate_and_fill(_manifest(), {})


def test_validation_rejects_empty_lengths_list():
    with pytest.raises(ToolInputError, match="lengths"):
        validate_and_fill(_manifest(), {"lengths": []})


def test_validation_fills_defaults():
    params = validate_and_fill(_manifest(), {"lengths": [100]})
    assert params["num_samples"] == 2
    assert params["nsteps"] == 400
    assert params["self_cond"] is True
    assert params["sc_scale_noise"] == pytest.approx(0.1)
    assert params["guidance_w"] == pytest.approx(1.0)
    assert params["seed"] == 5


def test_validation_rejects_length_above_500():
    with pytest.raises(ToolInputError, match="lengths"):
        validate_and_fill(_manifest(), {"lengths": [900]})


def test_validation_allows_sc_scale_noise_of_zero():
    # Corner case: 0.0 (deterministic corrector) must not be treated as
    # missing/falsy.
    params = validate_and_fill(
        _manifest(), {"lengths": [100], "sc_scale_noise": 0.0}
    )
    assert params["sc_scale_noise"] == 0.0
