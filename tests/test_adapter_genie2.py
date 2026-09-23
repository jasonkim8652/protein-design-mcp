from pathlib import Path

import pytest

from protein_design_mcp.adapters.genie2 import build_args, parse_output
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


def _manifest():
    return next(m for m in load_manifests(manifest_dir()) if m.name == "run_genie2")


def test_manifest_loads_and_requires_gpu():
    m = _manifest()
    assert m.category == "monomer_generation"
    assert m.requires.gpu is True
    assert m.timeout_s == 7200


def test_manifest_uses_a_mounted_prefix():
    assert _manifest().engine.prefix.endswith("/genie2")


def test_build_args_uses_fixed_checkpoint_and_relative_outdir():
    args = build_args(
        _manifest(),
        {
            "min_length": 50,
            "max_length": 50,
            "length_step": 10,
            "num_samples": 2,
            "batch_size": 4,
            "scale": 0.6,
            "sequential_order": False,
        },
    )
    assert "--name" in args and "base" in args
    assert "--epoch" in args and "40" in args
    assert "--outdir" in args
    assert args[args.index("--outdir") + 1] == "output"
    assert "--sequential_order" not in args


def test_build_args_appends_sequential_order_flag_when_true():
    args = build_args(
        _manifest(),
        {
            "min_length": 50,
            "max_length": 50,
            "length_step": 10,
            "num_samples": 2,
            "batch_size": 4,
            "scale": 0.6,
            "sequential_order": True,
        },
    )
    assert "--sequential_order" in args


def test_parse_output_counts_ca_atoms_per_backbone(tmp_path: Path):
    pdb_path = tmp_path / "50_0.pdb"
    pdb_path.write_text(MINIMAL_PDB)
    run = CompletedRun(
        returncode=0,
        stdout="",
        stderr="",
        workdir=tmp_path,
        outputs={"backbones": [str(pdb_path)]},
    )
    result = parse_output(_manifest(), run)
    assert result["num_backbones"] == 1
    assert result["backbones"][0] == {"id": "50_0", "length": 2}


def test_parse_output_raises_when_no_backbones_collected():
    run = CompletedRun(returncode=0, stdout="", stderr="", workdir=Path("/tmp"), outputs={})
    with pytest.raises(ValueError, match="backbones"):
        parse_output(_manifest(), run)


def test_parse_output_handles_single_string_output_not_just_list(tmp_path: Path):
    # Corner case: multiple=True specs are documented to always return a
    # list, but the adapter tolerates a bare string defensively too.
    pdb_path = tmp_path / "50_0.pdb"
    pdb_path.write_text(MINIMAL_PDB)
    run = CompletedRun(
        returncode=0, stdout="", stderr="", workdir=tmp_path,
        outputs={"backbones": str(pdb_path)},
    )
    result = parse_output(_manifest(), run)
    assert result["num_backbones"] == 1


def test_validation_requires_min_and_max_length():
    with pytest.raises(ToolInputError, match="min_length"):
        validate_and_fill(_manifest(), {"max_length": 50})


def test_validation_fills_defaults():
    params = validate_and_fill(_manifest(), {"min_length": 50, "max_length": 50})
    assert params["num_samples"] == 2
    assert params["batch_size"] == 4
    assert params["scale"] == pytest.approx(0.6)
    assert params["length_step"] == 10
    assert params["sequential_order"] is False


def test_validation_rejects_scale_above_one():
    with pytest.raises(ToolInputError, match="scale"):
        validate_and_fill(
            _manifest(), {"min_length": 50, "max_length": 50, "scale": 1.5}
        )


def test_validation_rejects_scale_of_exactly_zero_is_allowed():
    # Corner case: 0 is a legitimate (deterministic) value, must not be
    # rejected as falsy.
    params = validate_and_fill(
        _manifest(), {"min_length": 50, "max_length": 50, "scale": 0.0}
    )
    assert params["scale"] == 0.0


def test_validation_rejects_length_step_of_zero():
    with pytest.raises(ToolInputError, match="length_step"):
        validate_and_fill(
            _manifest(),
            {"min_length": 50, "max_length": 60, "length_step": 0},
        )
