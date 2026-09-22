import json
from pathlib import Path

import pytest

from protein_design_mcp.adapters.protpardelle import build_args, parse_output
from protein_design_mcp.app import manifest_dir
from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.loader import load_manifests
from protein_design_mcp.validation import ToolInputError, validate_and_fill

TWO_CHAIN_PDB = (
    "ATOM      1  N   GLY A   1      11.104  13.207   2.145  1.00  0.00           N\n"
    "ATOM      2  CA  GLY A   1      12.560  13.207   2.145  1.00  0.00           C\n"
    "ATOM      3  C   GLY A   1      13.090  14.630   2.145  1.00  0.00           C\n"
    "ATOM      4  O   GLY A   1      12.400  15.630   2.145  1.00  0.00           O\n"
    "TER\n"
    "ATOM      5  N   GLY B   1      21.104  13.207   2.145  1.00  0.00           N\n"
    "ATOM      6  CA  GLY B   1      22.560  13.207   2.145  1.00  0.00           C\n"
    "ATOM      7  C   GLY B   1      23.090  14.630   2.145  1.00  0.00           C\n"
    "ATOM      8  O   GLY B   1      22.400  15.630   2.145  1.00  0.00           O\n"
    "ATOM      9  N   GLY B   2      24.400  14.700   2.200  1.00  0.00           N\n"
    "ATOM     10  CA  GLY B   2      25.100  16.000   2.300  1.00  0.00           C\n"
    "ATOM     11  C   GLY B   2      26.100  16.500   2.300  1.00  0.00           C\n"
    "ATOM     12  O   GLY B   2      27.100  16.000   2.300  1.00  0.00           O\n"
    "TER\n"
)

DEFAULT_PARAMS = {
    "target_pdb": "target.pdb",
    "contig": "A1-128;/;120-120",
    "total_lengths": [[128, 128], [120, 120]],
    "hotspots": ["A19", "A76"],
    "model": "cc83",
    "step_scale": 1.2,
    "schurn": 200.0,
    "crop_cond_start": 0.0,
    "translation": [0.0, 0.0, 0.0],
    "num_samples": 2,
    "batch_size": 8,
}


def _manifest():
    return next(m for m in load_manifests(manifest_dir()) if m.name == "run_protpardelle")


def test_manifest_loads_and_requires_gpu():
    m = _manifest()
    assert m.category == "binder_generation"
    assert m.requires.gpu is True


def test_manifest_validates_contig_pattern():
    import re
    pattern = re.compile(_manifest().schema["contig"]["pattern"])
    assert pattern.match("A1-128;/;120-120")
    assert pattern.match("0-20;A1-50;0-20;/;100-100")
    assert pattern.match("A1-79;/;B1-141;/;70-150")
    assert not pattern.match("A1-128 / 120-120")  # space-separated, not ';/;'
    assert not pattern.match("")


def test_build_args_serializes_total_lengths_and_hotspots():
    args = build_args(_manifest(), DEFAULT_PARAMS)
    assert json.loads(args[args.index("--total-lengths") + 1]) == [[128, 128], [120, 120]]
    assert json.loads(args[args.index("--hotspots") + 1]) == ["A19", "A76"]


def test_build_args_passes_null_hotspots_through():
    args = build_args(_manifest(), {**DEFAULT_PARAMS, "hotspots": None})
    assert json.loads(args[args.index("--hotspots") + 1]) is None


def test_build_args_omits_seed_when_not_given():
    args = build_args(_manifest(), DEFAULT_PARAMS)
    assert "--seed" not in args


def test_build_args_includes_seed_when_given():
    args = build_args(_manifest(), {**DEFAULT_PARAMS, "seed": 37})
    assert args[args.index("--seed") + 1] == "37"


def test_build_args_rejects_malformed_hotspot_tag():
    with pytest.raises(ValueError, match="malformed tag"):
        build_args(_manifest(), {**DEFAULT_PARAMS, "hotspots": ["not-a-tag"]})


def test_build_args_rejects_total_lengths_chain_count_mismatch():
    with pytest.raises(ValueError, match="contig"):
        build_args(_manifest(), {**DEFAULT_PARAMS, "total_lengths": [[128, 128]]})


def test_build_args_rejects_total_lengths_min_greater_than_max():
    with pytest.raises(ValueError, match="min <= max"):
        build_args(
            _manifest(),
            {**DEFAULT_PARAMS, "total_lengths": [[128, 128], [200, 100]]},
        )


def test_parse_output_reports_every_chain(tmp_path: Path):
    pdb_path = tmp_path / "sample_0.pdb"
    pdb_path.write_text(TWO_CHAIN_PDB)
    run = CompletedRun(
        returncode=0, stdout="", stderr="", workdir=tmp_path,
        outputs={"samples": [str(pdb_path)]},
    )
    result = parse_output(_manifest(), run)
    assert result["num_samples"] == 1
    chains = {c["chain_id"]: c["length"] for c in result["samples"][0]["chains"]}
    assert chains == {"A": 1, "B": 2}


def test_parse_output_raises_when_no_samples_collected():
    run = CompletedRun(returncode=0, stdout="", stderr="", workdir=Path("/tmp"), outputs={})
    with pytest.raises(ValueError, match="samples"):
        parse_output(_manifest(), run)


def test_validation_requires_target_pdb_contig_total_lengths_hotspots():
    with pytest.raises(ToolInputError):
        validate_and_fill(_manifest(), {})


def test_validation_rejects_malformed_contig_at_schema_level():
    with pytest.raises(ToolInputError, match="contig"):
        validate_and_fill(
            _manifest(),
            {
                "target_pdb": "t.pdb",
                "contig": "A1-128 / 120-120",
                "total_lengths": [[128, 128], [120, 120]],
                "hotspots": None,
            },
        )


def test_validation_fills_defaults():
    params = validate_and_fill(
        _manifest(),
        {
            "target_pdb": "t.pdb",
            "contig": "A1-128;/;120-120",
            "total_lengths": [[128, 128], [120, 120]],
            "hotspots": None,
        },
    )
    assert params["model"] == "cc83"
    assert params["step_scale"] == pytest.approx(1.2)
    assert params["schurn"] == pytest.approx(200.0)
    assert params.get("seed") is None
    assert params["translation"] == [0.0, 0.0, 0.0]


def test_validation_accepts_explicit_null_hotspots():
    # Corner case: None is a legitimate, required-to-be-stated value here,
    # not "missing".
    params = validate_and_fill(
        _manifest(),
        {
            "target_pdb": "t.pdb",
            "contig": "A1-128;/;120-120",
            "total_lengths": [[128, 128], [120, 120]],
            "hotspots": None,
        },
    )
    assert params["hotspots"] is None
