from pathlib import Path

import pytest

from protein_design_mcp.adapters.genie3_binder import build_args, parse_output
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
    "target_pdb": "target.pdb",
    "hotspot_residues": ["A19", "A76"],
    "binder_min_length": 60,
    "binder_max_length": 90,
    "num_samples": 2,
    "model_variant": "v1",
    "direction_scale": 1.0,
    "eta": 1.0,
    "n_sample_step": 100,
    "noise_scale": 1.0,
    "predict_sidechain": False,
    "seed": 0,
    "expand_interface": False,
    "interface_cutoff_angstrom": 6.0,
    "interface_rsa_threshold": 0.25,
    "interface_abs_sasa_threshold": 10.0,
}


def _manifest():
    return next(m for m in load_manifests(manifest_dir()) if m.name == "run_genie3_binder")


def test_manifest_loads_and_requires_gpu():
    m = _manifest()
    assert m.category == "binder_generation"
    assert m.requires.gpu is True


def test_manifest_validates_hotspot_pattern():
    spec = _manifest().schema["hotspot_residues"]
    import re
    pattern = re.compile(spec["items"]["pattern"])
    assert pattern.match("A19")
    assert pattern.match("A113A")  # with insertion code
    assert not pattern.match("19")  # missing chain
    assert not pattern.match("chainA19")  # multi-char malformed


def test_build_args_serializes_hotspots_as_json():
    args = build_args(_manifest(), DEFAULT_PARAMS)
    idx = args.index("--hotspot-residues")
    assert args[idx + 1] == '["A19", "A76"]'


def test_build_args_includes_binder_length_range():
    args = build_args(_manifest(), DEFAULT_PARAMS)
    assert "--binder-min-length" in args
    assert args[args.index("--binder-min-length") + 1] == "60"
    assert args[args.index("--binder-max-length") + 1] == "90"


def test_parse_output_reports_first_chain_as_binder_length(tmp_path: Path):
    # Corner case: the file holds the FULL COMPLEX (binder + target), not
    # just the binder -- "length" must count only the first chain (Genie 3
    # always writes the binder first), not every CA in the file.
    complex_pdb = (
        "ATOM      1  N   GLY A   1      11.104  13.207   2.145  1.00  0.00           N\n"
        "ATOM      2  CA  GLY A   1      12.560  13.207   2.145  1.00  0.00           C\n"
        "ATOM      3  CA  GLY A   2      13.560  13.207   2.145  1.00  0.00           C\n"
        "ATOM      4  N   GLY B   1      21.104  13.207   2.145  1.00  0.00           N\n"
        "ATOM      5  CA  GLY B   1      22.560  13.207   2.145  1.00  0.00           C\n"
        "TER\n"
    )
    pdb_path = tmp_path / "target_0.pdb"
    pdb_path.write_text(complex_pdb)
    run = CompletedRun(
        returncode=0, stdout="", stderr="", workdir=tmp_path,
        outputs={"binders": [str(pdb_path)]},
    )
    result = parse_output(_manifest(), run)
    assert result["num_binders"] == 1
    binder = result["binders"][0]
    assert binder["binder_chain_id"] == "A"
    assert binder["length"] == 2
    assert binder["chain_lengths"] == {"A": 2, "B": 1}


def test_parse_output_raises_when_no_binders_collected():
    run = CompletedRun(returncode=0, stdout="", stderr="", workdir=Path("/tmp"), outputs={})
    with pytest.raises(ValueError, match="binders"):
        parse_output(_manifest(), run)


def test_validation_requires_target_pdb_and_hotspots():
    with pytest.raises(ToolInputError, match="target_pdb"):
        validate_and_fill(_manifest(), {"hotspot_residues": ["A19"]})
    with pytest.raises(ToolInputError, match="hotspot_residues"):
        validate_and_fill(_manifest(), {"target_pdb": "t.pdb"})


def test_validation_rejects_malformed_hotspot_tag():
    with pytest.raises(ToolInputError, match="hotspot_residues"):
        validate_and_fill(
            _manifest(),
            {
                "target_pdb": "t.pdb",
                "hotspot_residues": ["not-a-tag"],
                "binder_min_length": 60,
                "binder_max_length": 90,
            },
        )


def test_validation_requires_binder_length_range():
    with pytest.raises(ToolInputError, match="binder_min_length"):
        validate_and_fill(
            _manifest(),
            {"target_pdb": "t.pdb", "hotspot_residues": ["A19"]},
        )


def test_validation_fills_defaults():
    params = validate_and_fill(
        _manifest(),
        {
            "target_pdb": "t.pdb",
            "hotspot_residues": ["A19"],
            "binder_min_length": 60,
            "binder_max_length": 90,
        },
    )
    assert params["num_samples"] == 2
    assert params["model_variant"] == "v1"
    assert params["predict_sidechain"] is False
    # expand_interface defaults to false: unchanged behavior for existing
    # callers who never asked for interface expansion.
    assert params["expand_interface"] is False
    assert params["interface_cutoff_angstrom"] == 6.0
    assert params["interface_rsa_threshold"] == 0.25
    assert params["interface_abs_sasa_threshold"] == 10.0


def test_manifest_runs_under_genie2_fixed_not_the_original_genie2():
    """The environment fix (wave-I): this tool's prefix must point at the
    Biopython-patched clone, not the original genie2 env, which still
    lacks Biopython (confirmed live, see the wave's report)."""
    assert _manifest().engine.prefix == "/home/jk661/.conda/envs/genie2_fixed"


def test_build_args_includes_interface_expansion_flags():
    args = build_args(_manifest(), DEFAULT_PARAMS)
    assert "--expand-interface" in args
    assert args[args.index("--expand-interface") + 1] == "false"
    assert "--interface-cutoff-angstrom" in args
    assert args[args.index("--interface-cutoff-angstrom") + 1] == "6.0"
    assert "--interface-rsa-threshold" in args
    assert args[args.index("--interface-rsa-threshold") + 1] == "0.25"
    assert "--interface-abs-sasa-threshold" in args
    assert args[args.index("--interface-abs-sasa-threshold") + 1] == "10.0"


def test_build_args_passes_expand_interface_true():
    params = dict(DEFAULT_PARAMS, expand_interface=True)
    args = build_args(_manifest(), params)
    assert args[args.index("--expand-interface") + 1] == "true"


def test_parse_output_reports_cond_strategy_from_interface_conditioning(tmp_path: Path):
    complex_pdb = (
        "ATOM      1  N   GLY A   1      11.104  13.207   2.145  1.00  0.00           N\n"
        "ATOM      2  CA  GLY A   1      12.560  13.207   2.145  1.00  0.00           C\n"
        "TER\n"
    )
    pdb_path = tmp_path / "target_0.pdb"
    pdb_path.write_text(complex_pdb)
    conditioning_path = tmp_path / "interface_conditioning.json"
    conditioning_path.write_text(
        '{"cond_strategy": "extended", "hotspot_residues": ["A19"], '
        '"extended_interface_residues": ["A18", "A19", "A20"]}'
    )
    run = CompletedRun(
        returncode=0,
        stdout="",
        stderr="",
        workdir=tmp_path,
        outputs={
            "binders": [str(pdb_path)],
            "interface_conditioning": str(conditioning_path),
        },
    )
    result = parse_output(_manifest(), run)
    assert result["cond_strategy"] == "extended"
    assert result["extended_interface_residues"] == ["A18", "A19", "A20"]


def test_parse_output_extended_interface_residues_null_when_not_expanded(tmp_path: Path):
    complex_pdb = (
        "ATOM      1  N   GLY A   1      11.104  13.207   2.145  1.00  0.00           N\n"
        "ATOM      2  CA  GLY A   1      12.560  13.207   2.145  1.00  0.00           C\n"
        "TER\n"
    )
    pdb_path = tmp_path / "target_0.pdb"
    pdb_path.write_text(complex_pdb)
    conditioning_path = tmp_path / "interface_conditioning.json"
    conditioning_path.write_text(
        '{"cond_strategy": "hotspot", "hotspot_residues": ["A19"], '
        '"extended_interface_residues": null}'
    )
    run = CompletedRun(
        returncode=0,
        stdout="",
        stderr="",
        workdir=tmp_path,
        outputs={
            "binders": [str(pdb_path)],
            "interface_conditioning": str(conditioning_path),
        },
    )
    result = parse_output(_manifest(), run)
    assert result["cond_strategy"] == "hotspot"
    assert result["extended_interface_residues"] is None
