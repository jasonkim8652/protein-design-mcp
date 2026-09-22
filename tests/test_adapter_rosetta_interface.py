from pathlib import Path

import pytest

from protein_design_mcp.adapters.rosetta_interface import build_args, parse_output
from protein_design_mcp.app import manifest_dir
from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.loader import load_manifests
from protein_design_mcp.validation import ToolInputError, validate_and_fill

MANIFEST_DIR = manifest_dir()

# A real result this wrapper produced, live, 2026-09-22, running the EXACT
# code in scripts/engines/rosetta_interface.py under a working (but
# differently-versioned) PyRosetta install found on this host
# (~/.conda/envs/BindCraft) over tests/fixtures/test_pdbs/1BRS.pdb (the
# barnase-barstar complex), interface "A_D" -- see the manifest's
# "Verification status" section and the wave report for why THIS tool's own
# (cp312) environment cannot run it.
SAMPLE_STDOUT = (
    '{"dG": 210.1646596282025, "dSASA": 1573.9739508012663, '
    '"shape_complementarity": 0.7199670970439911, "interface_hbonds": 13, '
    '"delta_unsat_hbonds": 9, "num_interface_residues": 67, '
    '"packstat": 0.536864567830277, "interface": "A_D", '
    '"score_function": "ref2015"}'
)


def _manifest():
    return next(m for m in load_manifests(MANIFEST_DIR) if m.name == "run_rosetta_interface")


def test_manifest_loads_and_is_not_composite():
    m = _manifest()
    assert m.category == "scoring"
    assert m.composite is False
    assert m.requires.gpu is False


def test_manifest_documents_verification_status():
    doc = _manifest().doc
    assert "## Verification status" in doc
    assert "NOT VERIFIED LIVE" in _manifest().summary


def test_build_args_serialises_all_job_fields():
    import json

    params = {
        "complex_pdb": "/tmp/c.pdb",
        "interface": "A_B",
        "score_function": "ref2015",
        "pack_separated": True,
        "pack_input": False,
        "pack_rounds": 1,
        "compute_packstat": True,
        "compute_interface_sc": True,
    }
    args = build_args(_manifest(), params)
    assert len(args) == 1
    assert json.loads(args[0]) == params


def test_parse_output_extracts_all_headline_fields():
    result = parse_output(
        _manifest(),
        CompletedRun(returncode=0, stdout=SAMPLE_STDOUT, stderr="", workdir=Path("/tmp")),
    )
    assert result["dG"] == pytest.approx(210.1646596282025)
    assert result["dSASA"] == pytest.approx(1573.9739508012663)
    assert result["shape_complementarity"] == pytest.approx(0.7199670970439911)
    assert result["interface_hbonds"] == 13
    assert result["delta_unsat_hbonds"] == 9
    assert result["num_interface_residues"] == 67
    assert result["packstat"] == pytest.approx(0.536864567830277)


def test_parse_output_reads_the_last_stdout_line_not_the_first():
    """PyRosetta is chatty on stdout even with -mute all -- the adapter must
    not be confused by noise preceding the real JSON line."""
    stdout = f"core.init: some PyRosetta banner line\n{SAMPLE_STDOUT}\n"
    result = parse_output(
        _manifest(),
        CompletedRun(returncode=0, stdout=stdout, stderr="", workdir=Path("/tmp")),
    )
    assert result["dG"] == pytest.approx(210.1646596282025)


def test_parse_output_raises_when_stdout_is_empty():
    with pytest.raises(ValueError, match="no output"):
        parse_output(
            _manifest(),
            CompletedRun(returncode=0, stdout="", stderr="some crash", workdir=Path("/tmp")),
        )


def test_parse_output_raises_when_stdout_is_not_json():
    with pytest.raises(ValueError, match="not valid JSON"):
        parse_output(
            _manifest(),
            CompletedRun(returncode=0, stdout="not json", stderr="", workdir=Path("/tmp")),
        )


def test_parse_output_raises_when_a_required_field_is_missing():
    import json

    stdout = json.dumps({"dG": 1.0})
    with pytest.raises(ValueError, match="dSASA"):
        parse_output(
            _manifest(),
            CompletedRun(returncode=0, stdout=stdout, stderr="", workdir=Path("/tmp")),
        )


# --- Corner cases required by CLAUDE.md's TDD workflow ---------------------


def test_parse_output_handles_zero_valued_dG():
    """dG = 0 must survive -- 0 is a valid float and must not be treated as
    falsy/absent."""
    import json

    stdout = json.dumps(
        {
            "dG": 0.0,
            "dSASA": 0.0,
            "shape_complementarity": None,
            "interface_hbonds": 0,
            "delta_unsat_hbonds": 0,
            "num_interface_residues": 0,
        }
    )
    result = parse_output(
        _manifest(),
        CompletedRun(returncode=0, stdout=stdout, stderr="", workdir=Path("/tmp")),
    )
    assert result["dG"] == 0.0
    assert result["interface_hbonds"] == 0
    assert result["num_interface_residues"] == 0


def test_parse_output_carries_explicit_none_for_disabled_optional_metrics():
    """shape_complementarity/packstat = null (compute_*=false) must survive
    as explicit None, distinct from the key being missing entirely."""
    import json

    stdout = json.dumps(
        {
            "dG": -50.0,
            "dSASA": 1500.0,
            "shape_complementarity": None,
            "interface_hbonds": 5,
            "delta_unsat_hbonds": 1,
            "num_interface_residues": 20,
            "packstat": None,
        }
    )
    result = parse_output(
        _manifest(),
        CompletedRun(returncode=0, stdout=stdout, stderr="", workdir=Path("/tmp")),
    )
    assert result["shape_complementarity"] is None
    assert result["packstat"] is None


def test_parse_output_missing_optional_packstat_key_defaults_to_none():
    """packstat is genuinely optional in the JSON (older/partial output) --
    must fall back to None via .get(), not raise a KeyError."""
    import json

    stdout = json.dumps(
        {
            "dG": 1.0,
            "dSASA": 2.0,
            "shape_complementarity": 0.5,
            "interface_hbonds": 1,
            "delta_unsat_hbonds": 0,
            "num_interface_residues": 1,
        }
    )
    result = parse_output(
        _manifest(),
        CompletedRun(returncode=0, stdout=stdout, stderr="", workdir=Path("/tmp")),
    )
    assert result["packstat"] is None


def test_validation_rejects_missing_interface():
    with pytest.raises(ToolInputError, match="interface"):
        validate_and_fill(_manifest(), {"complex_pdb": "c.pdb"})


def test_validation_rejects_an_interface_without_an_underscore():
    with pytest.raises(ToolInputError, match="interface"):
        validate_and_fill(_manifest(), {"complex_pdb": "c.pdb", "interface": "AB"})


def test_validation_accepts_multi_chain_group_interface():
    params = validate_and_fill(
        _manifest(), {"complex_pdb": "c.pdb", "interface": "AB_HL"}
    )
    assert params["interface"] == "AB_HL"


def test_validation_fills_defaults():
    params = validate_and_fill(_manifest(), {"complex_pdb": "c.pdb", "interface": "A_B"})
    assert params["score_function"] == "ref2015"
    assert params["pack_separated"] is True
    assert params["pack_input"] is False
    assert params["pack_rounds"] == 1
    assert params["compute_packstat"] is True
    assert params["compute_interface_sc"] is True


def test_validation_rejects_pack_rounds_zero():
    with pytest.raises(ToolInputError, match="pack_rounds"):
        validate_and_fill(
            _manifest(),
            {"complex_pdb": "c.pdb", "interface": "A_B", "pack_rounds": 0},
        )


def test_validation_accepts_pack_rounds_at_the_boundary():
    params = validate_and_fill(
        _manifest(),
        {"complex_pdb": "c.pdb", "interface": "A_B", "pack_rounds": 20},
    )
    assert params["pack_rounds"] == 20
