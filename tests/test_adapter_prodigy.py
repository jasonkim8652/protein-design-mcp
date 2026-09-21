import pytest

from protein_design_mcp.adapters.prodigy import build_args, parse_output
from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.loader import load_manifests
from protein_design_mcp.validation import ToolInputError, validate_and_fill
from pathlib import Path

MANIFEST_DIR = Path(__file__).resolve().parents[1] / "manifests"

SAMPLE_STDOUT = """\
[+] Reading structure file: /tmp/complex.pdb
[+] Parsed structure file complex (2 chains, 210 residues)
[+] No. of intermolecular contacts: 72
[+] Predicted binding affinity (kcal.mol-1): -11.30
[+] Predicted dissociation constant (M) at 25.0C: 5.2e-09
"""


def _manifest():
    return next(m for m in load_manifests(MANIFEST_DIR) if m.name == "run_prodigy")


def test_manifest_loads_and_is_not_composite():
    m = _manifest()
    assert m.category == "scoring"
    assert m.composite is False
    assert m.requires.gpu is False


def test_manifest_documents_what_it_needs():
    doc = _manifest().doc
    assert "## What this is" in doc
    assert "## What you must supply" in doc


def test_build_args_maps_chains_to_two_flags():
    args = build_args({"complex_pdb": "/tmp/c.pdb", "chain_a": "A", "chain_b": "B",
                       "temperature": 25.0})
    assert "/tmp/c.pdb" in args
    assert "--selection" in args
    assert "A" in args and "B" in args


def test_build_args_includes_temperature():
    args = build_args({"complex_pdb": "/tmp/c.pdb", "chain_a": "A", "chain_b": "B",
                       "temperature": 37.0})
    assert "--temperature" in args
    assert "37.0" in args


def test_parse_output_extracts_affinity_and_kd():
    result = parse_output(
        CompletedRun(returncode=0, stdout=SAMPLE_STDOUT, stderr="", workdir=Path("/tmp"))
    )
    assert result["binding_affinity_kcal_per_mol"] == pytest.approx(-11.30)
    assert result["dissociation_constant_M"] == pytest.approx(5.2e-09)
    assert result["intermolecular_contacts"] == 72


def test_parse_output_carries_the_calibration_warning():
    result = parse_output(
        CompletedRun(returncode=0, stdout=SAMPLE_STDOUT, stderr="", workdir=Path("/tmp"))
    )
    assert "de novo" in result["caveat"]


def test_parse_output_raises_when_affinity_is_absent():
    with pytest.raises(ValueError, match="affinity"):
        parse_output(
            CompletedRun(returncode=0, stdout="nothing useful", stderr="",
                         workdir=Path("/tmp"))
        )


def test_validation_rejects_a_non_structure_path():
    with pytest.raises(ToolInputError, match="complex_pdb"):
        validate_and_fill(_manifest(), {"complex_pdb": "notes.txt",
                                        "chain_a": "A", "chain_b": "B"})


def test_validation_rejects_a_multi_character_chain_id():
    with pytest.raises(ToolInputError, match="chain_a"):
        validate_and_fill(_manifest(), {"complex_pdb": "c.pdb",
                                        "chain_a": "AB", "chain_b": "B"})


def test_validation_fills_the_default_temperature():
    params = validate_and_fill(_manifest(), {"complex_pdb": "c.pdb",
                                             "chain_a": "A", "chain_b": "B"})
    assert params["temperature"] == 25.0
