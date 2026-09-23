from pathlib import Path

import pytest

from protein_design_mcp.adapters.openmm_minimize import build_args, parse_output
from protein_design_mcp.app import manifest_dir
from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.loader import load_manifests
from protein_design_mcp.validation import ToolInputError, validate_and_fill

SAMPLE_STDOUT = """\
initial_potential_energy_kj_mol: 15234.7812
final_potential_energy_kj_mol: -8811.2044
iterations: 500
output_pdb: minimized.pdb
"""


def _manifest():
    return next(
        m for m in load_manifests(manifest_dir()) if m.name == "run_openmm_minimize"
    )


def test_manifest_declares_its_output():
    (out,) = _manifest().outputs
    assert out.name == "minimized_pdb"
    assert out.pattern == "minimized.pdb"


def test_manifest_sets_its_own_timeout():
    assert _manifest().timeout_s == 1800


def test_build_args_names_the_workdir_relative_output():
    args = build_args(
        _manifest(),
        {"input_pdb": "/tmp/in.pdb", "max_iterations": 500,
         "forcefield": "amber14"},
    )
    assert "/tmp/in.pdb" in args
    assert "minimized.pdb" in args
    assert "500" in args


def test_parse_output_reports_the_energy_change():
    result = parse_output(
        _manifest(),
        CompletedRun(returncode=0, stdout=SAMPLE_STDOUT, stderr="",
                     workdir=Path("/tmp")),
    )
    assert result["initial_potential_energy_kj_mol"] == pytest.approx(15234.7812)
    assert result["final_potential_energy_kj_mol"] == pytest.approx(-8811.2044)
    assert result["energy_change_kj_mol"] == pytest.approx(-24045.9856)
    assert result["iterations"] == 500


def test_parse_output_raises_when_energies_are_absent():
    with pytest.raises(ValueError, match="energy"):
        parse_output(
            _manifest(),
            CompletedRun(returncode=0, stdout="nothing", stderr="",
                         workdir=Path("/tmp")),
        )


def test_validation_rejects_an_out_of_range_iteration_count():
    with pytest.raises(ToolInputError, match="max_iterations"):
        validate_and_fill(_manifest(), {"input_pdb": "in.pdb",
                                        "max_iterations": 0})


def test_validation_rejects_an_unknown_forcefield():
    with pytest.raises(ToolInputError, match="forcefield"):
        validate_and_fill(_manifest(), {"input_pdb": "in.pdb",
                                        "forcefield": "charmm99"})
