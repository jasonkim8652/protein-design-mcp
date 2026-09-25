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


# --- molecular mechanics needs complete residues ----------------------------


def _pdb(tmp_path, lines):
    path = tmp_path / "s.pdb"
    path.write_text("\n".join(lines) + "\n")
    return path


def _atom(serial, name, resname, chain, resseq):
    return (f"ATOM  {serial:5d}  {name:<3s} {resname} {chain}{resseq:4d}"
            "       0.000   0.000   0.000  1.00  0.00           C")


def test_a_backbone_only_structure_is_refused(tmp_path):
    """OpenMM's `addHydrogens` needs every residue's full heavy-atom set. A
    backbone (N, CA, C, O) has none of the side chain, and OpenMM says so from
    deep inside Modeller, naming an index nobody chose:

        ValueError: HIS residue (118) has the wrong set of atoms

    A live round hit this by minimising `run_rebuild_backbone`'s output -- a
    reconstructed BACKBONE, correct for a sequence designer and unusable for
    molecular mechanics. Say that here, where the cause is still visible.
    """
    from protein_design_mcp.adapters.openmm_minimize import build_args
    from protein_design_mcp.validation import ToolInputError

    path = _pdb(tmp_path, [
        _atom(1, "N", "HIS", "A", 1), _atom(2, "CA", "HIS", "A", 1),
        _atom(3, "C", "HIS", "A", 1), _atom(4, "O", "HIS", "A", 1),
    ])
    with pytest.raises(ToolInputError) as excinfo:
        build_args(None, {"input_pdb": str(path), "max_iterations": 500,
                          "forcefield": "amber14"})
    message = str(excinfo.value)
    assert "HIS" in message
    assert "side chain" in message or "backbone" in message


def test_a_complete_residue_is_accepted(tmp_path):
    """A folded complex -- the intended input -- has every atom."""
    from protein_design_mcp.adapters.openmm_minimize import build_args

    names = ["N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"]
    path = _pdb(tmp_path, [
        _atom(i + 1, n, "HIS", "A", 1) for i, n in enumerate(names)
    ])
    assert str(path) in build_args(None, {
        "input_pdb": str(path), "max_iterations": 500, "forcefield": "amber14"})


def test_glycine_needs_no_side_chain(tmp_path):
    """GLY's complete heavy-atom set IS the backbone; refusing it would ban a
    real structure."""
    from protein_design_mcp.adapters.openmm_minimize import build_args

    path = _pdb(tmp_path, [
        _atom(1, "N", "GLY", "A", 1), _atom(2, "CA", "GLY", "A", 1),
        _atom(3, "C", "GLY", "A", 1), _atom(4, "O", "GLY", "A", 1),
    ])
    assert str(path) in build_args(None, {
        "input_pdb": str(path), "max_iterations": 500, "forcefield": "amber14"})


def test_an_unreadable_structure_is_left_to_the_engine(tmp_path):
    from protein_design_mcp.adapters.openmm_minimize import build_args

    missing = tmp_path / "nope.pdb"
    assert str(missing) in build_args(None, {
        "input_pdb": str(missing), "max_iterations": 500, "forcefield": "amber14"})
