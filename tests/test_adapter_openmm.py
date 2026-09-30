from pathlib import Path
import gzip

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
    out = next(o for o in _manifest().outputs if o.name == "minimized_pdb")
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


def test_parse_output_preserves_actual_physical_settings():
    result = parse_output(_manifest(), CompletedRun(
        returncode=0, stdout=SAMPLE_STDOUT + "\nforce_field: amber14-all.xml\nsolvent_model: none\nenergy_units: kJ/mol\n",
        stderr="", workdir=Path("/tmp")))
    assert result["force_field"] == "amber14-all.xml"
    assert result["solvent_model"] == "none"
    assert result["energy_units"] == "kJ/mol"


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


@pytest.mark.parametrize("suffix", [".pdb", ".cif", ".mmcif", ".pdb.gz", ".cif.gz", ".mmcif.gz"])
def test_schema_accepts_native_structure_formats(suffix):
    params = validate_and_fill(_manifest(), {"input_pdb": "folded" + suffix})
    assert params["input_pdb"] == "folded" + suffix
    assert params["forcefield"] == "amber14" and params["max_iterations"] == 500


def _cif_from_pdb(source, destination):
    from Bio.PDB import MMCIFIO, PDBParser
    writer = MMCIFIO()
    writer.set_structure(PDBParser(QUIET=True).get_structure("folded", str(source)))
    writer.save(str(destination))


@pytest.mark.parametrize("suffix", [".cif", ".mmcif", ".cif.gz", ".mmcif.gz"])
@pytest.mark.parametrize("complete", [True, False])
def test_cif_preflight_accepts_folded_residues_and_refuses_backbone_only(tmp_path, suffix, complete):
    names = ["N", "CA", "C", "O"]
    if complete:
        names.extend(["CB", "CG", "ND1", "CD2", "CE1", "NE2"])
    pdb = _pdb(tmp_path, [_atom(i + 1, name, "HIS", "B", 5) for i, name in enumerate(names)])
    cif = tmp_path / ("structure" + suffix)
    _cif_from_pdb(pdb, cif)
    if suffix.endswith(".gz"):
        cif.write_bytes(gzip.compress(cif.read_bytes()))
    original = cif.read_bytes()
    params = {"input_pdb": str(cif), "max_iterations": 17, "forcefield": "charmm36"}
    if complete:
        assert build_args(None, params) == [str(cif), "minimized.pdb", "--max-iterations", "17", "--forcefield", "charmm36", "--platform", "CUDA", "--precision", "double"]
    else:
        with pytest.raises(ToolInputError, match="HIS B5.*backbone atoms only"):
            build_args(None, params)
    assert cif.read_bytes() == original


def test_runtime_arguments_are_explicit_and_validated():
    params = validate_and_fill(_manifest(), {'input_pdb': '/tmp/in.pdb'})
    assert params['platform'] == 'CUDA'
    assert params['precision'] == 'double'
    args = build_args(_manifest(), params)
    assert args[args.index('--platform')+1] == 'CUDA'
    assert args[args.index('--precision')+1] == 'double'


def test_parse_full_scientific_notation_and_reject_nonfinite():
    run = CompletedRun(returncode=0, stdout='initial_potential_energy_kj_mol: 1.2e+20\nfinal_potential_energy_kj_mol: -3.4e+3\n', stderr='', workdir=Path('/tmp'))
    assert parse_output(_manifest(), run)['initial_potential_energy_kj_mol'] == 1.2e20
    with pytest.raises(ValueError, match='finite'):
        parse_output(_manifest(), CompletedRun(returncode=0, stdout='initial_potential_energy_kj_mol: inf\nfinal_potential_energy_kj_mol: nan\n', stderr='', workdir=Path('/tmp')))


def test_diagnostics_preserve_failed_geometry_and_actual_iterations(tmp_path):
    import json
    metadata = {'protocol': 'soft-repulsion-flexible-hbonds-v1', 'status': 'geometry_failed',
                'geometry_passed': False, 'iterations': 1400, 'stages': [
                    {'platform': 'CUDA', 'platform_properties': {'Precision': 'double'}, 'reporter_calls': 1400}]}
    (tmp_path/'openmm_diagnostics.json').write_text(json.dumps(metadata))
    minimized = tmp_path/'minimized.pdb'; minimized.write_text('ATOM\n')
    result = parse_output(_manifest(), CompletedRun(returncode=0, stdout=SAMPLE_STDOUT, stderr='', workdir=tmp_path, outputs={'minimized_pdb': str(minimized)}))
    assert result['geometry_passed'] is False
    assert result['iterations'] == 1400
    assert result['minimization_diagnostics'] == metadata
    assert result['platform'] == 'CUDA'
    assert result['precision'] == 'double'


def test_explicit_numerical_failure_returns_missing_measurement(tmp_path):
    import json
    metadata = {'protocol': 'soft-repulsion-flexible-hbonds-v1', 'status': 'numerical_failure',
                'requested_platform': 'CUDA', 'requested_precision': 'double',
                'error': {'type': 'NumericalFailure', 'message': 'Nonfinite coordinates'}}
    (tmp_path/'openmm_diagnostics.json').write_text(json.dumps(metadata))
    result = parse_output(_manifest(), CompletedRun(returncode=0, stdout='', stderr='', workdir=tmp_path))
    assert result['numerical_failure'] is True
    assert result['final_potential_energy_kj_mol'] is None
    assert result['geometry_passed'] is False


def test_diagnostics_are_read_from_collected_output_after_workdir_cleanup(tmp_path):
    import json
    path = tmp_path/'collected.json'
    path.write_text(json.dumps({'protocol': 'soft-repulsion-flexible-hbonds-v1', 'geometry_passed': True, 'iterations': 99, 'stages': []}))
    result = parse_output(_manifest(), CompletedRun(returncode=0, stdout=SAMPLE_STDOUT, stderr='', workdir=tmp_path/'deleted', outputs={'minimization_diagnostics': str(path)}))
    assert result['minimization_protocol'] == 'soft-repulsion-flexible-hbonds-v1'
    assert result['iterations'] == 99


def test_numerical_failure_survives_real_collection_without_structure(tmp_path, monkeypatch):
    import json
    import shutil
    from protein_design_mcp.results import collect_outputs
    monkeypatch.setenv('PROTEIN_MCP_RESULTS_DIR', str(tmp_path/'results'))
    work = tmp_path/'scratch'; work.mkdir()
    (work/'openmm_diagnostics.json').write_text(json.dumps({'protocol': 'soft-repulsion-flexible-hbonds-v1', 'status': 'numerical_failure', 'requested_platform': 'CUDA', 'requested_precision': 'double'}))
    outputs = collect_outputs(_manifest().outputs, work, 'numeric')
    shutil.rmtree(work)
    result = parse_output(_manifest(), CompletedRun(returncode=0, stdout='', stderr='', workdir=work, outputs=outputs))
    assert result['numerical_failure'] is True
    assert 'minimized_pdb' not in outputs


def test_completed_diagnostics_require_collected_minimized_structure(tmp_path):
    import json
    path = tmp_path/'diagnostics.json'
    path.write_text(json.dumps({'status': 'completed', 'stages': [], 'geometry_passed': True}))
    with pytest.raises(ValueError, match='minimized'):
        parse_output(_manifest(), CompletedRun(returncode=0, stdout=SAMPLE_STDOUT, stderr='', workdir=tmp_path, outputs={'minimization_diagnostics': str(path)}))


@pytest.mark.parametrize('platform,requested,expected', [
    ('CPU', 'double', 'platform_default'),
    ('CPU', 'mixed', 'platform_default'),
    ('Reference', 'mixed', 'double'),
    ('Reference', 'double', 'double'),
    ('CUDA', 'mixed', 'mixed'),
    ('CUDA', 'double', 'double'),
])
@pytest.mark.parametrize('has_stage', [False, True])
def test_numerical_failure_reports_effective_native_precision(tmp_path, platform, requested, expected, has_stage):
    import json
    stage = {'platform': platform, 'platform_properties': {'Precision': requested} if platform == 'CUDA' else {}, 'reporter_calls': 2}
    metadata = {'protocol': 'soft-repulsion-flexible-hbonds-v1', 'status': 'numerical_failure',
                'requested_platform': platform, 'requested_precision': requested,
                'stages': [stage] if has_stage else []}
    path = tmp_path/'diagnostics.json'
    path.write_text(json.dumps(metadata))
    result = parse_output(_manifest(), CompletedRun(returncode=0, stdout='', stderr='', workdir=tmp_path/'deleted', outputs={'minimization_diagnostics': str(path)}))
    assert result['platform'] == platform
    assert result['precision'] == expected
    assert result['numerical_failure'] is True
    assert result['minimization_diagnostics']['requested_precision'] == requested
