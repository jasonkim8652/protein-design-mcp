"""A BoltzGen design spec is built from parameters, not demanded as a file.

Every other tool on this server takes typed parameters through its MCP schema
and the model fills them. The six run_boltzgen_* tools instead required a
`design_spec` YAML, and nothing here produced one -- so they could not be
reached from a planned workflow at all. A round planned
`run_epitope_scan -> run_rfdiffusion3_binder -> run_mpnn -> run_boltzgen_fold`,
got through three steps, and the fourth refused: "design_spec is required but
was not provided", with nothing upstream that could have supplied one.

Documenting it as caller-authored justified the inconsistency instead of
removing it. The spec's content is structured parameters -- which chain is
designed, how long, which target chains to condition on -- so the tool builds
it, exactly as every other tool builds its own engine invocation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "engines"))

from boltzgen_design import build_design_spec  # noqa: E402


def test_a_binder_against_one_target_chain():
    spec = build_design_spec(target_structure="/w/1g13.cif", target_chains=["A"],
                             binder_length_min=80, binder_length_max=140,
                             binder_chain_id="C")
    assert spec == {"entities": [
        {"protein": {"id": "C", "sequence": "80..140"}},
        {"file": {"path": "/w/1g13.cif", "include": [{"chain": {"id": "A"}}]}},
    ]}


def test_a_fixed_length_binder_is_still_written_as_a_range():
    """BoltzGen's grammar is `min..max`; a bare integer is a different token
    and the two are not interchangeable."""
    spec = build_design_spec(target_structure="/w/t.cif", target_chains=["A"],
                             binder_length_min=70, binder_length_max=70,
                             binder_chain_id="B")
    assert spec["entities"][0]["protein"]["sequence"] == "70..70"


def test_several_target_chains_each_get_an_include():
    spec = build_design_spec(target_structure="/w/t.cif", target_chains=["A", "B"],
                             binder_length_min=50, binder_length_max=60,
                             binder_chain_id="C")
    assert spec["entities"][1]["file"]["include"] == [
        {"chain": {"id": "A"}}, {"chain": {"id": "B"}}]


def test_the_binder_chain_may_not_collide_with_a_target_chain():
    """Two entities claiming one chain id is a spec BoltzGen accepts and then
    behaves unpredictably on -- the designed chain would overwrite a target."""
    with pytest.raises(ValueError, match="already a target chain"):
        build_design_spec(target_structure="/w/t.cif", target_chains=["A"],
                          binder_length_min=50, binder_length_max=60,
                          binder_chain_id="A")


def test_an_inverted_length_range_is_refused():
    with pytest.raises(ValueError, match="max"):
        build_design_spec(target_structure="/w/t.cif", target_chains=["A"],
                          binder_length_min=90, binder_length_max=40,
                          binder_chain_id="C")


def test_no_target_chain_is_refused():
    """An empty include means "condition on nothing", which is unconditional
    generation wearing a binder tool's name."""
    with pytest.raises(ValueError, match="at least one target chain"):
        build_design_spec(target_structure="/w/t.cif", target_chains=[],
                          binder_length_min=50, binder_length_max=60,
                          binder_chain_id="C")


def test_the_spec_round_trips_through_yaml():
    """It is written to disk for the engine to read, so it has to survive
    serialisation -- `80..140` in particular must stay a string and not be
    read back as a float."""
    spec = build_design_spec(target_structure="/w/t.cif", target_chains=["A"],
                             binder_length_min=80, binder_length_max=140,
                             binder_chain_id="C")
    reloaded = yaml.safe_load(yaml.safe_dump(spec))
    assert reloaded["entities"][0]["protein"]["sequence"] == "80..140"
    assert isinstance(reloaded["entities"][0]["protein"]["sequence"], str)


def test_redesign_spec_keeps_structure_and_explicit_design_mask(tmp_path):
    from boltzgen_inverse_fold import build_redesign_spec
    source = tmp_path / "complex.cif"
    spec = build_redesign_spec(str(source), ["A"])
    assert spec == {"entities": [{"file": {
        "path": str(source), "design": [{"chain": {"id": "A"}}]}}]}
    # No include filter: all non-designed target chains stay in the input.
    assert "include" not in spec["entities"][0]["file"]


def test_inverse_fold_wrapper_writes_spec_and_calls_engine(tmp_path, monkeypatch):
    import boltzgen_inverse_fold as wrapper
    source = tmp_path / 'backbone.cif'
    source.write_text('existing structure')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['wrapper', '--structure', str(source),
                                    '--design-chains', 'A', '--passthrough',
                                    '--steps', 'inverse_folding', '--only_inverse_fold'])
    calls = []
    def run(argv):
        calls.append(argv)
        assert yaml.safe_load(Path(argv[2]).read_text())['entities'][0]['file']['design'] == [{'chain': {'id': 'A'}}]
        return type('Result', (), {'returncode': 0})()
    monkeypatch.setattr(wrapper.subprocess, 'run', run)
    with pytest.raises(SystemExit) as exc:
        wrapper.main()
    assert exc.value.code == 0
    assert calls[0][-3:] == ['--steps', 'inverse_folding', '--only_inverse_fold']


def test_advanced_spec_export_preserves_relative_structure_reference(tmp_path, monkeypatch):
    import boltzgen_inverse_fold as wrapper
    source_dir = tmp_path / 'source'
    source_dir.mkdir()
    spec = source_dir / 'advanced.yaml'
    spec.write_text('entities:\n- file:\n    path: backbone.cif\n    design: [{chain: {id: A}}]\n')
    workdir = tmp_path / 'work'
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    monkeypatch.setattr(sys, 'argv', ['wrapper', '--design-spec', str(spec)])
    monkeypatch.setattr(wrapper.subprocess, 'run', lambda _: type('Result', (), {'returncode': 0})())
    with pytest.raises(SystemExit):
        wrapper.main()
    result = yaml.safe_load((workdir / 'design_spec.yaml').read_text())
    assert result['entities'][0]['file']['path'] == str(source_dir / 'backbone.cif')
