from pathlib import Path

import pytest

from protein_design_mcp.staging import stage_inputs


def test_stages_a_named_path_parameter_into_its_own_subdirectory(tmp_path):
    source = tmp_path / "in" / "model.pdb"
    source.parent.mkdir()
    source.write_text("ATOM ...")
    workdir = tmp_path / "work"
    workdir.mkdir()

    staged = stage_inputs(["structure"], {"structure": str(source)}, workdir)

    staged_path = Path(staged["structure"])
    assert staged_path == workdir / "structure" / "model.pdb"
    assert staged_path.read_text() == "ATOM ..."


def test_returns_a_new_dict_without_mutating_the_original(tmp_path):
    source = tmp_path / "model.pdb"
    source.write_text("x")
    workdir = tmp_path / "work"
    workdir.mkdir()

    original = {"structure": str(source)}
    staged = stage_inputs(["structure"], original, workdir)

    assert original["structure"] == str(source)
    assert staged["structure"] != original["structure"]


def test_two_staged_inputs_sharing_a_basename_do_not_collide(tmp_path):
    """The exact failure collect_outputs was already hardened against, on
    the input side: two DIFFERENT source files that happen to share a
    basename must both survive, each under its own param-name subdirectory,
    rather than the second silently overwriting the first."""
    source_a = tmp_path / "a" / "model.pdb"
    source_a.parent.mkdir()
    source_a.write_text("first")
    source_b = tmp_path / "b" / "model.pdb"
    source_b.parent.mkdir()
    source_b.write_text("second")
    workdir = tmp_path / "work"
    workdir.mkdir()

    staged = stage_inputs(
        ["structure", "reference"],
        {"structure": str(source_a), "reference": str(source_b)},
        workdir,
    )

    assert Path(staged["structure"]).read_text() == "first"
    assert Path(staged["reference"]).read_text() == "second"
    assert staged["structure"] != staged["reference"]


def test_only_named_parameters_are_staged(tmp_path):
    source = tmp_path / "model.pdb"
    source.write_text("x")
    workdir = tmp_path / "work"
    workdir.mkdir()

    staged = stage_inputs(
        ["structure"], {"structure": str(source), "other": "unchanged"}, workdir
    )

    assert staged["other"] == "unchanged"


def test_a_missing_optional_parameter_is_skipped_not_staged(tmp_path):
    workdir = tmp_path / "work"
    workdir.mkdir()

    staged = stage_inputs(["optional_path"], {"optional_path": None}, workdir)

    assert staged["optional_path"] is None
    assert not (workdir / "optional_path").exists()


def test_a_nonexistent_source_file_raises_file_not_found(tmp_path):
    workdir = tmp_path / "work"
    workdir.mkdir()

    with pytest.raises(FileNotFoundError):
        stage_inputs(["structure"], {"structure": str(tmp_path / "nope.pdb")}, workdir)
