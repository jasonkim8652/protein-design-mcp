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


# --- array-of-path staging: many files, one shared directory (BoltzGen's
# fold/design_fold/analyze read a design's .cif AND .npz out of the SAME
# design_dir -- a caller passes both as one param's file list, and every
# item lands together under workdir/<name>/, matched by basename). ---------


def test_stages_a_list_of_paths_into_one_shared_subdirectory(tmp_path):
    cif = tmp_path / "gen" / "design_0.cif"
    npz = tmp_path / "meta" / "design_0.npz"
    cif.parent.mkdir()
    npz.parent.mkdir()
    cif.write_text("cif-data")
    npz.write_text("npz-data")
    workdir = tmp_path / "work"
    workdir.mkdir()

    staged = stage_inputs(
        ["generated_files"], {"generated_files": [str(cif), str(npz)]}, workdir
    )

    staged_paths = [Path(p) for p in staged["generated_files"]]
    assert all(p.parent == workdir / "generated_files" for p in staged_paths)
    assert {p.name for p in staged_paths} == {"design_0.cif", "design_0.npz"}
    assert (workdir / "generated_files" / "design_0.cif").read_text() == "cif-data"
    assert (workdir / "generated_files" / "design_0.npz").read_text() == "npz-data"


def test_staging_an_empty_list_leaves_an_empty_list(tmp_path):
    """Corner case: a caller-supplied empty list (e.g. no designs survived
    an upstream filter) must not crash and must not be confused with the
    None/absent case."""
    workdir = tmp_path / "work"
    workdir.mkdir()

    staged = stage_inputs(["generated_files"], {"generated_files": []}, workdir)

    assert staged["generated_files"] == []


def test_a_single_item_list_is_staged_like_any_other(tmp_path):
    cif = tmp_path / "design_0.cif"
    cif.write_text("x")
    workdir = tmp_path / "work"
    workdir.mkdir()

    staged = stage_inputs(["generated_files"], {"generated_files": [str(cif)]}, workdir)

    assert staged["generated_files"] == [str(workdir / "generated_files" / "design_0.cif")]


def test_two_list_items_sharing_a_basename_the_second_overwrites_the_first(tmp_path):
    """Unlike two DIFFERENT param names (which get separate subdirectories),
    two entries of the SAME list are expected to share a directory by
    design -- a same-named collision within one list is a genuine caller
    error (duplicate file), not something this function can safely rename
    around, so it behaves exactly like a plain shutil.copy2 would: last
    write wins. Documented here so this remains a deliberate choice, not an
    accidental one, if it is ever revisited."""
    source_a = tmp_path / "a" / "design_0.cif"
    source_b = tmp_path / "b" / "design_0.cif"
    source_a.parent.mkdir()
    source_b.parent.mkdir()
    source_a.write_text("first")
    source_b.write_text("second")
    workdir = tmp_path / "work"
    workdir.mkdir()

    staged = stage_inputs(
        ["generated_files"],
        {"generated_files": [str(source_a), str(source_b)]},
        workdir,
    )

    assert (workdir / "generated_files" / "design_0.cif").read_text() == "second"
    assert len(staged["generated_files"]) == 2


def test_a_nonexistent_item_in_a_list_raises_file_not_found(tmp_path):
    workdir = tmp_path / "work"
    workdir.mkdir()

    with pytest.raises(FileNotFoundError):
        stage_inputs(
            ["generated_files"],
            {"generated_files": [str(tmp_path / "nope.cif")]},
            workdir,
        )
