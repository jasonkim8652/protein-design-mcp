import os
from pathlib import Path

import pytest

from protein_design_mcp.manifest.schema import OutputSpec
from protein_design_mcp.results import collect_outputs, results_dir


def test_results_dir_honours_the_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv("PROTEIN_MCP_RESULTS_DIR", str(tmp_path / "r"))
    assert results_dir() == tmp_path / "r"


def test_results_dir_is_created(tmp_path, monkeypatch):
    monkeypatch.setenv("PROTEIN_MCP_RESULTS_DIR", str(tmp_path / "made"))
    results_dir().mkdir(parents=True, exist_ok=True)
    assert (tmp_path / "made").is_dir()


def test_declared_output_is_copied_out(tmp_path, monkeypatch):
    monkeypatch.setenv("PROTEIN_MCP_RESULTS_DIR", str(tmp_path / "res"))
    workdir = tmp_path / "wd"
    workdir.mkdir()
    (workdir / "minimized.pdb").write_text("ATOM\n")

    specs = (OutputSpec(name="minimized_pdb", pattern="minimized.pdb"),)
    collected = collect_outputs(specs, workdir, "run123")

    assert set(collected) == {"minimized_pdb"}
    copied = Path(collected["minimized_pdb"])
    assert copied.read_text() == "ATOM\n"
    assert not copied.is_relative_to(workdir)


def test_glob_pattern_collects_the_single_match(tmp_path, monkeypatch):
    monkeypatch.setenv("PROTEIN_MCP_RESULTS_DIR", str(tmp_path / "res"))
    workdir = tmp_path / "wd"
    workdir.mkdir()
    (workdir / "out_0.pdb").write_text("X\n")

    specs = (OutputSpec(name="design", pattern="out_*.pdb"),)
    assert Path(collect_outputs(specs, workdir, "r")["design"]).name == "out_0.pdb"


def test_missing_declared_output_raises_naming_it(tmp_path, monkeypatch):
    monkeypatch.setenv("PROTEIN_MCP_RESULTS_DIR", str(tmp_path / "res"))
    workdir = tmp_path / "wd"
    workdir.mkdir()
    specs = (OutputSpec(name="minimized_pdb", pattern="minimized.pdb"),)
    with pytest.raises(FileNotFoundError, match="minimized_pdb"):
        collect_outputs(specs, workdir, "r")


def test_no_specs_collects_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("PROTEIN_MCP_RESULTS_DIR", str(tmp_path / "res"))
    workdir = tmp_path / "wd"
    workdir.mkdir()
    assert collect_outputs((), workdir, "r") == {}


def test_same_basename_in_different_subdirs_does_not_collide(tmp_path, monkeypatch):
    monkeypatch.setenv("PROTEIN_MCP_RESULTS_DIR", str(tmp_path / "res"))
    workdir = tmp_path / "wd"
    (workdir / "designs").mkdir(parents=True)
    (workdir / "scores").mkdir(parents=True)
    (workdir / "designs" / "out.pdb").write_text("DESIGN\n")
    (workdir / "scores" / "out.pdb").write_text("SCORE\n")

    specs = (
        OutputSpec(name="design_pdb", pattern="designs/out.pdb"),
        OutputSpec(name="score_pdb", pattern="scores/out.pdb"),
    )
    collected = collect_outputs(specs, workdir, "r")

    design_path = Path(collected["design_pdb"])
    score_path = Path(collected["score_pdb"])
    assert design_path.exists()
    assert score_path.exists()
    assert design_path != score_path
    assert design_path.read_text() == "DESIGN\n"
    assert score_path.read_text() == "SCORE\n"


def test_ambiguous_single_valued_pattern_raises_naming_all_matches(tmp_path, monkeypatch):
    monkeypatch.setenv("PROTEIN_MCP_RESULTS_DIR", str(tmp_path / "res"))
    workdir = tmp_path / "wd"
    workdir.mkdir()
    (workdir / "out_0.pdb").write_text("A\n")
    (workdir / "out_1.pdb").write_text("B\n")

    specs = (OutputSpec(name="design", pattern="out_*.pdb"),)
    with pytest.raises(FileNotFoundError, match="design") as exc:
        collect_outputs(specs, workdir, "r")
    assert "out_0.pdb" in str(exc.value)
    assert "out_1.pdb" in str(exc.value)


def test_multiple_true_returns_a_list_of_all_matches(tmp_path, monkeypatch):
    monkeypatch.setenv("PROTEIN_MCP_RESULTS_DIR", str(tmp_path / "res"))
    workdir = tmp_path / "wd"
    workdir.mkdir()
    (workdir / "out_0.pdb").write_text("A\n")
    (workdir / "out_1.pdb").write_text("B\n")

    specs = (OutputSpec(name="designs", pattern="out_*.pdb", multiple=True),)
    collected = collect_outputs(specs, workdir, "r")

    assert isinstance(collected["designs"], list)
    names = sorted(Path(p).name for p in collected["designs"])
    assert names == ["out_0.pdb", "out_1.pdb"]
    for p in collected["designs"]:
        assert Path(p).exists()


def test_multiple_true_with_zero_matches_still_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("PROTEIN_MCP_RESULTS_DIR", str(tmp_path / "res"))
    workdir = tmp_path / "wd"
    workdir.mkdir()

    specs = (OutputSpec(name="designs", pattern="out_*.pdb", multiple=True),)
    with pytest.raises(FileNotFoundError, match="designs"):
        collect_outputs(specs, workdir, "r")


def test_single_valued_pattern_with_exactly_one_match_returns_a_string(tmp_path, monkeypatch):
    monkeypatch.setenv("PROTEIN_MCP_RESULTS_DIR", str(tmp_path / "res"))
    workdir = tmp_path / "wd"
    workdir.mkdir()
    (workdir / "out_0.pdb").write_text("A\n")

    specs = (OutputSpec(name="design", pattern="out_*.pdb"),)
    collected = collect_outputs(specs, workdir, "r")

    assert isinstance(collected["design"], str)
    assert Path(collected["design"]).name == "out_0.pdb"
