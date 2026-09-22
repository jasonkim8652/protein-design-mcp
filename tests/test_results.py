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
