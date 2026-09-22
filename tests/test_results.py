import os
from pathlib import Path

import pytest

from protein_design_mcp.manifest.schema import OutputSpec
from protein_design_mcp.results import (
    AmbiguousOutputError,
    OutputPathEscapeError,
    collect_outputs,
    results_dir,
)


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
    with pytest.raises(AmbiguousOutputError, match="design") as exc:
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


def test_multiple_true_same_basename_in_different_subdirs_does_not_collide(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("PROTEIN_MCP_RESULTS_DIR", str(tmp_path / "res"))
    workdir = tmp_path / "wd"
    (workdir / "x").mkdir(parents=True)
    (workdir / "y").mkdir(parents=True)
    (workdir / "x" / "out.fa").write_text("X\n")
    (workdir / "y" / "out.fa").write_text("Y\n")

    specs = (OutputSpec(name="designs_fasta", pattern="*/out.fa", multiple=True),)
    collected = collect_outputs(specs, workdir, "r")

    paths = collected["designs_fasta"]
    assert isinstance(paths, list)
    assert len(paths) == 2
    assert len(set(paths)) == 2, "the two matches must not collide on one path"
    contents = sorted(Path(p).read_text() for p in paths)
    assert contents == ["X\n", "Y\n"]


def test_symlinked_scratch_root_never_raises_a_bare_value_error(tmp_path, monkeypatch):
    """Deterministic reproduction of the scratch-root-is-a-symlink shape:
    a real directory tree with a symlink pointing at it, collecting through
    a workdir reached via that symlink. On this platform/Python version,
    Path.glob already builds match paths that share the workdir's own
    (unresolved) prefix, so this case happens to succeed even before the
    fix — but the assertion here is about what must NEVER happen, not about
    forcing a particular outcome: collect_outputs must not let a bare
    ValueError escape. Either a correct collection or a diagnosable
    OSError is acceptable; a bare ValueError is not, and would fail this
    test by propagating out uncaught.
    """
    monkeypatch.setenv("PROTEIN_MCP_RESULTS_DIR", str(tmp_path / "res"))
    real_root = tmp_path / "real"
    real_root.mkdir()
    link_root = tmp_path / "link"
    link_root.symlink_to(real_root, target_is_directory=True)

    workdir = link_root / "wd"
    workdir.mkdir()
    (workdir / "out.fa").write_text("hello\n")

    specs = (OutputSpec(name="design", pattern="out.fa"),)

    try:
        collected = collect_outputs(specs, workdir, "r")
    except OSError:
        # A diagnosable OSError (e.g. OutputPathEscapeError) is acceptable.
        # A bare ValueError is NOT caught by this clause and would
        # propagate out of the test, failing it -- which is exactly the
        # regression this test guards against.
        pass
    else:
        assert Path(collected["design"]).read_text() == "hello\n"


def test_output_matching_a_symlink_outside_the_workdir_is_refused(tmp_path, monkeypatch):
    """Deterministic escape case: a file *inside* the workdir that is a
    symlink pointing outside it. Resolving both sides (the fix) makes this
    genuinely fall outside the workdir once resolved, so it must be refused
    with a diagnosable OutputPathEscapeError -- not silently copied from
    wherever it points, and not a bare ValueError.
    """
    monkeypatch.setenv("PROTEIN_MCP_RESULTS_DIR", str(tmp_path / "res"))
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "leak.fa").write_text("secret\n")

    workdir = tmp_path / "wd"
    workdir.mkdir()
    (workdir / "leak.fa").symlink_to(outside / "leak.fa")

    specs = (OutputSpec(name="designs", pattern="*.fa", multiple=True),)
    with pytest.raises(OutputPathEscapeError, match="designs"):
        collect_outputs(specs, workdir, "r")
