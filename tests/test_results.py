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


# ---------------------------------------------------------------------------
# Containment matrix.
#
# Containment is a property of *every* matched source, so it has to be tested
# across every axis that changes which code path a match travels. `multiple`
# is that axis: rounds 1 and 3 each fixed one of its two branches and left the
# other exposed, precisely because each round's test only exercised the branch
# it had just edited. These four tests pin both branches against both
# outcomes -- refuse the escape, and do NOT over-reject the ordinary file --
# so a fix to one branch alone can no longer look green.
# ---------------------------------------------------------------------------

BOTH_MULTIPLE_MODES = pytest.mark.parametrize(
    "multiple",
    [False, True],
    ids=["multiple_false_the_default", "multiple_true"],
)


@BOTH_MULTIPLE_MODES
def test_symlink_escape_is_refused_for_both_multiple_modes(tmp_path, monkeypatch, multiple):
    """A symlink inside the workdir pointing outside it must be refused.

    The default (``multiple=False``) branch is the one every simple engine
    uses, so an escape there is the one that matters most.
    """
    monkeypatch.setenv("PROTEIN_MCP_RESULTS_DIR", str(tmp_path / "res"))
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("SECRET\n")

    workdir = tmp_path / "wd"
    workdir.mkdir()
    (workdir / "link.txt").symlink_to(outside / "secret.txt")

    specs = (OutputSpec(name="leak", pattern="link.txt", multiple=multiple),)
    with pytest.raises(OutputPathEscapeError, match="leak"):
        collect_outputs(specs, workdir, f"r{multiple}")

    # Nothing from outside the workdir may have been written into results.
    results_root = tmp_path / "res"
    leaked = [p for p in results_root.rglob("*") if p.is_file()]
    assert leaked == [], f"refused escape still copied files: {leaked}"


@BOTH_MULTIPLE_MODES
def test_ordinary_file_is_still_collected_for_both_multiple_modes(
    tmp_path, monkeypatch, multiple
):
    """Over-rejection guard: a plain, non-symlinked file must still collect.

    A containment check that refuses everything would make the escape tests
    above pass vacuously.
    """
    monkeypatch.setenv("PROTEIN_MCP_RESULTS_DIR", str(tmp_path / "res"))
    workdir = tmp_path / "wd"
    workdir.mkdir()
    (workdir / "plain.txt").write_text("ORDINARY\n")

    specs = (OutputSpec(name="plain", pattern="plain.txt", multiple=multiple),)
    collected = collect_outputs(specs, workdir, f"r{multiple}")

    got = collected["plain"]
    if multiple:
        assert isinstance(got, list) and len(got) == 1
        got = got[0]
    else:
        assert isinstance(got, str)
    assert Path(got).read_text() == "ORDINARY\n"
    assert not Path(got).is_relative_to(workdir)


@BOTH_MULTIPLE_MODES
def test_symlink_pointing_inside_the_workdir_is_allowed(tmp_path, monkeypatch, multiple):
    """Second over-rejection guard: symlink-ness alone is not an escape.

    A symlink that resolves to a file still inside the workdir is contained,
    so it must be collected. Only leaving the workdir is refused.
    """
    monkeypatch.setenv("PROTEIN_MCP_RESULTS_DIR", str(tmp_path / "res"))
    workdir = tmp_path / "wd"
    (workdir / "real").mkdir(parents=True)
    (workdir / "real" / "target.txt").write_text("INSIDE\n")
    (workdir / "alias.txt").symlink_to(workdir / "real" / "target.txt")

    specs = (OutputSpec(name="alias", pattern="alias.txt", multiple=multiple),)
    collected = collect_outputs(specs, workdir, f"r{multiple}")

    got = collected["alias"]
    if multiple:
        got = got[0]
    assert Path(got).read_text() == "INSIDE\n"


def test_no_output_is_copied_when_a_later_spec_escapes(tmp_path, monkeypatch):
    """Containment is checked for every spec before any copy happens.

    This pins the choke point's *position*: a valid spec listed before an
    escaping one must not have been copied out by the time the escape is
    refused. Without a validate-everything-first pass, the first spec's file
    would already be sitting in the results directory.
    """
    monkeypatch.setenv("PROTEIN_MCP_RESULTS_DIR", str(tmp_path / "res"))
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("SECRET\n")

    workdir = tmp_path / "wd"
    workdir.mkdir()
    (workdir / "good.txt").write_text("GOOD\n")
    (workdir / "link.txt").symlink_to(outside / "secret.txt")

    specs = (
        OutputSpec(name="good", pattern="good.txt"),
        OutputSpec(name="leak", pattern="link.txt"),
    )
    with pytest.raises(OutputPathEscapeError, match="leak"):
        collect_outputs(specs, workdir, "r")

    copied = [p for p in (tmp_path / "res").rglob("*") if p.is_file()]
    assert copied == [], f"a copy happened before all specs were checked: {copied}"
