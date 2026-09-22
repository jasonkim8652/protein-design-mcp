"""Task 1: a malformed manifest must not remove every tool (gate item G1).

Regression coverage for the defect at src/protein_design_mcp/manifest/loader.py:147 —

    manifests = [_load_one(p) for p in sorted(directory.glob("*.yaml"))]

One raise anywhere in that comprehension used to abort loading every
manifest in the directory: a single bad file silently emptied the whole
tool registry, and the model just saw a one-tool (or zero-tool) server with
no indication why the other, perfectly valid, manifests vanished.

``load_manifests_resilient`` fixes this: a manifest that fails to load is
excluded individually, with a reason, and every other manifest stays
available. ``load_manifests`` itself is untouched (existing callers keep
its current all-or-nothing behaviour); this is deliberately a *second*
function so callers who want strict all-or-nothing semantics still get
them without opting into anything.

``STRICT_MANIFESTS=1`` restores strict, all-or-nothing behaviour even
through the resilient entry point — required for CI. The suite runs strict
by default (see tests/conftest.py); tests here that exercise lenient
behaviour turn it off explicitly.
"""

from __future__ import annotations

import textwrap

import pytest

from protein_design_mcp.manifest.loader import load_manifests_resilient
from protein_design_mcp.manifest.registry import ToolNotAvailable, ToolRegistry
from protein_design_mcp.manifest.schema import ManifestError


def _write(tmp_path, filename, body):
    path = tmp_path / filename
    path.write_text(textwrap.dedent(body))
    return path


VALID = """\
    name: run_prodigy
    category: scoring
    engine: {repo: prodigy, env: scoring, entry: [prodigy]}
    summary: Estimate binding free energy.
    doc: |
      ## What this is
      PRODIGY.
    schema:
      complex_pdb: {type: string, required: true, example: c.pdb}
"""

# Different category from VALID's "scoring" so pairing it with VALID never
# accidentally trips the sibling-doc-heading requirement (>=2 manifests in
# one category) while a test is only trying to exercise cross-references.
BAD_REF = """\
    name: run_needs_ref
    category: generation
    engine: {repo: x, env: e, entry: [x]}
    summary: Does a thing. See run_ghost_tool for the alternative.
    doc: |
      ## What this is
      Does a thing.
    schema: {}
"""


def _lenient(monkeypatch):
    """Undo the strict-by-default set in tests/conftest.py for this test."""
    monkeypatch.delenv("STRICT_MANIFESTS", raising=False)


# --- Cases 1 & 2: one valid, one malformed manifest -------------------------


def test_malformed_manifest_does_not_take_down_a_valid_sibling(tmp_path, monkeypatch):
    _lenient(monkeypatch)
    _write(tmp_path, "run_prodigy.yaml", VALID)
    _write(tmp_path, "broken.yaml", "name: run_broken\ncategory: not_a_real_category\n")

    result = load_manifests_resilient(tmp_path)

    assert [m.name for m in result.manifests] == ["run_prodigy"]
    assert "broken.yaml" in result.reasons
    # the reason names the file
    assert "broken.yaml" in result.reasons["broken.yaml"]


def test_surviving_tool_is_resolvable_not_merely_counted(tmp_path, monkeypatch):
    """The assertion that would have caught G1: listing a tool is not the
    same as it being callable."""
    _lenient(monkeypatch)
    _write(tmp_path, "run_prodigy.yaml", VALID)
    _write(tmp_path, "broken.yaml", "name: run_broken\ncategory: not_a_real_category\n")

    result = load_manifests_resilient(tmp_path)
    registry = ToolRegistry(result.manifests, load_failures=result.reasons)

    resolved = registry.resolve("run_prodigy")
    assert resolved.name == "run_prodigy"


# --- Case 3: duplicate tool name --------------------------------------------


def test_duplicate_tool_name_excludes_both_manifests(tmp_path, monkeypatch):
    _lenient(monkeypatch)
    _write(tmp_path, "a.yaml", VALID)
    _write(tmp_path, "b.yaml", VALID)  # both declare run_prodigy

    result = load_manifests_resilient(tmp_path)

    assert result.manifests == []
    assert "run_prodigy" in result.reasons
    # reason names the conflict (the duplicated tool name), not a made-up winner
    assert "run_prodigy" in result.reasons["run_prodigy"]

    registry = ToolRegistry(result.manifests, load_failures=result.reasons)
    assert registry.tools() == []
    with pytest.raises(ToolNotAvailable, match="run_prodigy"):
        registry.resolve("run_prodigy")


# --- Case 4: cross-reference / doc violation --------------------------------


def test_bad_cross_reference_excludes_only_the_referencing_manifest(tmp_path, monkeypatch):
    _lenient(monkeypatch)
    _write(tmp_path, "run_prodigy.yaml", VALID)
    _write(tmp_path, "run_needs_ref.yaml", BAD_REF)

    result = load_manifests_resilient(tmp_path)

    assert [m.name for m in result.manifests] == ["run_prodigy"]
    assert "run_needs_ref" in result.reasons
    assert "run_ghost_tool" in result.reasons["run_needs_ref"]

    registry = ToolRegistry(result.manifests, load_failures=result.reasons)
    assert registry.resolve("run_prodigy").name == "run_prodigy"
    with pytest.raises(ToolNotAvailable, match="run_ghost_tool"):
        registry.resolve("run_needs_ref")


# --- Case 5: STRICT_MANIFESTS=1 restores all-or-nothing --------------------


def test_strict_mode_raises_for_a_malformed_file(tmp_path, monkeypatch):
    monkeypatch.setenv("STRICT_MANIFESTS", "1")
    _write(tmp_path, "run_prodigy.yaml", VALID)
    _write(tmp_path, "broken.yaml", "name: run_broken\ncategory: not_a_real_category\n")

    with pytest.raises(ManifestError, match="broken.yaml"):
        load_manifests_resilient(tmp_path)


def test_strict_mode_raises_for_duplicate_names(tmp_path, monkeypatch):
    monkeypatch.setenv("STRICT_MANIFESTS", "1")
    _write(tmp_path, "a.yaml", VALID)
    _write(tmp_path, "b.yaml", VALID)

    with pytest.raises(ManifestError, match="duplicate"):
        load_manifests_resilient(tmp_path)


def test_strict_mode_raises_for_a_bad_cross_reference(tmp_path, monkeypatch):
    monkeypatch.setenv("STRICT_MANIFESTS", "1")
    _write(tmp_path, "run_prodigy.yaml", VALID)
    _write(tmp_path, "run_needs_ref.yaml", BAD_REF)

    with pytest.raises(ManifestError, match="run_ghost_tool"):
        load_manifests_resilient(tmp_path)


# --- Case 6: every manifest malformed ---------------------------------------


def test_every_manifest_malformed_yields_empty_registry_not_an_exception(
    tmp_path, monkeypatch
):
    _lenient(monkeypatch)
    _write(tmp_path, "a.yaml", "category: not_a_real_category\n")  # missing name
    _write(tmp_path, "b.yaml", "key: [1, 2\n")  # invalid YAML

    result = load_manifests_resilient(tmp_path)

    assert result.manifests == []
    assert set(result.reasons) == {"a.yaml", "b.yaml"}

    registry = ToolRegistry(result.manifests, load_failures=result.reasons)
    assert registry.tools() == []


# --- Case 7: corner cases ----------------------------------------------------


def test_empty_directory_returns_empty_result_not_an_error(tmp_path, monkeypatch):
    _lenient(monkeypatch)
    result = load_manifests_resilient(tmp_path)
    assert result.manifests == []
    assert result.reasons == {}


def test_zero_byte_yaml_file_is_excluded_not_crashing(tmp_path, monkeypatch):
    _lenient(monkeypatch)
    (tmp_path / "empty.yaml").write_text("")
    _write(tmp_path, "run_prodigy.yaml", VALID)

    result = load_manifests_resilient(tmp_path)

    assert [m.name for m in result.manifests] == ["run_prodigy"]
    assert "empty.yaml" in result.reasons


def test_comment_only_yaml_file_is_excluded_not_treated_as_empty_dict(
    tmp_path, monkeypatch
):
    """A file parsing to None (comment-only) must not be silently confused
    with a manifest parsing to {} — both are invalid, but that confusion
    ('key absent vs value empty') has bitten this codebase before."""
    _lenient(monkeypatch)
    (tmp_path / "comment.yaml").write_text("# just a comment, nothing else\n")
    _write(tmp_path, "run_prodigy.yaml", VALID)

    result = load_manifests_resilient(tmp_path)

    assert [m.name for m in result.manifests] == ["run_prodigy"]
    assert "comment.yaml" in result.reasons
