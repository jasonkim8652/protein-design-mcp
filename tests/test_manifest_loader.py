import textwrap

import pytest

from protein_design_mcp.manifest.loader import SIBLING_DOC_HEADING, load_manifests
from protein_design_mcp.manifest.schema import ManifestError


def _write(tmp_path, filename, body):
    path = tmp_path / filename
    path.write_text(textwrap.dedent(body))
    return path


SOLO = """\
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


def test_loads_a_single_manifest(tmp_path):
    _write(tmp_path, "run_prodigy.yaml", SOLO)
    manifests = load_manifests(tmp_path)
    assert [m.name for m in manifests] == ["run_prodigy"]


def test_empty_directory_returns_empty_list(tmp_path):
    assert load_manifests(tmp_path) == []


def test_nonexistent_directory_raises_manifest_error(tmp_path):
    """Regression for FIX 1: a missing manifest directory must raise
    ManifestError (which build_registry can catch and degrade from),
    never an unhandled FileNotFoundError or similar."""
    missing = tmp_path / "does-not-exist"
    assert not missing.exists()
    with pytest.raises(ManifestError, match="not found"):
        load_manifests(missing)


def test_duplicate_names_are_rejected(tmp_path):
    _write(tmp_path, "a.yaml", SOLO)
    _write(tmp_path, "b.yaml", SOLO)
    with pytest.raises(ManifestError, match="duplicate"):
        load_manifests(tmp_path)


def test_siblings_must_document_how_to_choose(tmp_path):
    _write(tmp_path, "a.yaml", SOLO)
    _write(
        tmp_path,
        "b.yaml",
        SOLO.replace("run_prodigy", "run_ipsae"),
    )
    with pytest.raises(ManifestError, match=SIBLING_DOC_HEADING):
        load_manifests(tmp_path)


def test_siblings_pass_when_they_document_the_choice(tmp_path):
    with_heading = SOLO.replace(
        "      PRODIGY.\n",
        f"      PRODIGY.\n\n      {SIBLING_DOC_HEADING}\n      Use run_ipsae instead when you already have a PAE matrix.\n",
    )
    _write(tmp_path, "a.yaml", with_heading)
    _write(tmp_path, "b.yaml", with_heading.replace("run_prodigy", "run_ipsae"))
    assert len(load_manifests(tmp_path)) == 2


def test_solo_category_needs_no_comparison_heading(tmp_path):
    _write(tmp_path, "run_prodigy.yaml", SOLO)
    assert len(load_manifests(tmp_path)) == 1


def test_malformed_yaml_names_the_file(tmp_path):
    _write(tmp_path, "broken.yaml", "name: designBinder\n")
    with pytest.raises(ManifestError, match="broken.yaml"):
        load_manifests(tmp_path)
