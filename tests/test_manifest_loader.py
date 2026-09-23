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


def test_doc_naming_an_unknown_tool_is_rejected(tmp_path):
    body = SOLO.replace(
        "      PRODIGY.\n",
        "      PRODIGY. See run_does_not_exist for the alternative.\n",
    )
    _write(tmp_path, "run_prodigy.yaml", body)
    with pytest.raises(ManifestError, match="run_does_not_exist"):
        load_manifests(tmp_path)


def test_doc_naming_a_known_sibling_is_accepted(tmp_path):
    heading = SIBLING_DOC_HEADING
    a = SOLO.replace(
        "      PRODIGY.\n",
        f"      PRODIGY.\n\n      {heading}\n      Use run_ipsae to rank designs.\n",
    )
    b = a.replace("run_prodigy", "run_ipsae")
    _write(tmp_path, "a.yaml", a)
    _write(tmp_path, "b.yaml", b)
    assert len(load_manifests(tmp_path)) == 2


def test_doc_may_name_a_tool_marked_not_yet_implemented(tmp_path):
    body = SOLO.replace(
        "      PRODIGY.\n",
        "      PRODIGY. run_rosetta_interface (not yet implemented) will "
        "give the physics breakdown.\n",
    )
    _write(tmp_path, "run_prodigy.yaml", body)
    assert len(load_manifests(tmp_path)) == 1


def test_summary_naming_an_unknown_tool_is_rejected(tmp_path):
    """The summary field (most visible) must also validate tool references."""
    body = SOLO.replace(
        'summary: Estimate binding free energy.',
        'summary: See run_does_not_exist for more.',
    )
    _write(tmp_path, "run_prodigy.yaml", body)
    with pytest.raises(ManifestError, match="run_does_not_exist"):
        load_manifests(tmp_path)


def test_summary_naming_a_tool_marked_not_yet_implemented_is_accepted(tmp_path):
    """Forward references in summary are OK if marked."""
    body = SOLO.replace(
        'summary: Estimate binding free energy.',
        'summary: See run_future_tool (not yet implemented) for more.',
    )
    _write(tmp_path, "run_prodigy.yaml", body)
    assert len(load_manifests(tmp_path)) == 1
