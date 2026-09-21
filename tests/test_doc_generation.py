from pathlib import Path

from protein_design_mcp.manifest.loader import load_manifests

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from generate_tool_docs import main, render_doc  # noqa: E402

MANIFEST_DIR = Path(__file__).resolve().parents[1] / "manifests"


def _prodigy():
    return next(m for m in load_manifests(MANIFEST_DIR) if m.name == "run_prodigy")


def test_rendered_doc_starts_with_the_tool_name():
    assert render_doc(_prodigy()).startswith("# run_prodigy")


def test_rendered_doc_contains_the_body():
    assert "## What this is" in render_doc(_prodigy())


def test_rendered_doc_has_a_parameter_table_with_constraints():
    rendered = render_doc(_prodigy())
    assert "| Parameter |" in rendered
    assert "complex_pdb" in rendered
    assert "25.0" in rendered  # the temperature default


def test_rendered_doc_names_the_engine_and_environment():
    rendered = render_doc(_prodigy())
    assert "prodigy" in rendered
    assert "scoring" in rendered


def test_main_writes_one_file_per_manifest(tmp_path):
    written = main(MANIFEST_DIR, tmp_path)
    assert (tmp_path / "run_prodigy.md").exists()
    assert len(written) == len(load_manifests(MANIFEST_DIR))


def test_generated_docs_are_current(tmp_path):
    """Fails when a manifest changed but docs/tools/ was not regenerated."""
    main(MANIFEST_DIR, tmp_path)
    committed = Path(__file__).resolve().parents[1] / "docs" / "tools"
    for generated in sorted(tmp_path.glob("*.md")):
        assert (committed / generated.name).read_text() == generated.read_text()
