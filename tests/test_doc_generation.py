import sys
from pathlib import Path

from protein_design_mcp.app import manifest_dir
from protein_design_mcp.manifest.loader import load_manifests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from generate_tool_docs import main, render_doc  # noqa: E402

MANIFEST_DIR = manifest_dir()


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
    # Detect orphaned docs (manifest deleted but doc file left behind)
    generated_names = {p.name for p in tmp_path.glob("*.md")}
    committed_names = {p.name for p in committed.glob("*.md")}
    assert generated_names == committed_names, (
        f"Orphaned doc file(s) in {committed}: {committed_names - generated_names}. "
        f"Delete the doc file when its manifest is removed."
    )


def test_parameter_table_has_valid_markdown_cells():
    """Verify each parameter table row has the correct number of unescaped pipe cells."""
    rendered = render_doc(_prodigy())
    lines = rendered.split("\n")

    # Find the header row
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith("| Parameter |"):
            header_idx = i
            break

    assert header_idx is not None, "Parameter table header not found"

    # Count unescaped pipes: pipes not preceded by backslash
    def count_unescaped_pipes(line: str) -> int:
        count = 0
        for i, char in enumerate(line):
            if char == "|" and (i == 0 or line[i - 1] != "\\"):
                count += 1
        return count

    header_line = lines[header_idx]
    expected_pipes = count_unescaped_pipes(header_line)

    # For each row after the separator, verify unescaped pipe count
    separator_idx = header_idx + 1
    for i in range(separator_idx + 1, len(lines)):
        line = lines[i]
        if not line.strip() or not line.startswith("|"):
            break
        actual_pipes = count_unescaped_pipes(line)
        assert actual_pipes == expected_pipes, (
            f"Row {i} has {actual_pipes} unescaped pipes but header has {expected_pipes}. "
            f"This breaks the markdown table. Line: {line}"
        )
