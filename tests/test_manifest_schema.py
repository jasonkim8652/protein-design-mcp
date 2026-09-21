import pytest

from protein_design_mcp.manifest.schema import (
    Manifest,
    ManifestError,
    parse_manifest,
)

MINIMAL = {
    "name": "run_prodigy",
    "category": "scoring",
    "engine": {"repo": "prodigy", "env": "scoring", "entry": ["prodigy"]},
    "summary": "Estimate binding free energy for a protein-protein complex.",
    "doc": "## What this is\nPRODIGY.\n",
    "schema": {
        "complex_pdb": {
            "type": "string",
            "pattern": r"\.(pdb|cif)$",
            "required": True,
            "description": "Path to the complex structure.",
            "example": "complex.pdb",
        }
    },
}


def test_parses_minimal_manifest():
    m = parse_manifest(MINIMAL)
    assert m.name == "run_prodigy"
    assert m.category == "scoring"
    assert m.engine.env == "scoring"
    assert m.engine.entry == ("prodigy",)
    assert m.composite is False
    assert m.requires.gpu is False
    assert m.requires.license_gated is False


def test_composite_flag_is_read():
    data = {**MINIMAL, "composite": True}
    assert parse_manifest(data).composite is True


def test_requires_block_is_read():
    data = {
        **MINIMAL,
        "requires": {"gpu": True, "weights": "ckpts/x.ckpt", "license_gated": True},
    }
    m = parse_manifest(data)
    assert m.requires.gpu is True
    assert m.requires.weights == "ckpts/x.ckpt"
    assert m.requires.license_gated is True


def test_rejects_bad_tool_name():
    data = {**MINIMAL, "name": "designBinder"}
    with pytest.raises(ManifestError, match="name"):
        parse_manifest(data)


def test_rejects_missing_summary():
    data = {k: v for k, v in MINIMAL.items() if k != "summary"}
    with pytest.raises(ManifestError, match="summary"):
        parse_manifest(data)


def test_rejects_unknown_category():
    data = {**MINIMAL, "category": "miscellaneous"}
    with pytest.raises(ManifestError, match="category"):
        parse_manifest(data)


def test_manifest_is_frozen():
    m = parse_manifest(MINIMAL)
    with pytest.raises(Exception):
        m.name = "other"


def test_rejects_whitespace_only_summary():
    data = {**MINIMAL, "summary": "   "}
    with pytest.raises(ManifestError, match="summary"):
        parse_manifest(data)


def test_accepts_empty_schema():
    data = {**MINIMAL, "schema": {}}
    m = parse_manifest(data)
    assert m.schema == {}


def test_rejects_missing_schema():
    data = {k: v for k, v in MINIMAL.items() if k != "schema"}
    with pytest.raises(ManifestError, match="schema"):
        parse_manifest(data)
