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


def test_rejects_a_schema_entry_that_is_not_a_mapping():
    """Regression for FIX 6: schema: {p: "string"} used to parse cleanly
    and only blow up later, deep inside ToolRegistry.tools(), with an
    opaque AttributeError that takes down tools/list for every tool."""
    data = {**MINIMAL, "schema": {"p": "string"}}
    with pytest.raises(ManifestError, match="p"):
        parse_manifest(data)


def test_rejects_minimum_without_a_type():
    """A typeless numeric spec lets a bool pass as 1/0 in validation.py's
    range check, since bool is an int subclass."""
    data = {**MINIMAL, "schema": {"n": {"minimum": 0}}}
    with pytest.raises(ManifestError, match="n"):
        parse_manifest(data)


def test_rejects_maximum_without_a_type():
    data = {**MINIMAL, "schema": {"n": {"maximum": 10}}}
    with pytest.raises(ManifestError, match="n"):
        parse_manifest(data)


def test_accepts_minimum_with_a_type():
    data = {**MINIMAL, "schema": {"n": {"type": "integer", "minimum": 0}}}
    m = parse_manifest(data)
    assert m.schema["n"]["minimum"] == 0


def test_outputs_default_to_empty():
    assert parse_manifest(MINIMAL).outputs == ()


def test_outputs_are_parsed():
    data = {
        **MINIMAL,
        "outputs": [
            {"name": "minimized_pdb", "pattern": "minimized.pdb",
             "description": "The relaxed structure."},
        ],
    }
    (out,) = parse_manifest(data).outputs
    assert out.name == "minimized_pdb"
    assert out.pattern == "minimized.pdb"
    assert out.description == "The relaxed structure."


def test_output_without_name_is_rejected():
    data = {**MINIMAL, "outputs": [{"pattern": "x.pdb"}]}
    with pytest.raises(ManifestError, match="name"):
        parse_manifest(data)


def test_output_without_pattern_is_rejected():
    data = {**MINIMAL, "outputs": [{"name": "x"}]}
    with pytest.raises(ManifestError, match="pattern"):
        parse_manifest(data)


def test_absolute_output_pattern_is_rejected():
    data = {**MINIMAL, "outputs": [{"name": "x", "pattern": "/etc/passwd"}]}
    with pytest.raises(ManifestError, match="relative"):
        parse_manifest(data)


def test_output_pattern_escaping_the_workdir_is_rejected():
    data = {**MINIMAL, "outputs": [{"name": "x", "pattern": "../escape.pdb"}]}
    with pytest.raises(ManifestError, match="relative"):
        parse_manifest(data)


def test_duplicate_output_names_are_rejected():
    data = {
        **MINIMAL,
        "outputs": [{"name": "x", "pattern": "a.pdb"},
                    {"name": "x", "pattern": "b.pdb"}],
    }
    with pytest.raises(ManifestError, match="duplicate"):
        parse_manifest(data)


def test_timeout_defaults_to_one_hour():
    assert parse_manifest(MINIMAL).timeout_s == 3600


def test_output_multiple_defaults_to_false():
    data = {
        **MINIMAL,
        "outputs": [{"name": "minimized_pdb", "pattern": "minimized.pdb"}],
    }
    (out,) = parse_manifest(data).outputs
    assert out.multiple is False


def test_output_multiple_is_parsed():
    data = {
        **MINIMAL,
        "outputs": [
            {"name": "designs_fasta", "pattern": "seqs/*.fa", "multiple": True},
        ],
    }
    (out,) = parse_manifest(data).outputs
    assert out.multiple is True


def test_timeout_is_parsed():
    assert parse_manifest({**MINIMAL, "timeout_s": 120}).timeout_s == 120


def test_nonpositive_timeout_is_rejected():
    with pytest.raises(ManifestError, match="timeout_s"):
        parse_manifest({**MINIMAL, "timeout_s": 0})


def test_boolean_true_timeout_is_rejected():
    with pytest.raises(ManifestError, match="timeout_s"):
        parse_manifest({**MINIMAL, "timeout_s": True})


def test_boolean_false_timeout_is_rejected():
    with pytest.raises(ManifestError, match="timeout_s"):
        parse_manifest({**MINIMAL, "timeout_s": False})


def test_float_timeout_is_rejected():
    with pytest.raises(ManifestError, match="timeout_s"):
        parse_manifest({**MINIMAL, "timeout_s": 1.9})


def test_integer_timeout_still_accepted():
    m = parse_manifest({**MINIMAL, "timeout_s": 120})
    assert m.timeout_s == 120
