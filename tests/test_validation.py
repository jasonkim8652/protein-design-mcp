import pytest

from protein_design_mcp.manifest.schema import parse_manifest
from protein_design_mcp.validation import ToolInputError, validate_and_fill

MANIFEST = parse_manifest(
    {
        "name": "run_demo",
        "category": "scoring",
        "engine": {"repo": "r", "env": "e", "entry": ["x"]},
        "summary": "Demo.",
        "doc": "## What this is\nDemo.\n",
        "schema": {
            "target_pdb": {
                "type": "string",
                "pattern": r"\.(pdb|cif)$",
                "required": True,
                "example": "target.pdb",
                "description": "Path to the target structure.",
            },
            "hotspot_residues": {
                "type": "array",
                "items": {"type": "string", "pattern": r"^[A-Za-z][0-9]+$"},
                "minItems": 1,
                "required": True,
                "example": ["A45", "A46"],
                "description": "Chain letter followed by residue number.",
            },
            "num_samples": {
                "type": "integer",
                "minimum": 1,
                "maximum": 100,
                "default": 8,
            },
            "model_type": {
                "type": "string",
                "enum": ["protein", "ligand", "soluble"],
                "default": "protein",
            },
        },
    }
)

OK = {"target_pdb": "t.pdb", "hotspot_residues": ["A45"]}


def test_valid_input_passes_through():
    assert validate_and_fill(MANIFEST, OK)["target_pdb"] == "t.pdb"


def test_defaults_are_filled_because_models_omit_them():
    result = validate_and_fill(MANIFEST, OK)
    assert result["num_samples"] == 8
    assert result["model_type"] == "protein"


def test_explicit_value_beats_default():
    result = validate_and_fill(MANIFEST, {**OK, "num_samples": 20})
    assert result["num_samples"] == 20


def test_missing_required_field_is_reported():
    with pytest.raises(ToolInputError, match="target_pdb"):
        validate_and_fill(MANIFEST, {"hotspot_residues": ["A45"]})


def test_pattern_violation_shows_the_expected_format_and_an_example():
    with pytest.raises(ToolInputError) as exc:
        validate_and_fill(MANIFEST, {**OK, "target_pdb": "target.txt"})
    message = str(exc.value)
    assert "target_pdb" in message
    assert r"\.(pdb|cif)$" in message
    assert "target.pdb" in message


def test_item_pattern_violation_names_the_index():
    with pytest.raises(ToolInputError) as exc:
        validate_and_fill(MANIFEST, {**OK, "hotspot_residues": ["A45", "46"]})
    message = str(exc.value)
    assert "hotspot_residues[1]" in message
    assert "A45" in message


def test_below_minimum_is_rejected():
    with pytest.raises(ToolInputError, match="minimum"):
        validate_and_fill(MANIFEST, {**OK, "num_samples": 0})


def test_above_maximum_is_rejected():
    with pytest.raises(ToolInputError, match="maximum"):
        validate_and_fill(MANIFEST, {**OK, "num_samples": 101})


def test_enum_violation_lists_allowed_values():
    with pytest.raises(ToolInputError) as exc:
        validate_and_fill(MANIFEST, {**OK, "model_type": "rna"})
    assert "protein" in str(exc.value)


def test_min_items_is_enforced():
    with pytest.raises(ToolInputError, match="minItems"):
        validate_and_fill(MANIFEST, {**OK, "hotspot_residues": []})


def test_unknown_parameter_is_rejected():
    with pytest.raises(ToolInputError, match="unexpected"):
        validate_and_fill(MANIFEST, {**OK, "temperature": 0.2})


def test_wrong_type_is_rejected():
    with pytest.raises(ToolInputError, match="integer"):
        validate_and_fill(MANIFEST, {**OK, "num_samples": "eight"})


def test_bool_is_not_accepted_as_integer():
    with pytest.raises(ToolInputError, match="integer"):
        validate_and_fill(MANIFEST, {**OK, "num_samples": True})


def test_zero_is_a_valid_value_not_a_missing_one():
    manifest = parse_manifest(
        {
            "name": "run_zero",
            "category": "scoring",
            "engine": {"repo": "r", "env": "e", "entry": ["x"]},
            "summary": "Zero.",
            "doc": "## What this is\nZero.\n",
            "schema": {"seed": {"type": "integer", "minimum": 0, "default": 42}},
        }
    )
    assert validate_and_fill(manifest, {"seed": 0})["seed"] == 0
