import pytest

from protein_design_mcp.manifest.registry import ToolRegistry
from protein_design_mcp.manifest.schema import parse_manifest
from protein_design_mcp.meta_tools import DESCRIBE_TOOL_MANIFEST, describe_tool

HEADING = "## When to use this instead of the alternatives"


def _m(name, category="cofolding", **over):
    return parse_manifest(
        {
            "name": name,
            "category": category,
            "engine": {"repo": name, "env": "e", "entry": ["x"]},
            "summary": f"Summary for {name}.",
            "doc": f"## What this is\n{name}.\n\n{HEADING}\nUse the other one otherwise.\n",
            "schema": {"seq": {"type": "string", "required": True, "example": "MKT"}},
            **over,
        }
    )


REGISTRY = ToolRegistry([_m("run_chai1"), _m("run_boltz"), _m("run_prodigy", category="scoring")])


def test_named_tool_returns_its_full_document():
    result = describe_tool(REGISTRY, name="run_chai1")
    assert result["name"] == "run_chai1"
    assert "run_chai1." in result["doc"]
    assert HEADING in result["doc"]


def test_named_tool_includes_its_parameters():
    result = describe_tool(REGISTRY, name="run_chai1")
    assert result["parameters"]["seq"]["required"] is True
    assert result["parameters"]["seq"]["example"] == "MKT"


def test_named_tool_omits_absent_default_and_example():
    # Verify that absent default/example are not in the response
    registry = ToolRegistry([_m("run_test", schema={"arg": {"type": "string", "required": True}})])
    result = describe_tool(registry, name="run_test")
    assert "default" not in result["parameters"]["arg"]
    assert "example" not in result["parameters"]["arg"]
    assert "required" in result["parameters"]["arg"]  # required is always present


def test_named_tool_includes_default_when_present():
    registry = ToolRegistry([_m("run_test", schema={"arg": {"type": "string", "default": "val"}})])
    result = describe_tool(registry, name="run_test")
    assert result["parameters"]["arg"]["default"] == "val"


def test_named_tool_includes_default_null_when_present():
    registry = ToolRegistry([_m("run_test", schema={"arg": {"type": "string", "default": None}})])
    result = describe_tool(registry, name="run_test")
    assert "default" in result["parameters"]["arg"]
    assert result["parameters"]["arg"]["default"] is None


def test_category_mode_lists_every_sibling_with_its_summary():
    result = describe_tool(REGISTRY, category="cofolding")
    names = {tool["name"] for tool in result["tools"]}
    assert names == {"run_boltz", "run_chai1"}
    assert all("summary" in tool for tool in result["tools"])


def test_category_mode_reports_the_category_back():
    assert describe_tool(REGISTRY, category="cofolding")["category"] == "cofolding"


def test_category_mode_returns_only_sibling_section_not_full_doc():
    result = describe_tool(REGISTRY, category="cofolding")
    for tool in result["tools"]:
        # Should contain the sibling section
        assert HEADING in tool["doc"]
        # Should NOT contain the full "## What this is" section
        assert "## What this is" not in tool["doc"]


def test_category_mode_includes_note_about_full_docs():
    result = describe_tool(REGISTRY, category="cofolding")
    assert "note" in result
    assert "describe_tool(name=" in result["note"]


def test_category_mode_fallback_full_doc_for_single_member():
    # When a category has only one member without a sibling section header,
    # fall back to the full doc (since there are no "siblings" to compare)
    registry = ToolRegistry([_m("run_solo", category="scoring", doc="No sibling section here.")])
    result = describe_tool(registry, category="scoring")
    assert result["tools"][0]["doc"] == "No sibling section here."


def test_unknown_tool_lists_available_names():
    result = describe_tool(REGISTRY, name="run_nope")
    assert "error" in result
    assert "run_chai1" in result["available"]


def test_excluded_tool_explains_why_rather_than_pretending_it_is_missing():
    registry = ToolRegistry([_m("run_boltzgen_run", composite=True)])
    result = describe_tool(registry, name="run_boltzgen_run")
    assert "composite" in result["error"]


def test_unknown_category_lists_known_categories():
    result = describe_tool(REGISTRY, category="nope")
    assert "error" in result
    assert "cofolding" in result["available"]


def test_requires_one_of_name_or_category():
    result = describe_tool(REGISTRY)
    assert "error" in result


def test_name_and_category_together_is_rejected():
    result = describe_tool(REGISTRY, name="run_chai1", category="cofolding")
    assert "error" in result


def test_the_meta_tool_manifest_is_itself_valid():
    assert DESCRIBE_TOOL_MANIFEST.name == "describe_tool"
    assert DESCRIBE_TOOL_MANIFEST.category == "meta"
    assert DESCRIBE_TOOL_MANIFEST.composite is False
