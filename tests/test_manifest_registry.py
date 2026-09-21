import pytest

from protein_design_mcp.manifest.registry import ToolNotAvailable, ToolRegistry
from protein_design_mcp.manifest.schema import parse_manifest

BASE = {
    "category": "scoring",
    "engine": {"repo": "r", "env": "e", "entry": ["x"]},
    "summary": "Summary text.",
    "doc": "## What this is\nDoc.\n",
    "schema": {
        "pdb": {"type": "string", "required": True, "example": "a.pdb"},
        "n": {"type": "integer", "default": 4, "minimum": 1, "maximum": 10},
    },
}


def _m(name, **over):
    return parse_manifest({**BASE, "name": name, **over})


def test_plain_tool_is_listed_and_resolvable():
    reg = ToolRegistry([_m("run_prodigy")])
    assert [t.name for t in reg.tools()] == ["run_prodigy"]
    assert reg.resolve("run_prodigy").name == "run_prodigy"


def test_composite_tool_is_not_listed():
    reg = ToolRegistry([_m("run_boltzgen_run", composite=True)])
    assert reg.tools() == []


def test_composite_tool_is_not_dispatchable():
    reg = ToolRegistry([_m("run_boltzgen_run", composite=True)])
    with pytest.raises(ToolNotAvailable, match="composite"):
        reg.resolve("run_boltzgen_run")


def test_gpu_tool_is_hidden_and_blocked_on_cpu():
    reg = ToolRegistry([_m("run_boltz", requires={"gpu": True})], device="cpu")
    assert reg.tools() == []
    with pytest.raises(ToolNotAvailable, match="GPU"):
        reg.resolve("run_boltz")


def test_gpu_tool_is_available_on_cuda():
    reg = ToolRegistry([_m("run_boltz", requires={"gpu": True})], device="cuda")
    assert [t.name for t in reg.tools()] == ["run_boltz"]


def test_license_gated_tool_is_hidden_when_not_licensed():
    reg = ToolRegistry(
        [_m("run_rosetta_interface", requires={"license_gated": True})],
        licensed=frozenset(),
    )
    assert reg.tools() == []
    with pytest.raises(ToolNotAvailable, match="licens"):
        reg.resolve("run_rosetta_interface")


def test_license_gated_tool_appears_when_licensed():
    reg = ToolRegistry(
        [_m("run_rosetta_interface", requires={"license_gated": True})],
        licensed=frozenset({"run_rosetta_interface"}),
    )
    assert [t.name for t in reg.tools()] == ["run_rosetta_interface"]


def test_unknown_tool_raises():
    reg = ToolRegistry([_m("run_prodigy")])
    with pytest.raises(ToolNotAvailable, match="unknown"):
        reg.resolve("run_nonexistent")


def test_input_schema_marks_required_and_forbids_extras():
    reg = ToolRegistry([_m("run_prodigy")])
    schema = reg.tools()[0].inputSchema
    assert schema["type"] == "object"
    assert schema["required"] == ["pdb"]
    assert schema["additionalProperties"] is False
    assert "required" not in schema["properties"]["pdb"]
    assert schema["properties"]["n"]["default"] == 4


def test_description_is_the_summary():
    reg = ToolRegistry([_m("run_prodigy")])
    assert reg.tools()[0].description == "Summary text."


def test_excluded_explains_why():
    reg = ToolRegistry([_m("run_boltzgen_run", composite=True)])
    assert "composite" in reg.excluded("run_boltzgen_run")
    assert reg.excluded("run_prodigy") is None


def test_by_category_lists_available_members_only():
    reg = ToolRegistry([_m("run_prodigy"), _m("run_hidden", composite=True)])
    assert [m.name for m in reg.by_category("scoring")] == ["run_prodigy"]
