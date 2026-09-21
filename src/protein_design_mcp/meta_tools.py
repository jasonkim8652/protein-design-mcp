"""Meta-tools that expose documentation through an ordinary tool call.

Many MCP clients never surface resources to the model. Tool names in this
server are deliberately mechanical (``run_<engine>_<step>``), so if
documentation were reachable only as a resource, those clients would leave the
model with a bare name and no semantics. ``describe_tool`` is the portable
path.
"""

from __future__ import annotations

from typing import Any

from protein_design_mcp.manifest.loader import SIBLING_DOC_HEADING
from protein_design_mcp.manifest.registry import ToolNotAvailable, ToolRegistry
from protein_design_mcp.manifest.schema import parse_manifest

DESCRIBE_TOOL_MANIFEST = parse_manifest(
    {
        "name": "describe_tool",
        "category": "meta",
        "engine": {"repo": "builtin", "env": "server", "entry": ["builtin"]},
        "summary": (
            "Read the full documentation for a tool, or compare every tool in a "
            "category. Tool names here name the engine they run and nothing more, "
            "so call this before choosing between similar tools. Pass exactly one of: "
            "name= for one tool, or category= for a comparison of all tools in that "
            "category (generation, monomer_generation, sequence_design, cofolding, scoring)."
        ),
        "doc": (
            "## What this is\n"
            "A lookup over this server's tool documentation.\n\n"
            "## When to use it\n"
            "Before calling any tool you have not used, and whenever several tools "
            "look interchangeable. `category='cofolding'` returns every structure "
            "prediction tool side by side with the condition that selects each.\n\n"
            "## What you must supply\n"
            "Exactly one of `name` or `category`.\n"
        ),
        "schema": {
            "name": {
                "type": "string",
                "description": "Tool to document, e.g. 'run_chai1'.",
                "example": "run_chai1",
            },
            "category": {
                "type": "string",
                "enum": [
                    "generation",
                    "monomer_generation",
                    "sequence_design",
                    "cofolding",
                    "scoring",
                    "meta",
                ],
                "description": "Category to compare.",
                "example": "cofolding",
            },
        },
    }
)


def _describe_one(registry: ToolRegistry, name: str) -> dict[str, Any]:
    try:
        manifest = registry.resolve(name)
    except ToolNotAvailable as exc:
        return {
            "error": str(exc),
            "available": [tool.name for tool in registry.tools()],
        }

    def _param_spec(spec: dict[str, Any]) -> dict[str, Any]:
        param = {
            "type": spec.get("type"),
            "required": bool(spec.get("required", False)),
            "description": spec.get("description", ""),
        }
        if "default" in spec:
            param["default"] = spec["default"]
        if "example" in spec:
            param["example"] = spec["example"]
        return param

    return {
        "name": manifest.name,
        "category": manifest.category,
        "summary": manifest.summary,
        "doc": manifest.doc,
        "engine": manifest.engine.repo,
        "parameters": {
            key: _param_spec(spec)
            for key, spec in manifest.schema.items()
        },
    }


def _extract_sibling_section(doc: str) -> str:
    """Extract the sibling comparison section, or return full doc if not present."""
    if SIBLING_DOC_HEADING not in doc:
        # Fallback: return full doc for single-member categories (no sibling section)
        return doc
    idx = doc.find(SIBLING_DOC_HEADING)
    return doc[idx:]


def _describe_category(registry: ToolRegistry, category: str) -> dict[str, Any]:
    members = registry.by_category(category)
    if not members:
        return {
            "error": f"no available tools in category {category!r}",
            "available": registry.categories(),
        }
    return {
        "category": category,
        "tools": [
            {
                "name": m.name,
                "summary": m.summary,
                "engine": m.engine.repo,
                "doc": _extract_sibling_section(m.doc),
            }
            for m in members
        ],
        "note": "Call describe_tool(name=...) for the full documentation of any tool.",
    }


def describe_tool(
    registry: ToolRegistry,
    *,
    name: str | None = None,
    category: str | None = None,
) -> dict[str, Any]:
    """Document one tool, or compare a category. Exactly one argument."""
    if bool(name) == bool(category):
        return {
            "error": "pass exactly one of 'name' or 'category'",
            "available": registry.categories(),
        }
    if name:
        return _describe_one(registry, name)
    return _describe_category(registry, str(category))
