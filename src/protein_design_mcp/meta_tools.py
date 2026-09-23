"""Meta-tools that expose documentation through an ordinary tool call.

Many MCP clients never surface resources to the model. Tool names in this
server are deliberately mechanical (``run_<engine>_<step>``), so if
documentation were reachable only as a resource, those clients would leave the
model with a bare name and no semantics. ``describe_tool`` is the portable
path.
"""

from __future__ import annotations

from typing import Any

from protein_design_mcp.job_status import get_design_status
from protein_design_mcp.manifest.loader import SIBLING_DOC_HEADING
from protein_design_mcp.manifest.registry import ToolNotAvailable, ToolRegistry
from protein_design_mcp.manifest.schema import CATEGORIES, parse_manifest

#: The categories ``describe_tool`` accepts, DERIVED from the registry's own
#: set rather than restated here. The hand-written copy that used to live below
#: drifted: ``target_analysis`` was added to ``CATEGORIES`` along with
#: run_interface_residues and run_epitope_scan, but not to the copy, so both
#: tools were registered and returned by ``tools/list`` while
#: ``describe_tool(category='target_analysis')`` was refused as an invalid
#: value -- the documented way to discover them denied they existed.
#: Sorted for a stable schema; a set would reorder between runs.
_CATEGORY_ENUM = sorted(CATEGORIES)

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
            "category (" + ", ".join(_CATEGORY_ENUM) + ")."
        ),
        "doc": (
            "## What this is\n"
            "A lookup over this server's tool documentation.\n\n"
            "## When to use it\n"
            "Before calling any tool you have not used, and whenever several tools "
            "look interchangeable. `category='structure_prediction'` returns every "
            "structure prediction tool side by side with the condition that selects "
            "each.\n\n"
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
                "enum": _CATEGORY_ENUM,
                "description": "Category to compare.",
                "example": "structure_prediction",
            },
        },
    }
)


GET_JOB_STATUS_MANIFEST = parse_manifest(
    {
        "name": "get_job_status",
        "category": "meta",
        # `get_job_status` has no engine at all -- see job_status.py and the
        # wave report for why. Same placeholder EngineSpec DESCRIBE_TOOL_MANIFEST
        # above uses: it is never passed to a dispatcher (call_tool special-cases
        # both meta-tools before manifest.resolve()/ADAPTERS are ever consulted),
        # so "repo"/"env"/"entry" here are inert labels, not a real invocation.
        "engine": {"repo": "builtin", "env": "server", "entry": ["builtin"]},
        "summary": (
            "Check the status of a long-running design job by the job_id it "
            "was started with. This server's generative and structure-"
            "prediction tools can run for minutes to hours; call this "
            "instead of blocking on the original tool call to see whether a "
            "job is queued, running, completed or failed, its progress so "
            "far, and (once at least one design has finished) an estimate "
            "of the time remaining."
        ),
        "doc": (
            "## What this is\n"
            "A lookup over one design job's status, tracked by job_id.\n\n"
            "## When to use it\n"
            "After starting a long-running generative or structure-"
            "prediction tool, to check whether it has finished without "
            "blocking on the original call.\n\n"
            "## What you must supply\n"
            "`job_id`, the identifier returned when the job was started.\n\n"
            "## What you get back\n"
            "`status` (`queued`, `running`, `completed`, or `failed`), "
            "`job_id`, `created_at`, and depending on status: `progress` "
            "(current step, designs completed/total, percent complete) and "
            "`estimated_time_remaining` for a running job; `result` for a "
            "completed job; `error` for a failed job.\n\n"
            "## How the time estimate works\n"
            "`estimated_time_remaining` is derived from THIS job's own "
            "observed progress rate (elapsed time since it started, divided "
            "by designs completed so far, projected across what remains) -- "
            "not a fixed per-engine timing table. It is omitted whenever "
            "that rate cannot be computed honestly yet, most commonly "
            "because no design has completed so far.\n"
        ),
        "schema": {
            "job_id": {
                "type": "string",
                "required": True,
                "description": (
                    "The job identifier returned when the long-running job "
                    "was started."
                ),
                "example": "a1b2c3d4",
            },
        },
    }
)


async def get_job_status(*, job_id: str) -> dict[str, Any]:
    """Look up one job's status by id.

    Thin wrapper over ``job_status.get_design_status`` that translates its
    "job not found" ``ValueError`` into the same {"error": ...} dict shape
    ``describe_tool`` already uses for every one of ITS failure modes, so
    ``app.call_tool`` can route both meta-tools' failures through the exact
    same ``_error_payload`` branch instead of one being a raised exception
    and the other a plain return value.
    """
    try:
        return await get_design_status(job_id=job_id)
    except ValueError as exc:
        return {"error": str(exc)}


def _describe_one(registry: ToolRegistry, name: str) -> dict[str, Any]:
    if name == DESCRIBE_TOOL_MANIFEST.name:
        # describe_tool is a built-in meta-tool, not something loaded from
        # manifests/*.yaml, so it is never in a registry's own manifest set.
        # Answer directly rather than failing registry.resolve() with
        # "unknown tool".
        manifest = DESCRIBE_TOOL_MANIFEST
    elif name == GET_JOB_STATUS_MANIFEST.name:
        # Same reasoning as describe_tool immediately above: get_job_status
        # has no engine and is never loaded from manifests/*.yaml either, so
        # it is equally absent from the registry's own manifest set. Without
        # this branch, describe_tool(name="get_job_status") would fail with
        # "unknown tool" even though the tool itself is callable -- a real
        # gap caught by driving this through the real describe_tool path
        # live rather than only unit-testing get_job_status in isolation.
        manifest = GET_JOB_STATUS_MANIFEST
    else:
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
