"""Parse tool manifests into frozen dataclasses.

A manifest is the single source of truth for one tool: its MCP schema, its
documentation, and the engine invocation behind it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

TOOL_NAME_RE = re.compile(r"^(run_[a-z0-9_]+|describe_tool|get_job_status)$")

CATEGORIES = frozenset(
    {
        "generation",
        "monomer_generation",
        "sequence_design",
        "cofolding",
        "scoring",
        "meta",
    }
)

MAX_SUMMARY_CHARS = 1024

DEFAULT_TIMEOUT_S = 3600


class ManifestError(ValueError):
    """A manifest is malformed."""


@dataclass(frozen=True)
class OutputSpec:
    """A file an engine writes into its scratch directory.

    ``pattern`` is a path relative to the scratch directory. Declaring outputs
    is what lets the dispatcher collect results and then remove the workdir;
    an engine whose results are only on stdout declares none.
    """

    name: str
    pattern: str
    description: str = ""


@dataclass(frozen=True)
class EngineSpec:
    repo: str
    env: str
    entry: tuple[str, ...]


@dataclass(frozen=True)
class Requirements:
    gpu: bool = False
    weights: str | None = None
    license_gated: bool = False


@dataclass(frozen=True)
class Manifest:
    """Tool manifest specification.

    The dataclass is frozen at the attribute level, but `schema` holds a
    mutable mapping. Callers must treat the schema dict as read-only and
    not modify its contents.
    """
    name: str
    category: str
    engine: EngineSpec
    summary: str
    doc: str
    schema: dict[str, Any]
    composite: bool = False
    requires: Requirements = field(default_factory=Requirements)
    max_residues: int | None = None
    outputs: tuple[OutputSpec, ...] = ()
    timeout_s: int = DEFAULT_TIMEOUT_S


def _require(data: dict, key: str) -> Any:
    if key not in data or data[key] in (None, "", [], {}):
        raise ManifestError(f"manifest is missing required key {key!r}")
    return data[key]


def _parse_engine(data: Any, name: str) -> EngineSpec:
    if not isinstance(data, dict):
        raise ManifestError(f"{name}: engine must be a mapping")
    entry = _require(data, "entry")
    if not isinstance(entry, list) or not all(isinstance(x, str) for x in entry):
        raise ManifestError(f"{name}: engine.entry must be a list of strings")
    return EngineSpec(
        repo=str(_require(data, "repo")),
        env=str(_require(data, "env")),
        entry=tuple(entry),
    )


def _parse_requires(data: Any, name: str) -> Requirements:
    if data is None:
        return Requirements()
    if not isinstance(data, dict):
        raise ManifestError(f"{name}: requires must be a mapping")
    weights = data.get("weights")
    return Requirements(
        gpu=bool(data.get("gpu", False)),
        weights=str(weights) if weights else None,
        license_gated=bool(data.get("license_gated", False)),
    )


def _validate_schema_entries(schema: dict, name: str) -> None:
    """Reject a malformed ``schema`` entry before it can reach the registry.

    ``ToolRegistry.tools()`` and ``validation.py`` both assume every schema
    entry is itself a mapping (e.g. they call ``spec.items()`` /
    ``spec.get(...)``). A single manifest typo like ``schema: {p: "string"}``
    used to parse cleanly and then raise ``AttributeError`` deep inside
    ``ToolRegistry.tools()`` — at 29 manifests loaded from one directory,
    that AttributeError takes down ``tools/list`` for every tool, not just
    the malformed one.

    Also reject a ``minimum``/``maximum`` with no ``type``: validation.py's
    range check only fires for ``isinstance(value, (int, float))``, and in
    Python ``bool`` is an ``int`` subclass, so a typeless numeric spec would
    let ``True`` silently pass as ``1``.
    """
    for key, spec in schema.items():
        if not isinstance(spec, dict):
            raise ManifestError(
                f"{name}: schema entry {key!r} must be a mapping, got "
                f"{type(spec).__name__}"
            )
        if ("minimum" in spec or "maximum" in spec) and "type" not in spec:
            raise ManifestError(
                f"{name}: schema entry {key!r} has a minimum/maximum "
                "constraint but no 'type'; add type: integer or type: "
                "number (a typeless numeric spec lets a bool pass as 1/0)"
            )


def _parse_outputs(data: Any, name: str) -> tuple[OutputSpec, ...]:
    if data is None:
        return ()
    if not isinstance(data, list):
        raise ManifestError(f"{name}: outputs must be a list")

    specs: list[OutputSpec] = []
    seen: set[str] = set()
    for index, entry in enumerate(data):
        label = f"{name}: outputs[{index}]"
        if not isinstance(entry, dict):
            raise ManifestError(f"{label} must be a mapping")

        out_name = entry.get("name")
        if not out_name:
            raise ManifestError(f"{label} is missing required key 'name'")
        if out_name in seen:
            raise ManifestError(f"{name}: duplicate output name {out_name!r}")
        seen.add(str(out_name))

        pattern = entry.get("pattern")
        if not pattern:
            raise ManifestError(f"{label} is missing required key 'pattern'")
        pattern = str(pattern)
        if pattern.startswith("/") or ".." in Path(pattern).parts:
            raise ManifestError(
                f"{label}: pattern {pattern!r} must be relative to the scratch "
                "directory and must not escape it"
            )

        specs.append(
            OutputSpec(
                name=str(out_name),
                pattern=pattern,
                description=str(entry.get("description", "")),
            )
        )
    return tuple(specs)


def _parse_timeout(data: Any, name: str) -> int:
    if data is None:
        return DEFAULT_TIMEOUT_S
    try:
        value = int(data)
    except (TypeError, ValueError) as exc:
        raise ManifestError(f"{name}: timeout_s must be an integer") from exc
    if value <= 0:
        raise ManifestError(f"{name}: timeout_s must be positive, got {value}")
    return value


def parse_manifest(data: dict) -> Manifest:
    """Parse one manifest mapping. Raises ManifestError if malformed."""
    if not isinstance(data, dict):
        raise ManifestError("manifest must be a mapping")

    name = str(_require(data, "name"))
    if not TOOL_NAME_RE.match(name):
        raise ManifestError(
            f"invalid tool name {name!r}: must match {TOOL_NAME_RE.pattern}"
        )

    category = str(_require(data, "category"))
    if category not in CATEGORIES:
        raise ManifestError(
            f"{name}: unknown category {category!r}; "
            f"expected one of {sorted(CATEGORIES)}"
        )

    summary = str(_require(data, "summary")).strip()
    if not summary:
        raise ManifestError(f"{name}: summary cannot be empty after stripping whitespace")
    if len(summary) > MAX_SUMMARY_CHARS:
        raise ManifestError(
            f"{name}: summary is {len(summary)} chars, max {MAX_SUMMARY_CHARS}"
        )

    if "schema" not in data:
        raise ManifestError("manifest is missing required key 'schema'")
    schema = data["schema"]
    if not isinstance(schema, dict):
        raise ManifestError(f"{name}: schema must be a mapping")
    _validate_schema_entries(schema, name)

    max_residues = data.get("max_residues")
    return Manifest(
        name=name,
        category=category,
        engine=_parse_engine(_require(data, "engine"), name),
        summary=summary,
        doc=str(_require(data, "doc")),
        schema=schema,
        composite=bool(data.get("composite", False)),
        requires=_parse_requires(data.get("requires"), name),
        max_residues=int(max_residues) if max_residues is not None else None,
        outputs=_parse_outputs(data.get("outputs"), name),
        timeout_s=_parse_timeout(data.get("timeout_s"), name),
    )
