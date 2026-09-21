"""Parse tool manifests into frozen dataclasses.

A manifest is the single source of truth for one tool: its MCP schema, its
documentation, and the engine invocation behind it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
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


class ManifestError(ValueError):
    """A manifest is malformed."""


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
    name: str
    category: str
    engine: EngineSpec
    summary: str
    doc: str
    schema: dict[str, Any]
    composite: bool = False
    requires: Requirements = field(default_factory=Requirements)
    max_residues: int | None = None


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
    if len(summary) > MAX_SUMMARY_CHARS:
        raise ManifestError(
            f"{name}: summary is {len(summary)} chars, max {MAX_SUMMARY_CHARS}"
        )

    schema = _require(data, "schema")
    if not isinstance(schema, dict):
        raise ManifestError(f"{name}: schema must be a mapping")

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
    )
