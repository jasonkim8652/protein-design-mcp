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

# An OutputSpec.name is used both as a dict key in the collected-results
# payload and (dispatch/env.py -> results.collect_outputs) as a results
# subdirectory name. A single path-safe token is the honest constraint: it
# rules out "/abs" and "../escape" the same way an allowlist of characters
# always does, without trying to special-case every path-traversal shape.
OUTPUT_NAME_RE = re.compile(r"^[A-Za-z0-9_]+$")

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

    ``multiple`` opts a spec into collecting every file the pattern matches,
    returned as a list. Without it, a pattern matching more than one file is
    treated as ambiguous and rejected rather than silently picking one.
    """

    name: str
    pattern: str
    description: str = ""
    multiple: bool = False


@dataclass(frozen=True)
class EngineSpec:
    """How to invoke one engine.

    ``stage`` names schema parameters (each must be ``format: path``) whose
    files the dispatcher must COPY into the engine's scratch working
    directory before running, rewriting that parameter's value to the
    staged copy's path. This exists for engines that write their results
    next to one of their INPUT files rather than into the process's cwd
    (ipSAE is the first: it always writes beside the structure file it was
    given) — without staging, that input resolves to an absolute path
    outside the scratch directory (see app._resolve_path_params), so the
    engine's outputs would land outside it too, where a relative
    ``outputs:`` pattern can never see them and the containment check in
    results.collect_outputs would refuse them even if it could. Most
    engines need no staging at all, hence the empty default.
    """

    repo: str
    env: str
    entry: tuple[str, ...]
    stage: tuple[str, ...] = ()


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
    stage = data.get("stage")
    if stage is None:
        stage = []
    if not isinstance(stage, list) or not all(isinstance(x, str) for x in stage):
        raise ManifestError(f"{name}: engine.stage must be a list of strings")
    if len(set(stage)) != len(stage):
        raise ManifestError(f"{name}: engine.stage lists a parameter more than once")
    return EngineSpec(
        repo=str(_require(data, "repo")),
        env=str(_require(data, "env")),
        entry=tuple(entry),
        stage=tuple(stage),
    )


def _validate_stage(engine: EngineSpec, schema: dict, name: str) -> None:
    """Every ``engine.stage`` entry must name a real, path-typed parameter.

    Checked here (after both ``engine`` and ``schema`` are parsed) rather
    than inside ``_parse_engine``, which only ever sees the ``engine:``
    sub-mapping and has no visibility into ``schema:``.
    """
    for param_name in engine.stage:
        spec = schema.get(param_name)
        if spec is None:
            raise ManifestError(
                f"{name}: engine.stage names {param_name!r}, which is not a "
                "schema parameter"
            )
        if spec.get("format") != "path":
            raise ManifestError(
                f"{name}: engine.stage names {param_name!r}, which is not "
                "format: path — only a path parameter's file can be staged"
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
        out_name = str(out_name)
        if not OUTPUT_NAME_RE.match(out_name):
            raise ManifestError(
                f"{label}: name {out_name!r} must match "
                f"{OUTPUT_NAME_RE.pattern!r} — it is used as a dict key and "
                "as a results directory name, so it must be a single "
                "path-safe token (no '/', no '..', not empty)"
            )
        if out_name in seen:
            raise ManifestError(f"{name}: duplicate output name {out_name!r}")
        seen.add(out_name)

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
                multiple=bool(entry.get("multiple", False)),
            )
        )
    return tuple(specs)


def _parse_timeout(data: Any, name: str) -> int:
    if data is None:
        return DEFAULT_TIMEOUT_S
    # Reject booleans first: bool is a subclass of int, so int(True) == 1
    if isinstance(data, bool):
        raise ManifestError(
            f"{name}: timeout_s must be an integer, got {type(data).__name__}"
        )
    # Require strict int type to prevent silent truncation of floats
    if not isinstance(data, int):
        raise ManifestError(
            f"{name}: timeout_s must be an integer, got {type(data).__name__}"
        )
    if data <= 0:
        raise ManifestError(f"{name}: timeout_s must be positive, got {data}")
    return data


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

    engine = _parse_engine(_require(data, "engine"), name)
    _validate_stage(engine, schema, name)

    max_residues = data.get("max_residues")
    return Manifest(
        name=name,
        category=category,
        engine=engine,
        summary=summary,
        doc=str(_require(data, "doc")),
        schema=schema,
        composite=bool(data.get("composite", False)),
        requires=_parse_requires(data.get("requires"), name),
        max_residues=int(max_residues) if max_residues is not None else None,
        outputs=_parse_outputs(data.get("outputs"), name),
        timeout_s=_parse_timeout(data.get("timeout_s"), name),
    )
