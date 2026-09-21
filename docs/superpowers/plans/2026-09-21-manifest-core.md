# Manifest-Driven Tool Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hardcoded 19-tool registry and 19-arm dispatch chain with a manifest-driven core that makes composite tools unregistrable, validates inputs server-side, and dispatches engines into isolated conda environments — proven end-to-end with one real engine.

**Architecture:** One YAML manifest per tool is the single source of truth. A loader parses manifests into frozen dataclasses; a registry filters them (composite, GPU, license-gated) and derives both the MCP `Tool` objects and the dispatch table from the same filtered set, so a tool that is hidden is also uncallable. A validation layer re-checks every schema constraint server-side and fills defaults, because client-side JSON Schema enforcement varies by model. An `EnvDispatcher` runs each engine via `micromamba run -n <env>` against a shared scratch directory.

**Tech Stack:** Python >=3.10, `mcp` SDK 1.25.0 (low-level `Server` API), PyYAML, pytest.

**Spec:** `docs/superpowers/specs/2026-09-21-atomistic-tool-refresh-design.md`

**Scope:** This is plan 1 of 2. It implements spec §4 (manifest architecture), §6 (harness/model compatibility), §5.2 (dispatch contract), and one engine end-to-end. Plan 2 covers spec §5.1/§5.3 (pixi multi-environment build, Dockerfile) and the remaining 28 engine adapters, and should not be written until this plan's dispatch contract is proven.

## Global Constraints

- Branch: `dev` in `~/projects/protein-design-mcp-dev`. Never touch `~/projects/protein-design-mcp` — it holds uncommitted work.
- Python `>=3.10` (`pyproject.toml:15`). Do not raise this floor; engine environments carry their own Python versions.
- `mcp>=1.25,<2` — replaces the effectively-unpinned `mcp>=0.1.0` at `pyproject.toml:38`.
- Manifests live in `manifests/*.yaml` at repository root. The directory is overridable via `PROTEIN_MCP_MANIFEST_DIR`.
- Tool names match `^(run_[a-z0-9_]+|describe_tool|get_job_status)$`.
- Every manifest in a category with more than one member must contain the literal heading `## When to use this instead of the alternatives` in its `doc`. This is spec §4.3 made machine-checkable.
- Composite tools are never registered and never dispatchable. Both properties derive from one filtered set.
- No tool may accept a parameter the engine silently ignores. If an engine has no seed, the manifest must not declare one.
- Existing tests that cover retained pipelines must stay green. Tests for removed tools are deleted in Task 10.
- TDD per `/home/jk661/CLAUDE.md`: failing test, run it, minimal implementation, run it, commit.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/protein_design_mcp/manifest/schema.py` | `Manifest`, `EngineSpec`, `Requirements` dataclasses; parse one dict into a `Manifest`; raise `ManifestError` on malformed input |
| `src/protein_design_mcp/manifest/loader.py` | Discover `*.yaml` in a directory, parse each, enforce cross-manifest rules (unique names, sibling-doc rule) |
| `src/protein_design_mcp/manifest/registry.py` | `ToolRegistry`: hold manifests, apply availability filters, build `mcp.types.Tool`, resolve a name to a manifest for dispatch |
| `src/protein_design_mcp/validation.py` | `validate_and_fill(manifest, arguments)` — server-side constraint enforcement and default application; `ToolInputError` |
| `src/protein_design_mcp/dispatch/serialize.py` | `to_jsonable(obj)` — numpy-safe JSON coercion |
| `src/protein_design_mcp/dispatch/env.py` | `EnvDispatcher` — build the `micromamba run` command line, execute, capture, surface failures |
| `src/protein_design_mcp/meta_tools.py` | `describe_tool` handler — single-tool document and category comparison |
| `manifests/run_prodigy.yaml` | First real engine manifest |
| `src/protein_design_mcp/adapters/prodigy.py` | Argument translation for PRODIGY |
| `scripts/generate_tool_docs.py` | Render `docs/tools/*.md` from manifests |

`server.py` is modified, not replaced: the `TOOLS` literal and the `call_tool` if/elif chain are deleted and replaced with registry lookups.

---

### Task 1: Manifest schema and parsing

**Files:**
- Create: `src/protein_design_mcp/manifest/__init__.py`
- Create: `src/protein_design_mcp/manifest/schema.py`
- Test: `tests/test_manifest_schema.py`

**Interfaces:**
- Consumes: nothing
- Produces: `Manifest`, `EngineSpec`, `Requirements`, `ManifestError`, `parse_manifest(data: dict) -> Manifest`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_manifest_schema.py
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
    assert m.engine.entry == ["prodigy"]
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_manifest_schema.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'protein_design_mcp.manifest'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/protein_design_mcp/manifest/__init__.py
"""Manifest-driven tool definitions."""
```

```python
# src/protein_design_mcp/manifest/schema.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_manifest_schema.py -v`
Expected: PASS, 7 tests

- [ ] **Step 5: Commit**

```bash
git add src/protein_design_mcp/manifest/ tests/test_manifest_schema.py
git commit -m "feat: add tool manifest schema and parser"
```

---

### Task 2: Manifest loader with cross-manifest rules

**Files:**
- Create: `src/protein_design_mcp/manifest/loader.py`
- Test: `tests/test_manifest_loader.py`

**Interfaces:**
- Consumes: `parse_manifest`, `Manifest`, `ManifestError` from Task 1
- Produces: `load_manifests(directory: Path) -> list[Manifest]`, `SIBLING_DOC_HEADING`

The sibling-doc rule is the machine-checkable form of spec §4.3: any category
holding more than one tool forces every member to document how to choose
between them.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_manifest_loader.py
import textwrap

import pytest

from protein_design_mcp.manifest.loader import SIBLING_DOC_HEADING, load_manifests
from protein_design_mcp.manifest.schema import ManifestError


def _write(tmp_path, filename, body):
    path = tmp_path / filename
    path.write_text(textwrap.dedent(body))
    return path


SOLO = """\
    name: run_prodigy
    category: scoring
    engine: {repo: prodigy, env: scoring, entry: [prodigy]}
    summary: Estimate binding free energy.
    doc: |
      ## What this is
      PRODIGY.
    schema:
      complex_pdb: {type: string, required: true, example: c.pdb}
"""


def test_loads_a_single_manifest(tmp_path):
    _write(tmp_path, "run_prodigy.yaml", SOLO)
    manifests = load_manifests(tmp_path)
    assert [m.name for m in manifests] == ["run_prodigy"]


def test_empty_directory_returns_empty_list(tmp_path):
    assert load_manifests(tmp_path) == []


def test_duplicate_names_are_rejected(tmp_path):
    _write(tmp_path, "a.yaml", SOLO)
    _write(tmp_path, "b.yaml", SOLO)
    with pytest.raises(ManifestError, match="duplicate"):
        load_manifests(tmp_path)


def test_siblings_must_document_how_to_choose(tmp_path):
    _write(tmp_path, "a.yaml", SOLO)
    _write(
        tmp_path,
        "b.yaml",
        SOLO.replace("run_prodigy", "run_ipsae"),
    )
    with pytest.raises(ManifestError, match=SIBLING_DOC_HEADING):
        load_manifests(tmp_path)


def test_siblings_pass_when_they_document_the_choice(tmp_path):
    with_heading = SOLO.replace(
        "      PRODIGY.\n",
        f"      PRODIGY.\n\n      {SIBLING_DOC_HEADING}\n      Use run_ipsae instead when you already have a PAE matrix.\n",
    )
    _write(tmp_path, "a.yaml", with_heading)
    _write(tmp_path, "b.yaml", with_heading.replace("run_prodigy", "run_ipsae"))
    assert len(load_manifests(tmp_path)) == 2


def test_solo_category_needs_no_comparison_heading(tmp_path):
    _write(tmp_path, "run_prodigy.yaml", SOLO)
    assert len(load_manifests(tmp_path)) == 1


def test_malformed_yaml_names_the_file(tmp_path):
    _write(tmp_path, "broken.yaml", "name: designBinder\n")
    with pytest.raises(ManifestError, match="broken.yaml"):
        load_manifests(tmp_path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_manifest_loader.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'protein_design_mcp.manifest.loader'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/protein_design_mcp/manifest/loader.py
"""Discover and load tool manifests from a directory."""

from __future__ import annotations

import collections
from pathlib import Path

import yaml

from protein_design_mcp.manifest.schema import Manifest, ManifestError, parse_manifest

SIBLING_DOC_HEADING = "## When to use this instead of the alternatives"


def _load_one(path: Path) -> Manifest:
    try:
        data = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise ManifestError(f"{path.name}: invalid YAML: {exc}") from exc
    try:
        return parse_manifest(data)
    except ManifestError as exc:
        raise ManifestError(f"{path.name}: {exc}") from exc


def _check_unique(manifests: list[Manifest]) -> None:
    counts = collections.Counter(m.name for m in manifests)
    duplicates = sorted(name for name, n in counts.items() if n > 1)
    if duplicates:
        raise ManifestError(f"duplicate tool names: {', '.join(duplicates)}")


def _check_sibling_docs(manifests: list[Manifest]) -> None:
    by_category: dict[str, list[Manifest]] = collections.defaultdict(list)
    for m in manifests:
        by_category[m.category].append(m)

    for category, members in sorted(by_category.items()):
        if len(members) < 2:
            continue
        for m in members:
            if SIBLING_DOC_HEADING not in m.doc:
                raise ManifestError(
                    f"{m.name}: category {category!r} has {len(members)} tools, so "
                    f"its doc must contain the heading {SIBLING_DOC_HEADING!r} "
                    "naming the sibling tools and when to prefer each"
                )


def load_manifests(directory: Path) -> list[Manifest]:
    """Load every ``*.yaml`` manifest in ``directory``, sorted by tool name."""
    directory = Path(directory)
    if not directory.is_dir():
        raise ManifestError(f"manifest directory not found: {directory}")

    manifests = [_load_one(p) for p in sorted(directory.glob("*.yaml"))]
    _check_unique(manifests)
    _check_sibling_docs(manifests)
    return sorted(manifests, key=lambda m: m.name)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_manifest_loader.py -v`
Expected: PASS, 7 tests

- [ ] **Step 5: Commit**

```bash
git add src/protein_design_mcp/manifest/loader.py tests/test_manifest_loader.py
git commit -m "feat: load manifests and enforce sibling-doc rule"
```

---

### Task 3: Registry — one filtered set drives listing and dispatch

**Files:**
- Create: `src/protein_design_mcp/manifest/registry.py`
- Test: `tests/test_manifest_registry.py`

**Interfaces:**
- Consumes: `Manifest`, `load_manifests` from Tasks 1-2
- Produces: `ToolRegistry(manifests, *, device, available_weights, licensed)` with
  `.tools() -> list[mcp.types.Tool]`, `.resolve(name) -> Manifest`,
  `.by_category(category) -> list[Manifest]`, `.excluded(name) -> str | None`,
  and `ToolNotAvailable`

This is the task that closes the `COMPOSITE_TOOL_NAMES` bug. `tools()` and
`resolve()` read the same `_available` dict, so a tool absent from the listing
cannot be invoked by name.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_manifest_registry.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_manifest_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'protein_design_mcp.manifest.registry'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/protein_design_mcp/manifest/registry.py
"""Hold manifests, decide which are available, and derive the MCP surface.

``tools()`` and ``resolve()`` read the same ``_available`` mapping. A tool that
is filtered out of the listing therefore cannot be invoked by name either,
which is the property the old ``COMPOSITE_TOOL_NAMES`` constant failed to
provide.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from mcp.types import Tool

from protein_design_mcp.manifest.schema import Manifest

# Manifest schema keys that describe the field for humans/validation but are
# not part of the JSON Schema handed to the client.
_NON_SCHEMA_KEYS = frozenset({"required", "example"})


class ToolNotAvailable(KeyError):
    """The named tool is not registered, or is registered but unavailable."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return self.message


def _json_schema_for(manifest: Manifest) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    required: list[str] = []
    for key, spec in manifest.schema.items():
        properties[key] = {k: v for k, v in spec.items() if k not in _NON_SCHEMA_KEYS}
        if spec.get("required"):
            required.append(key)
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


class ToolRegistry:
    """Availability rules and derived MCP objects for a set of manifests."""

    def __init__(
        self,
        manifests: Iterable[Manifest],
        *,
        device: str = "cuda",
        available_weights: frozenset[str] = frozenset(),
        licensed: frozenset[str] = frozenset(),
    ) -> None:
        self._all: dict[str, Manifest] = {m.name: m for m in manifests}
        self._device = device
        self._available_weights = available_weights
        self._licensed = licensed
        self._reasons: dict[str, str] = {}
        self._available: dict[str, Manifest] = {}
        for name, manifest in self._all.items():
            reason = self._exclusion_reason(manifest)
            if reason is None:
                self._available[name] = manifest
            else:
                self._reasons[name] = reason

    def _exclusion_reason(self, manifest: Manifest) -> str | None:
        if manifest.composite:
            return (
                f"{manifest.name} is a composite pipeline and is not exposed. "
                "Call the individual steps instead so you control each stage."
            )
        if manifest.requires.gpu and self._device == "cpu":
            return (
                f"{manifest.name} requires a GPU but DEVICE is 'cpu'. "
                "Set DEVICE=cuda or run the GPU image."
            )
        if manifest.requires.license_gated and manifest.name not in self._licensed:
            return (
                f"{manifest.name} depends on a license-gated component that is not "
                "installed. See its documentation for how to supply it."
            )
        if manifest.requires.weights and manifest.requires.weights not in (
            self._available_weights
        ):
            return (
                f"{manifest.name} needs weights at {manifest.requires.weights!r}, "
                "which were not found."
            )
        return None

    def tools(self) -> list[Tool]:
        """MCP Tool objects for every available tool, sorted by name."""
        return [
            Tool(
                name=m.name,
                description=m.summary,
                inputSchema=_json_schema_for(m),
            )
            for m in sorted(self._available.values(), key=lambda m: m.name)
        ]

    def resolve(self, name: str) -> Manifest:
        """Return the manifest for ``name``, or raise ToolNotAvailable."""
        if name in self._available:
            return self._available[name]
        if name in self._reasons:
            raise ToolNotAvailable(self._reasons[name])
        raise ToolNotAvailable(f"unknown tool: {name!r}")

    def excluded(self, name: str) -> str | None:
        """Why ``name`` is unavailable, or None if it is available."""
        return self._reasons.get(name)

    def by_category(self, category: str) -> list[Manifest]:
        """Available manifests in ``category``, sorted by name."""
        return sorted(
            (m for m in self._available.values() if m.category == category),
            key=lambda m: m.name,
        )

    def categories(self) -> list[str]:
        return sorted({m.category for m in self._available.values()})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_manifest_registry.py -v`
Expected: PASS, 12 tests

- [ ] **Step 5: Commit**

```bash
git add src/protein_design_mcp/manifest/registry.py tests/test_manifest_registry.py
git commit -m "feat: registry derives listing and dispatch from one filtered set"
```

---

### Task 4: Server-side validation and defaults

**Files:**
- Create: `src/protein_design_mcp/validation.py`
- Test: `tests/test_validation.py`

**Interfaces:**
- Consumes: `Manifest` from Task 1
- Produces: `validate_and_fill(manifest, arguments) -> dict`, `ToolInputError`

Spec §6: Gemini's function calling does not enforce `pattern`, and no model
reliably applies schema `default`. Client-side validation is a hint; this is
the enforcement. Error messages must name the constraint and show a correct
example, because the caller is a language model that will retry.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_validation.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_validation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'protein_design_mcp.validation'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/protein_design_mcp/validation.py
"""Server-side enforcement of manifest schema constraints.

JSON Schema support differs across models: Gemini's function calling accepts a
restricted OpenAPI subset and does not enforce ``pattern``, and no provider
reliably applies ``default``. Client-side validation is therefore advisory.
This module is the actual boundary.

Error messages name the offending parameter, state the constraint, and show a
correct example, because the caller is a language model that will read the
error and retry.
"""

from __future__ import annotations

import re
from typing import Any

from protein_design_mcp.manifest.schema import Manifest

_TYPE_CHECKS = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
}


class ToolInputError(ValueError):
    """Input failed a manifest constraint."""


def _example_clause(spec: dict[str, Any]) -> str:
    if "example" in spec:
        return f" Example: {spec['example']!r}."
    return ""


def _check_type(label: str, value: Any, spec: dict[str, Any]) -> None:
    expected = spec.get("type")
    if expected is None:
        return
    check = _TYPE_CHECKS.get(expected)
    if check is not None and not check(value):
        raise ToolInputError(
            f"{label} must be of type {expected}, got "
            f"{type(value).__name__}.{_example_clause(spec)}"
        )


def _check_scalar(label: str, value: Any, spec: dict[str, Any]) -> None:
    _check_type(label, value, spec)

    pattern = spec.get("pattern")
    if pattern is not None and isinstance(value, str):
        if re.search(pattern, value) is None:
            raise ToolInputError(
                f"{label} = {value!r} does not match the required format "
                f"{pattern}.{_example_clause(spec)}"
            )

    enum = spec.get("enum")
    if enum is not None and value not in enum:
        raise ToolInputError(
            f"{label} = {value!r} is not allowed; expected one of "
            f"{list(enum)}.{_example_clause(spec)}"
        )

    minimum = spec.get("minimum")
    if minimum is not None and isinstance(value, (int, float)) and value < minimum:
        raise ToolInputError(f"{label} = {value!r} is below the minimum of {minimum}.")

    maximum = spec.get("maximum")
    if maximum is not None and isinstance(value, (int, float)) and value > maximum:
        raise ToolInputError(f"{label} = {value!r} is above the maximum of {maximum}.")


def _check_value(label: str, value: Any, spec: dict[str, Any]) -> None:
    _check_scalar(label, value, spec)

    if spec.get("type") == "array" and isinstance(value, list):
        min_items = spec.get("minItems")
        if min_items is not None and len(value) < min_items:
            raise ToolInputError(
                f"{label} has {len(value)} entries but minItems is {min_items}."
                f"{_example_clause(spec)}"
            )
        max_items = spec.get("maxItems")
        if max_items is not None and len(value) > max_items:
            raise ToolInputError(
                f"{label} has {len(value)} entries but maxItems is {max_items}."
            )
        item_spec = spec.get("items")
        if isinstance(item_spec, dict):
            merged = dict(item_spec)
            merged.setdefault("example", spec.get("example", [None])[0]
                              if isinstance(spec.get("example"), list) and spec["example"]
                              else None)
            if merged.get("example") is None:
                merged.pop("example", None)
            for index, item in enumerate(value):
                _check_scalar(f"{label}[{index}]", item, merged)


def validate_and_fill(manifest: Manifest, arguments: dict[str, Any]) -> dict[str, Any]:
    """Validate ``arguments`` against ``manifest`` and apply defaults.

    Returns a new dict. Raises ToolInputError with a message written for a
    model that will retry.
    """
    arguments = dict(arguments or {})

    unexpected = sorted(set(arguments) - set(manifest.schema))
    if unexpected:
        raise ToolInputError(
            f"{manifest.name} received unexpected parameter(s): "
            f"{', '.join(unexpected)}. Accepted parameters: "
            f"{', '.join(sorted(manifest.schema))}."
        )

    result: dict[str, Any] = {}
    for key, spec in manifest.schema.items():
        label = f"{manifest.name}.{key}"
        if key in arguments:
            value = arguments[key]
            _check_value(label, value, spec)
            result[key] = value
        elif "default" in spec:
            result[key] = spec["default"]
        elif spec.get("required"):
            description = spec.get("description", "")
            raise ToolInputError(
                f"{label} is required but was not provided. "
                f"{description}{_example_clause(spec)}".strip()
            )
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_validation.py -v`
Expected: PASS, 14 tests

- [ ] **Step 5: Commit**

```bash
git add src/protein_design_mcp/validation.py tests/test_validation.py
git commit -m "feat: enforce manifest constraints server-side and fill defaults"
```

---

### Task 5: numpy-safe JSON serialization

**Files:**
- Create: `src/protein_design_mcp/dispatch/__init__.py`
- Create: `src/protein_design_mcp/dispatch/serialize.py`
- Test: `tests/test_serialize.py`

**Interfaces:**
- Consumes: nothing
- Produces: `to_jsonable(obj) -> Any`

Scientific runners return numpy scalars and arrays. `json.dumps` raises
`TypeError: Object of type float32 is not JSON serializable` on them. The old
checkout fixed this in commit `d35e50d`, which was never pushed to `origin/main`
and is therefore absent from this branch.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_serialize.py
import json
import math

import numpy as np
import pytest

from protein_design_mcp.dispatch.serialize import to_jsonable


@pytest.mark.parametrize(
    "value,expected",
    [
        (np.float32(0.5), 0.5),
        (np.float64(1.5), 1.5),
        (np.int64(7), 7),
        (np.int32(-3), -3),
        (np.bool_(True), True),
    ],
)
def test_numpy_scalars_become_python_scalars(value, expected):
    result = to_jsonable(value)
    assert result == expected
    assert type(result) in (int, float, bool)


def test_numpy_array_becomes_nested_lists():
    assert to_jsonable(np.array([[1, 2], [3, 4]])) == [[1, 2], [3, 4]]


def test_nested_structures_are_converted():
    payload = {"scores": [np.float32(0.1)], "meta": {"n": np.int64(2)}}
    assert to_jsonable(payload) == {"scores": [0.1], "meta": {"n": 2}}


def test_result_is_json_serializable():
    payload = {"iptm": np.float32(0.83), "pae": np.zeros((2, 2))}
    json.dumps(to_jsonable(payload))


def test_nan_and_inf_become_none_because_json_has_no_literal():
    assert to_jsonable(np.float32("nan")) is None
    assert to_jsonable(float("inf")) is None
    assert to_jsonable(-math.inf) is None


def test_plain_values_pass_through_unchanged():
    assert to_jsonable({"a": 1, "b": "x", "c": None, "d": True}) == {
        "a": 1,
        "b": "x",
        "c": None,
        "d": True,
    }


def test_tuples_and_sets_become_lists():
    assert to_jsonable((1, 2)) == [1, 2]
    assert sorted(to_jsonable({1, 2})) == [1, 2]


def test_unknown_object_becomes_its_string_form():
    class Opaque:
        def __repr__(self) -> str:
            return "<opaque>"

    assert to_jsonable(Opaque()) == "<opaque>"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_serialize.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'protein_design_mcp.dispatch'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/protein_design_mcp/dispatch/__init__.py
"""Engine dispatch into isolated environments."""
```

```python
# src/protein_design_mcp/dispatch/serialize.py
"""Coerce scientific Python results into JSON-serializable values.

Engine runners return numpy scalars and arrays, which ``json.dumps`` rejects.
NaN and infinity are mapped to ``None`` because JSON has no literal for them
and emitting bare ``NaN`` produces output that strict parsers reject.
"""

from __future__ import annotations

import math
from typing import Any


def _is_nonfinite(value: Any) -> bool:
    return isinstance(value, float) and not math.isfinite(value)


def to_jsonable(obj: Any) -> Any:
    """Recursively convert ``obj`` into JSON-serializable values."""
    if obj is None or isinstance(obj, (str, bool)):
        return obj

    if isinstance(obj, int):
        return obj

    if isinstance(obj, float):
        return None if _is_nonfinite(obj) else obj

    # numpy scalars and arrays, without importing numpy at module scope.
    if hasattr(obj, "item") and hasattr(obj, "dtype") and getattr(obj, "shape", None) == ():
        return to_jsonable(obj.item())

    if hasattr(obj, "tolist") and hasattr(obj, "dtype"):
        return to_jsonable(obj.tolist())

    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_jsonable(v) for v in obj]

    return str(obj)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_serialize.py -v`
Expected: PASS, 15 tests (5 parametrized + 10)

- [ ] **Step 5: Commit**

```bash
git add src/protein_design_mcp/dispatch/ tests/test_serialize.py
git commit -m "feat: add numpy-safe JSON coercion for engine results"
```

---

### Task 6: EnvDispatcher

**Files:**
- Create: `src/protein_design_mcp/dispatch/env.py`
- Test: `tests/test_env_dispatcher.py`

**Interfaces:**
- Consumes: `Manifest`, `EngineSpec` from Task 1
- Produces: `EnvDispatcher(runner="micromamba", scratch_root=None)` with
  `.build_command(engine, args) -> list[str]`,
  `async .run(engine, args, *, timeout) -> CompletedRun`;
  `CompletedRun(returncode, stdout, stderr, workdir)`; `EngineError`

Tests use `sys.executable` as the runner so they execute anywhere. The real
`micromamba run -n <env>` prefix is exercised by `build_command`, which is pure.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_env_dispatcher.py
import sys

import pytest

from protein_design_mcp.dispatch.env import EngineError, EnvDispatcher
from protein_design_mcp.manifest.schema import EngineSpec

ENGINE = EngineSpec(repo="prodigy", env="scoring", entry=("prodigy",))


def test_command_is_wrapped_in_micromamba_run():
    d = EnvDispatcher()
    cmd = d.build_command(ENGINE, ["--input", "a.pdb"])
    assert cmd == [
        "micromamba",
        "run",
        "-n",
        "scoring",
        "prodigy",
        "--input",
        "a.pdb",
    ]


def test_multiword_entry_is_preserved():
    engine = EngineSpec(repo="complexa", env="complexa", entry=("complexa", "generate"))
    cmd = EnvDispatcher().build_command(engine, ["--n", "4"])
    assert cmd[4:] == ["complexa", "generate", "--n", "4"]


def test_runner_can_be_overridden_for_local_execution():
    d = EnvDispatcher(runner=None)
    assert d.build_command(ENGINE, ["-x"]) == ["prodigy", "-x"]


def test_arguments_are_stringified():
    cmd = EnvDispatcher(runner=None).build_command(ENGINE, ["--n", 4, "--t", 0.1])
    assert cmd == ["prodigy", "--n", "4", "--t", "0.1"]


@pytest.mark.asyncio
async def test_successful_run_captures_stdout(tmp_path):
    d = EnvDispatcher(runner=None, scratch_root=tmp_path)
    engine = EngineSpec(repo="py", env="unused", entry=(sys.executable,))
    result = await d.run(engine, ["-c", "print('hello')"], timeout=30)
    assert result.returncode == 0
    assert result.stdout.strip() == "hello"


@pytest.mark.asyncio
async def test_each_run_gets_its_own_scratch_directory(tmp_path):
    d = EnvDispatcher(runner=None, scratch_root=tmp_path)
    engine = EngineSpec(repo="py", env="unused", entry=(sys.executable,))
    a = await d.run(engine, ["-c", "pass"], timeout=30)
    b = await d.run(engine, ["-c", "pass"], timeout=30)
    assert a.workdir != b.workdir
    assert a.workdir.is_dir() and b.workdir.is_dir()


@pytest.mark.asyncio
async def test_nonzero_exit_raises_with_stderr_in_the_message(tmp_path):
    d = EnvDispatcher(runner=None, scratch_root=tmp_path)
    engine = EngineSpec(repo="py", env="unused", entry=(sys.executable,))
    with pytest.raises(EngineError) as exc:
        await d.run(
            engine,
            ["-c", "import sys; sys.stderr.write('boom'); sys.exit(3)"],
            timeout=30,
        )
    assert "boom" in str(exc.value)
    assert "3" in str(exc.value)


@pytest.mark.asyncio
async def test_timeout_raises_engine_error_naming_the_limit(tmp_path):
    d = EnvDispatcher(runner=None, scratch_root=tmp_path)
    engine = EngineSpec(repo="py", env="unused", entry=(sys.executable,))
    with pytest.raises(EngineError, match="timed out"):
        await d.run(engine, ["-c", "import time; time.sleep(10)"], timeout=1)


@pytest.mark.asyncio
async def test_cuda_oom_is_translated_into_actionable_advice(tmp_path):
    d = EnvDispatcher(runner=None, scratch_root=tmp_path)
    engine = EngineSpec(repo="py", env="unused", entry=(sys.executable,))
    script = (
        "import sys; sys.stderr.write("
        "'torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 75.94 GiB'"
        "); sys.exit(1)"
    )
    with pytest.raises(EngineError) as exc:
        await d.run(engine, ["-c", script], timeout=30)
    assert "out of memory" in str(exc.value).lower()
    assert "reduce" in str(exc.value).lower()


@pytest.mark.asyncio
async def test_missing_runner_reports_the_environment_name(tmp_path):
    d = EnvDispatcher(runner="definitely-not-a-real-binary", scratch_root=tmp_path)
    with pytest.raises(EngineError, match="scoring"):
        await d.run(ENGINE, [], timeout=30)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_env_dispatcher.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'protein_design_mcp.dispatch.env'`

Note: if the failure is instead `'asyncio' not found in markers`, add
`asyncio_mode = "auto"` under `[tool.pytest.ini_options]` in `pyproject.toml`
and install `pytest-asyncio`; then re-run and confirm the ModuleNotFoundError.

- [ ] **Step 3: Write minimal implementation**

```python
# src/protein_design_mcp/dispatch/env.py
"""Run an engine inside its own conda environment.

Each engine pins a Python, numpy and torch version that conflict with the
others (spec §5.1), so every engine lives in its own micromamba environment
inside one image and is invoked as a subprocess. This generalises the
``conda run -n <env>`` branch that previously existed only in
``pipelines/boltz_runner.py``.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from pathlib import Path
from tempfile import gettempdir
from typing import Any, Sequence

from protein_design_mcp.manifest.schema import EngineSpec

_OOM_MARKERS = ("out of memory", "outofmemoryerror", "cuda error: out of memory")


class EngineError(RuntimeError):
    """An engine subprocess failed, timed out, or could not be started."""


@dataclass(frozen=True)
class CompletedRun:
    returncode: int
    stdout: str
    stderr: str
    workdir: Path


class EnvDispatcher:
    """Build and execute ``micromamba run -n <env> <entry> <args>`` commands."""

    def __init__(
        self,
        runner: str | None = "micromamba",
        scratch_root: Path | None = None,
    ) -> None:
        self._runner = runner
        self._scratch_root = Path(scratch_root) if scratch_root else Path(gettempdir())

    def build_command(self, engine: EngineSpec, args: Sequence[Any]) -> list[str]:
        """Return the full argv. Pure — safe to assert on in tests."""
        prefix: list[str] = []
        if self._runner:
            prefix = [self._runner, "run", "-n", engine.env]
        return [*prefix, *engine.entry, *(str(a) for a in args)]

    def _make_workdir(self) -> Path:
        workdir = self._scratch_root / f"pdmcp-{uuid.uuid4().hex[:12]}"
        workdir.mkdir(parents=True, exist_ok=False)
        return workdir

    async def run(
        self,
        engine: EngineSpec,
        args: Sequence[Any],
        *,
        timeout: float,
    ) -> CompletedRun:
        """Execute the engine. Raises EngineError on any failure."""
        command = self.build_command(engine, args)
        workdir = self._make_workdir()

        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(workdir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except (FileNotFoundError, PermissionError) as exc:
            raise EngineError(
                f"could not start engine {engine.repo!r} in environment "
                f"{engine.env!r}: {exc}. Check that the environment exists "
                f"and that {command[0]!r} is on PATH."
            ) from exc

        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError as exc:
            process.kill()
            await process.wait()
            raise EngineError(
                f"engine {engine.repo!r} timed out after {timeout:.0f}s. "
                "Reduce the sample count or raise the tool's timeout."
            ) from exc

        stdout = stdout_b.decode("utf-8", errors="replace")
        stderr = stderr_b.decode("utf-8", errors="replace")

        if process.returncode != 0:
            lowered = stderr.lower()
            if any(marker in lowered for marker in _OOM_MARKERS):
                raise EngineError(
                    f"engine {engine.repo!r} ran out of GPU memory. Reduce the "
                    "number of samples, shorten the input, or use a smaller "
                    f"model variant.\n{stderr.strip()[-2000:]}"
                )
            raise EngineError(
                f"engine {engine.repo!r} exited with code {process.returncode}.\n"
                f"{stderr.strip()[-2000:]}"
            )

        return CompletedRun(
            returncode=process.returncode,
            stdout=stdout,
            stderr=stderr,
            workdir=workdir,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_env_dispatcher.py -v`
Expected: PASS, 10 tests

- [ ] **Step 5: Commit**

```bash
git add src/protein_design_mcp/dispatch/env.py tests/test_env_dispatcher.py
git commit -m "feat: add EnvDispatcher for cross-environment engine execution"
```

---

### Task 7: describe_tool

**Files:**
- Create: `src/protein_design_mcp/meta_tools.py`
- Test: `tests/test_describe_tool.py`

**Interfaces:**
- Consumes: `ToolRegistry`, `ToolNotAvailable` from Task 3
- Produces: `describe_tool(registry, *, name=None, category=None) -> dict`,
  `DESCRIBE_TOOL_MANIFEST` (a `Manifest` describing the meta-tool itself)

Spec §4.3 layer 2: many clients never surface MCP resources to the model, so
the full documentation must also be reachable through an ordinary tool call.
The category mode is the mitigation for eight similar co-folding tools.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_describe_tool.py
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


def test_category_mode_lists_every_sibling_with_its_summary():
    result = describe_tool(REGISTRY, category="cofolding")
    names = {tool["name"] for tool in result["tools"]}
    assert names == {"run_boltz", "run_chai1"}
    assert all("summary" in tool for tool in result["tools"])


def test_category_mode_reports_the_category_back():
    assert describe_tool(REGISTRY, category="cofolding")["category"] == "cofolding"


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_describe_tool.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'protein_design_mcp.meta_tools'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/protein_design_mcp/meta_tools.py
"""Meta-tools that expose documentation through an ordinary tool call.

Many MCP clients never surface resources to the model. Tool names in this
server are deliberately mechanical (``run_<engine>_<step>``), so if
documentation were reachable only as a resource, those clients would leave the
model with a bare name and no semantics. ``describe_tool`` is the portable
path.
"""

from __future__ import annotations

from typing import Any

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
            "so call this before choosing between similar tools. Pass name= for one "
            "tool, or category= for a comparison of all tools in that category "
            "(generation, monomer_generation, sequence_design, cofolding, scoring)."
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
    return {
        "name": manifest.name,
        "category": manifest.category,
        "summary": manifest.summary,
        "doc": manifest.doc,
        "engine": manifest.engine.repo,
        "parameters": {
            key: {
                "type": spec.get("type"),
                "required": bool(spec.get("required", False)),
                "default": spec.get("default"),
                "description": spec.get("description", ""),
                "example": spec.get("example"),
            }
            for key, spec in manifest.schema.items()
        },
    }


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
                "doc": m.doc,
            }
            for m in members
        ],
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_describe_tool.py -v`
Expected: PASS, 10 tests

- [ ] **Step 5: Commit**

```bash
git add src/protein_design_mcp/meta_tools.py tests/test_describe_tool.py
git commit -m "feat: add describe_tool for harnesses without MCP resources"
```

---

### Task 8: First engine — run_prodigy manifest and adapter

**Files:**
- Create: `manifests/run_prodigy.yaml`
- Create: `src/protein_design_mcp/adapters/__init__.py`
- Create: `src/protein_design_mcp/adapters/prodigy.py`
- Test: `tests/test_adapter_prodigy.py`

**Interfaces:**
- Consumes: `validate_and_fill`, `EnvDispatcher`, `CompletedRun`, `to_jsonable`
- Produces: `build_args(params: dict) -> list[str]`,
  `parse_output(run: CompletedRun) -> dict`,
  `async run_prodigy(dispatcher, manifest, arguments) -> dict`

PRODIGY is chosen as the contract-proving engine instead of `run_mpnn` (spec
§9 step 3) because it is CPU-only, pure Python plus Biopython, and needs no
model weights — so its test runs in CI. The dispatch contract it proves is the
same one every GPU engine will use. `run_mpnn` moves to plan 2, where
`dauparas/LigandMPNN` is installed; only the `ligandmpnn_env` conda environment
exists on this machine today, without the source checkout.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_adapter_prodigy.py
import pytest

from protein_design_mcp.adapters.prodigy import build_args, parse_output
from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.loader import load_manifests
from protein_design_mcp.validation import ToolInputError, validate_and_fill
from pathlib import Path

MANIFEST_DIR = Path(__file__).resolve().parents[1] / "manifests"

SAMPLE_STDOUT = """\
[+] Reading structure file: /tmp/complex.pdb
[+] Parsed structure file complex (2 chains, 210 residues)
[+] No. of intermolecular contacts: 72
[+] Predicted binding affinity (kcal.mol-1): -11.30
[+] Predicted dissociation constant (M) at 25.0C: 5.2e-09
"""


def _manifest():
    return next(m for m in load_manifests(MANIFEST_DIR) if m.name == "run_prodigy")


def test_manifest_loads_and_is_not_composite():
    m = _manifest()
    assert m.category == "scoring"
    assert m.composite is False
    assert m.requires.gpu is False


def test_manifest_documents_what_it_needs():
    doc = _manifest().doc
    assert "## What this is" in doc
    assert "## What you must supply" in doc


def test_build_args_maps_chains_to_two_flags():
    args = build_args({"complex_pdb": "/tmp/c.pdb", "chain_a": "A", "chain_b": "B",
                       "temperature": 25.0})
    assert "/tmp/c.pdb" in args
    assert "--selection" in args
    assert "A" in args and "B" in args


def test_build_args_includes_temperature():
    args = build_args({"complex_pdb": "/tmp/c.pdb", "chain_a": "A", "chain_b": "B",
                       "temperature": 37.0})
    assert "--temperature" in args
    assert "37.0" in args


def test_parse_output_extracts_affinity_and_kd():
    result = parse_output(
        CompletedRun(returncode=0, stdout=SAMPLE_STDOUT, stderr="", workdir=Path("/tmp"))
    )
    assert result["binding_affinity_kcal_per_mol"] == pytest.approx(-11.30)
    assert result["dissociation_constant_M"] == pytest.approx(5.2e-09)
    assert result["intermolecular_contacts"] == 72


def test_parse_output_carries_the_calibration_warning():
    result = parse_output(
        CompletedRun(returncode=0, stdout=SAMPLE_STDOUT, stderr="", workdir=Path("/tmp"))
    )
    assert "de novo" in result["caveat"]


def test_parse_output_raises_when_affinity_is_absent():
    with pytest.raises(ValueError, match="affinity"):
        parse_output(
            CompletedRun(returncode=0, stdout="nothing useful", stderr="",
                         workdir=Path("/tmp"))
        )


def test_validation_rejects_a_non_structure_path():
    with pytest.raises(ToolInputError, match="complex_pdb"):
        validate_and_fill(_manifest(), {"complex_pdb": "notes.txt",
                                        "chain_a": "A", "chain_b": "B"})


def test_validation_rejects_a_multi_character_chain_id():
    with pytest.raises(ToolInputError, match="chain_a"):
        validate_and_fill(_manifest(), {"complex_pdb": "c.pdb",
                                        "chain_a": "AB", "chain_b": "B"})


def test_validation_fills_the_default_temperature():
    params = validate_and_fill(_manifest(), {"complex_pdb": "c.pdb",
                                             "chain_a": "A", "chain_b": "B"})
    assert params["temperature"] == 25.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_adapter_prodigy.py -v`
Expected: FAIL — `manifests/` does not exist and
`protein_design_mcp.adapters.prodigy` is not importable

- [ ] **Step 3: Write minimal implementation**

```yaml
# manifests/run_prodigy.yaml
name: run_prodigy
category: scoring
composite: false

engine:
  repo: prodigy
  env: scoring
  entry: ["prodigy"]

requires:
  gpu: false

summary: >-
  Estimate the binding free energy of an existing protein-protein complex from
  its interfacial contacts (PRODIGY). Runs on CPU in milliseconds and needs no
  model weights, so it is the cheapest first look at an interface. It scores a
  complex you already have; it does not predict structure and does not design.
  Fold a candidate with run_chai1 or run_esmfold2 first.

doc: |
  ## What this is
  PRODIGY predicts the binding affinity of a protein-protein complex with a
  linear regression over the number and type of interfacial residue contacts.
  It returns a binding free energy in kcal/mol and a dissociation constant.

  ## What it is for
  A fast, interpretable, absolute-scale sanity floor on an interface. Because
  it is CPU-only and takes milliseconds, it is cheap enough to run on every
  candidate before spending GPU time on anything else.

  ## When to use this instead of the alternatives
  - `run_rosetta_interface` gives a physics-based decomposition (dG_separated,
    buried surface area, shape complementarity, hydrogen bond counts) and is
    the better choice when you need to know *why* an interface scores as it
    does. It is slower and depends on a license-gated PyRosetta install.
  - `run_ipsae` and the ipTM fields from a co-folding tool measure model
    *confidence* in the interface, not its energy. Those discriminate binders
    from non-binders better than PRODIGY does. Prefer them for ranking designs.
  - Use PRODIGY when you want an absolute number on a physical scale rather
    than a model-internal confidence score.

  ## Important caveat
  PRODIGY is calibrated on natural protein complexes from the affinity
  benchmark. It systematically mis-ranks de novo designed binders. Treat its
  output as a sanity floor, never as the ranking criterion for a design
  campaign.

  ## What you must supply
  A PDB or mmCIF file containing both partners, and the chain identifier of
  each partner.

  ## What you get back
  `binding_affinity_kcal_per_mol`, `dissociation_constant_M`,
  `intermolecular_contacts`, and a `caveat` string restating the calibration
  limitation.

schema:
  complex_pdb:
    type: string
    pattern: '\.(pdb|cif|ent)$'
    required: true
    description: Path to a structure file containing both partners.
    example: complex.pdb
  chain_a:
    type: string
    pattern: '^[A-Za-z0-9]$'
    required: true
    description: Chain identifier of the first partner, a single character.
    example: A
  chain_b:
    type: string
    pattern: '^[A-Za-z0-9]$'
    required: true
    description: Chain identifier of the second partner, a single character.
    example: B
  temperature:
    type: number
    minimum: 0.0
    maximum: 100.0
    default: 25.0
    description: Temperature in Celsius used for the Kd conversion.
    example: 25.0
```

```python
# src/protein_design_mcp/adapters/__init__.py
"""Per-engine argument translation and output parsing."""
```

```python
# src/protein_design_mcp/adapters/prodigy.py
"""Adapter for PRODIGY (prodigy-prot).

PRODIGY writes its results to stdout as ``[+] key: value`` lines rather than a
machine-readable file, so the adapter parses stdout.
"""

from __future__ import annotations

import re
from typing import Any

from protein_design_mcp.dispatch.env import CompletedRun

CAVEAT = (
    "PRODIGY is calibrated on natural complexes and systematically mis-ranks "
    "de novo designed binders. Use it as a sanity floor, not as a ranking "
    "criterion."
)

_AFFINITY_RE = re.compile(r"binding affinity \(kcal\.mol-1\):\s*(-?[\d.]+)", re.I)
_KD_RE = re.compile(r"dissociation constant \(M\)[^:]*:\s*([\d.eE+-]+)", re.I)
_CONTACTS_RE = re.compile(r"intermolecular contacts:\s*(\d+)", re.I)


def build_args(params: dict[str, Any]) -> list[str]:
    """Translate validated parameters into PRODIGY's argv."""
    return [
        str(params["complex_pdb"]),
        "--selection",
        str(params["chain_a"]),
        str(params["chain_b"]),
        "--temperature",
        str(params["temperature"]),
    ]


def parse_output(run: CompletedRun) -> dict[str, Any]:
    """Extract PRODIGY's numbers from stdout."""
    affinity = _AFFINITY_RE.search(run.stdout)
    if affinity is None:
        raise ValueError(
            "PRODIGY produced no binding affinity line. Output was:\n"
            f"{run.stdout.strip()[-1000:]}"
        )

    kd = _KD_RE.search(run.stdout)
    contacts = _CONTACTS_RE.search(run.stdout)
    return {
        "binding_affinity_kcal_per_mol": float(affinity.group(1)),
        "dissociation_constant_M": float(kd.group(1)) if kd else None,
        "intermolecular_contacts": int(contacts.group(1)) if contacts else None,
        "caveat": CAVEAT,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_adapter_prodigy.py -v`
Expected: PASS, 10 tests

- [ ] **Step 5: Commit**

```bash
git add manifests/run_prodigy.yaml src/protein_design_mcp/adapters/ tests/test_adapter_prodigy.py
git commit -m "feat: add run_prodigy manifest and adapter"
```

---

### Task 9: Wire the registry into server.py

**Files:**
- Modify: `src/protein_design_mcp/server.py` (delete `TOOLS` at lines 53-630, `GPU_ONLY_TOOLS`/`COMPOSITE_TOOL_NAMES` at 41-44, `list_tools` at 633-639, and the `call_tool` if/elif chain at 641 onward)
- Create: `src/protein_design_mcp/app.py`
- Test: `tests/test_server_wiring.py`

**Interfaces:**
- Consumes: everything from Tasks 1-8
- Produces: `build_registry() -> ToolRegistry`, `ServerApp` with
  `async .list_tools() -> list[Tool]` and
  `async .call_tool(name, arguments) -> list[TextContent]`

The logic moves into `app.py` so it is testable without an MCP transport;
`server.py` keeps only the SDK handler registration and the entry point.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_server_wiring.py
import json
from pathlib import Path

import pytest

from protein_design_mcp.app import ServerApp
from protein_design_mcp.manifest.registry import ToolRegistry
from protein_design_mcp.manifest.schema import parse_manifest

MANIFEST_DIR = Path(__file__).resolve().parents[1] / "manifests"


def _composite():
    return parse_manifest(
        {
            "name": "run_boltzgen_run",
            "category": "generation",
            "composite": True,
            "engine": {"repo": "boltzgen", "env": "boltzgen", "entry": ["boltzgen", "run"]},
            "summary": "Full pipeline.",
            "doc": "## What this is\nFull pipeline.\n",
            "schema": {"spec": {"type": "string", "required": True, "example": "s.yaml"}},
        }
    )


@pytest.mark.asyncio
async def test_list_tools_includes_describe_tool():
    app = ServerApp(ToolRegistry([]))
    assert "describe_tool" in {t.name for t in await app.list_tools()}


@pytest.mark.asyncio
async def test_composite_tool_is_absent_from_the_listing():
    app = ServerApp(ToolRegistry([_composite()]))
    assert "run_boltzgen_run" not in {t.name for t in await app.list_tools()}


@pytest.mark.asyncio
async def test_calling_a_composite_tool_by_name_is_refused_with_a_reason():
    app = ServerApp(ToolRegistry([_composite()]))
    payload = json.loads((await app.call_tool("run_boltzgen_run", {}))[0].text)
    assert "composite" in payload["error"]


@pytest.mark.asyncio
async def test_calling_an_unknown_tool_reports_it():
    app = ServerApp(ToolRegistry([]))
    payload = json.loads((await app.call_tool("run_nope", {}))[0].text)
    assert "unknown" in payload["error"]


@pytest.mark.asyncio
async def test_describe_tool_is_dispatched_without_a_subprocess():
    registry = ToolRegistry([])
    app = ServerApp(registry)
    payload = json.loads(
        (await app.call_tool("describe_tool", {"name": "describe_tool"}))[0].text
    )
    assert payload["name"] == "describe_tool"


@pytest.mark.asyncio
async def test_invalid_input_returns_a_correctable_error_not_a_crash():
    registry = ToolRegistry(
        [m for m in _load_real() if m.name == "run_prodigy"]
    )
    app = ServerApp(registry)
    payload = json.loads(
        (await app.call_tool("run_prodigy", {"complex_pdb": "notes.txt",
                                             "chain_a": "A", "chain_b": "B"}))[0].text
    )
    assert "complex_pdb" in payload["error"]
    assert "complex.pdb" in payload["error"]


def _load_real():
    from protein_design_mcp.manifest.loader import load_manifests

    return load_manifests(MANIFEST_DIR)


def test_real_manifests_all_load():
    assert {m.name for m in _load_real()} >= {"run_prodigy"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_server_wiring.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'protein_design_mcp.app'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/protein_design_mcp/app.py
"""Transport-independent MCP application logic.

Keeping this out of ``server.py`` means the listing and dispatch behaviour can
be tested without standing up a transport.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from mcp.types import TextContent, Tool

from protein_design_mcp.adapters import prodigy
from protein_design_mcp.dispatch.env import EngineError, EnvDispatcher
from protein_design_mcp.dispatch.serialize import to_jsonable
from protein_design_mcp.manifest.loader import load_manifests
from protein_design_mcp.manifest.registry import ToolNotAvailable, ToolRegistry
from protein_design_mcp.meta_tools import DESCRIBE_TOOL_MANIFEST, describe_tool
from protein_design_mcp.validation import ToolInputError, validate_and_fill

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = float(os.environ.get("PROTEIN_MCP_TIMEOUT", "3600"))

# Engine name -> (build_args, parse_output). Populated per adapter.
ADAPTERS = {
    "prodigy": (prodigy.build_args, prodigy.parse_output),
}


def manifest_dir() -> Path:
    override = os.environ.get("PROTEIN_MCP_MANIFEST_DIR")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2] / "manifests"


def build_registry(device: str = "cuda") -> ToolRegistry:
    """Load manifests from disk into a registry."""
    return ToolRegistry(load_manifests(manifest_dir()), device=device)


def _error(message: str) -> list[TextContent]:
    return [TextContent(type="text", text=json.dumps({"error": message}, indent=2))]


def _ok(payload: Any) -> list[TextContent]:
    return [
        TextContent(type="text", text=json.dumps(to_jsonable(payload), indent=2))
    ]


class ServerApp:
    """Owns the registry and turns MCP calls into engine runs."""

    def __init__(
        self,
        registry: ToolRegistry,
        dispatcher: EnvDispatcher | None = None,
    ) -> None:
        self._registry = registry
        self._dispatcher = dispatcher or EnvDispatcher()

    async def list_tools(self) -> list[Tool]:
        tools = self._registry.tools()
        tools.append(
            Tool(
                name=DESCRIBE_TOOL_MANIFEST.name,
                description=DESCRIBE_TOOL_MANIFEST.summary,
                inputSchema={
                    "type": "object",
                    "properties": {
                        key: {
                            k: v
                            for k, v in spec.items()
                            if k not in ("required", "example")
                        }
                        for key, spec in DESCRIBE_TOOL_MANIFEST.schema.items()
                    },
                    "required": [],
                    "additionalProperties": False,
                },
            )
        )
        return tools

    async def call_tool(
        self, name: str, arguments: dict[str, Any] | None
    ) -> list[TextContent]:
        arguments = arguments or {}
        logger.info("tool call: %s %s", name, arguments)

        if name == DESCRIBE_TOOL_MANIFEST.name:
            return _ok(
                describe_tool(
                    self._registry,
                    name=arguments.get("name"),
                    category=arguments.get("category"),
                )
            )

        try:
            manifest = self._registry.resolve(name)
        except ToolNotAvailable as exc:
            return _error(str(exc))

        try:
            params = validate_and_fill(manifest, arguments)
        except ToolInputError as exc:
            return _error(str(exc))

        adapter = ADAPTERS.get(manifest.engine.repo)
        if adapter is None:
            return _error(
                f"{name} has no adapter registered for engine "
                f"{manifest.engine.repo!r}"
            )

        build_args, parse_output = adapter
        try:
            run = await self._dispatcher.run(
                manifest.engine, build_args(params), timeout=DEFAULT_TIMEOUT_S
            )
            return _ok(parse_output(run))
        except EngineError as exc:
            return _error(str(exc))
        except ValueError as exc:
            return _error(f"{name}: could not parse engine output: {exc}")
```

Then edit `server.py`. Delete lines 40-44 (`GPU_ONLY_TOOLS`,
`COMPOSITE_TOOL_NAMES`), the entire `TOOLS = [...]` literal, the existing
`list_tools`, and the entire `call_tool` if/elif chain and its handler
functions. Replace the handler registration with:

```python
from protein_design_mcp.app import ServerApp, build_registry

_app = ServerApp(build_registry(device=DEVICE))


@server.list_tools()
async def list_tools() -> list[Tool]:
    return await _app.list_tools()


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    return await _app.call_tool(name, arguments)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_server_wiring.py -v`
Expected: PASS, 7 tests

- [ ] **Step 5: Commit**

```bash
git add src/protein_design_mcp/app.py src/protein_design_mcp/server.py tests/test_server_wiring.py
git commit -m "refactor: drive list_tools and call_tool from the manifest registry"
```

---

### Task 10: Transport flag, SDK pin, and removal of superseded tests

**Files:**
- Modify: `src/protein_design_mcp/server.py` (the `run_server`/`main` block at the end)
- Modify: `pyproject.toml:38`
- Delete: `tests/test_design_binder.py`, `tests/test_validate_design.py`, `tests/test_optimize.py`, `tests/test_hotspots.py`, `tests/test_tools.py`
- Test: `tests/test_transport.py`

**Interfaces:**
- Consumes: `ServerApp` from Task 9
- Produces: `parse_args(argv) -> argparse.Namespace`, `async run_server(transport, host, port)`

Spec §6: every tool needs a GPU, so the natural deployment is the server on the
L40S host with clients elsewhere. stdio cannot do that without an SSH tunnel.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_transport.py
import pytest

from protein_design_mcp.server import parse_args


def test_stdio_is_the_default():
    assert parse_args([]).transport == "stdio"


def test_http_transport_can_be_selected():
    assert parse_args(["--transport", "http"]).transport == "http"


def test_http_has_a_default_host_and_port():
    args = parse_args(["--transport", "http"])
    assert args.host == "127.0.0.1"
    assert args.port == 8765


def test_host_and_port_are_overridable():
    args = parse_args(["--transport", "http", "--host", "0.0.0.0", "--port", "9000"])
    assert args.host == "0.0.0.0"
    assert args.port == 9000


def test_unknown_transport_is_rejected():
    with pytest.raises(SystemExit):
        parse_args(["--transport", "carrier-pigeon"])


def test_mcp_is_pinned_below_v2():
    from pathlib import Path

    text = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    assert "mcp>=1.25,<2" in text
    assert "mcp>=0.1.0" not in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_transport.py -v`
Expected: FAIL with `ImportError: cannot import name 'parse_args'`

- [ ] **Step 3: Write minimal implementation**

Replace the `run_server`/`main` block at the end of `server.py`:

```python
import argparse


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments. Pure — safe to assert on in tests."""
    parser = argparse.ArgumentParser(prog="protein-design-mcp")
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help=(
            "stdio for a local client; http to serve over streamable HTTP so "
            "clients on other machines can reach the GPU host."
        ),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args(argv)


async def run_server(transport: str = "stdio", host: str = "127.0.0.1", port: int = 8765):
    """Run the MCP server over the chosen transport."""
    if transport == "http":
        from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
        import uvicorn

        manager = StreamableHTTPSessionManager(app=server)
        config = uvicorn.Config(
            manager.handle_request, host=host, port=port, log_level="info"
        )
        await uvicorn.Server(config).serve()
        return

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream, server.create_initialization_options()
        )


def main():
    args = parse_args()
    asyncio.run(run_server(args.transport, args.host, args.port))


if __name__ == "__main__":
    main()
```

In `pyproject.toml`, change `"mcp>=0.1.0",` to `"mcp>=1.25,<2",` and add
`"uvicorn>=0.30",` to `dependencies`.

Delete the superseded test files:

```bash
git rm tests/test_design_binder.py tests/test_validate_design.py \
       tests/test_optimize.py tests/test_hotspots.py tests/test_tools.py
```

- [ ] **Step 4: Run the full suite**

Run: `pytest tests/ -v`
Expected: PASS. Tests for retained pipelines (`test_proteinmpnn.py`,
`test_rfdiffusion.py`, `test_esmfold.py`, `test_alphafold2.py`,
`test_pdb_utils.py`, `test_sasa.py`, `test_conservation.py`,
`test_job_queue.py`, `test_uniprot.py`, `test_pubmed.py`, `test_analyze.py`)
must still pass. If `test_server_handlers.py` references deleted handlers,
delete it too and note it in the commit message.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add http transport, pin mcp SDK, drop superseded tool tests"
```

---

### Task 11: Generated tool documentation

**Files:**
- Create: `scripts/generate_tool_docs.py`
- Create: `docs/tools/` (generated, committed)
- Test: `tests/test_doc_generation.py`

**Interfaces:**
- Consumes: `load_manifests` from Task 2
- Produces: `render_doc(manifest) -> str`, `main(manifest_dir, out_dir) -> list[Path]`

Spec §4.2: `docs/tools/<name>.md` is the fourth artifact derived from the
manifest. Generating it rather than hand-writing it is what prevents drift.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_doc_generation.py
from pathlib import Path

from protein_design_mcp.manifest.loader import load_manifests

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from generate_tool_docs import main, render_doc  # noqa: E402

MANIFEST_DIR = Path(__file__).resolve().parents[1] / "manifests"


def _prodigy():
    return next(m for m in load_manifests(MANIFEST_DIR) if m.name == "run_prodigy")


def test_rendered_doc_starts_with_the_tool_name():
    assert render_doc(_prodigy()).startswith("# run_prodigy")


def test_rendered_doc_contains_the_body():
    assert "## What this is" in render_doc(_prodigy())


def test_rendered_doc_has_a_parameter_table_with_constraints():
    rendered = render_doc(_prodigy())
    assert "| Parameter |" in rendered
    assert "complex_pdb" in rendered
    assert "25.0" in rendered  # the temperature default


def test_rendered_doc_names_the_engine_and_environment():
    rendered = render_doc(_prodigy())
    assert "prodigy" in rendered
    assert "scoring" in rendered


def test_main_writes_one_file_per_manifest(tmp_path):
    written = main(MANIFEST_DIR, tmp_path)
    assert (tmp_path / "run_prodigy.md").exists()
    assert len(written) == len(load_manifests(MANIFEST_DIR))


def test_generated_docs_are_current(tmp_path):
    """Fails when a manifest changed but docs/tools/ was not regenerated."""
    main(MANIFEST_DIR, tmp_path)
    committed = Path(__file__).resolve().parents[1] / "docs" / "tools"
    for generated in sorted(tmp_path.glob("*.md")):
        assert (committed / generated.name).read_text() == generated.read_text()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_doc_generation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'generate_tool_docs'`

- [ ] **Step 3: Write minimal implementation**

```python
#!/usr/bin/env python3
# scripts/generate_tool_docs.py
"""Render docs/tools/<name>.md from the manifests.

The manifest is the single source of truth; this script is how the
human-readable copy stays in step with the MCP schema and describe_tool.
Run it whenever a manifest changes: tests/test_doc_generation.py fails if the
committed output is stale.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from protein_design_mcp.manifest.loader import load_manifests  # noqa: E402
from protein_design_mcp.manifest.schema import Manifest  # noqa: E402

_CONSTRAINT_KEYS = (
    "pattern",
    "enum",
    "minimum",
    "maximum",
    "minItems",
    "maxItems",
)


def _constraints(spec: dict) -> str:
    parts = [f"{key}: `{spec[key]}`" for key in _CONSTRAINT_KEYS if key in spec]
    return "<br>".join(parts) if parts else "—"


def render_doc(manifest: Manifest) -> str:
    lines = [
        f"# {manifest.name}",
        "",
        f"**Category:** {manifest.category}  ",
        f"**Engine:** `{manifest.engine.repo}`  ",
        f"**Environment:** `{manifest.engine.env}`  ",
        f"**GPU required:** {'yes' if manifest.requires.gpu else 'no'}",
        "",
        "> This file is generated from "
        f"`manifests/{manifest.name}.yaml`. Edit the manifest, then run "
        "`python scripts/generate_tool_docs.py`.",
        "",
        "## Summary",
        "",
        manifest.summary,
        "",
        manifest.doc.rstrip(),
        "",
        "## Parameters",
        "",
        "| Parameter | Type | Required | Default | Constraints | Description |",
        "|---|---|---|---|---|---|",
    ]
    for key, spec in manifest.schema.items():
        default = spec.get("default", "—")
        lines.append(
            f"| `{key}` | {spec.get('type', '—')} | "
            f"{'yes' if spec.get('required') else 'no'} | "
            f"`{default}` | {_constraints(spec)} | "
            f"{spec.get('description', '')} |"
        )
    lines.append("")
    return "\n".join(lines)


def main(manifest_dir: Path, out_dir: Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for manifest in load_manifests(Path(manifest_dir)):
        path = out_dir / f"{manifest.name}.md"
        path.write_text(render_doc(manifest))
        written.append(path)
    return written


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    for path in main(root / "manifests", root / "docs" / "tools"):
        print(f"wrote {path}")
```

Then generate and commit the output:

```bash
python scripts/generate_tool_docs.py
```

- [ ] **Step 4: Run the full suite**

Run: `pytest tests/ -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/generate_tool_docs.py docs/tools/ tests/test_doc_generation.py
git commit -m "feat: generate tool docs from manifests"
```

---

## Self-Review

**Spec coverage.**

| Spec section | Task |
|---|---|
| §4.1-4.2 manifest as single source of truth | 1, 2, 3, 11 |
| §4.2 composite unregistrable *and* uncallable | 3, 9 |
| §4.3 three documentation layers | 3 (layer 1), 7 (layer 2), 11 (repo copy) |
| §4.3 sibling comparison requirement | 2 (enforced), 8 (exemplified) |
| §5.2 dispatch contract, scratch dir, numpy JSON | 5, 6 |
| §5.3 OOM handling | 6 |
| §6 transport | 10 |
| §6 server-side validation and defaults | 4 |
| §6 SDK pin | 10 |
| §8 migration, superseded tests removed | 10 |

**Gaps carried to plan 2, deliberately:** MCP Resource layer (§4.3 layer 3),
`PROFILE` env var (§6), `requires.license_gated`/`weights` *detection* (Task 3
implements the filter; discovering what is actually installed belongs with the
environment build), `max_residues` enforcement (parsed in Task 1, enforced per
engine), `get_job_status`, and the 28 remaining engines. Plan 2 also installs
`dauparas/LigandMPNN` and adds `run_mpnn`.

**Placeholder scan.** No TBDs. Every code step carries runnable code; every
test step carries real assertions.

**Type consistency.** `Manifest`/`EngineSpec`/`Requirements` (Task 1) are used
unchanged in Tasks 2, 3, 4, 6, 7, 11. `CompletedRun` (Task 6) is the input to
`parse_output` (Task 8) and is constructed directly in Task 8's tests.
`ToolNotAvailable` (Task 3) is caught in Tasks 7 and 9. `ToolInputError`
(Task 4) is caught in Task 9. `EngineError` (Task 6) is caught in Task 9.
`to_jsonable` (Task 5) is used in Task 9's `_ok`. `SIBLING_DOC_HEADING`
(Task 2) is asserted in Task 7's fixtures and satisfied by Task 8's manifest.

**One deviation from the spec, recorded:** spec §9 step 3 names `run_mpnn` as
the engine that proves the dispatcher. Task 8 uses `run_prodigy` instead,
because PRODIGY is CPU-only and weightless so its test runs in CI, whereas
LigandMPNN is not checked out on this machine — only the `ligandmpnn_env`
conda environment exists. The contract proven is identical.
