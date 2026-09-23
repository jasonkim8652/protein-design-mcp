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


def json_schema_for(manifest: Manifest) -> dict[str, Any]:
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
        load_failures: dict[str, str] | None = None,
    ) -> None:
        """``load_failures`` seeds exclusion reasons for names that never
        made it into ``manifests`` at all — e.g. a tool name claimed by two
        manifest files, both excluded before construction (see
        ``manifest.loader.load_manifests_resilient``). Without this,
        ``resolve()`` would tell the model the tool was simply unknown
        rather than why it is unavailable.
        """
        self._all: dict[str, Manifest] = {m.name: m for m in manifests}
        self._device = device
        self._available_weights = available_weights
        self._licensed = licensed
        self._reasons: dict[str, str] = dict(load_failures or {})
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
                inputSchema=json_schema_for(m),
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
