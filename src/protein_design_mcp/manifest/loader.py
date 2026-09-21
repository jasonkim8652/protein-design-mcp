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
