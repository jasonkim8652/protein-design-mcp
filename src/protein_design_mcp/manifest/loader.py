"""Discover and load tool manifests from a directory."""

from __future__ import annotations

import collections
import re
from pathlib import Path

import yaml

from protein_design_mcp.manifest.schema import Manifest, ManifestError, parse_manifest

SIBLING_DOC_HEADING = "## When to use this instead of the alternatives"

_TOOL_MENTION_RE = re.compile(r"\brun_[a-z0-9_]+\b")
_UNAVAILABLE_MARKER = "not yet implemented"
_FENCED_CODE_BLOCK_RE = re.compile(r"^```", re.MULTILINE)


def _extract_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs, skipping fenced code blocks.

    A paragraph is a block of non-empty lines separated by blank lines.
    Fenced code blocks (``` delimited) are treated as atomic units.
    """
    paragraphs = []
    current_paragraph = []
    in_code_block = False

    for line in text.splitlines():
        # Track code block state
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            # Include code block lines in the current paragraph if building one
            if current_paragraph or in_code_block:
                current_paragraph.append(line)
            continue

        # In a code block: accumulate but don't end paragraph
        if in_code_block:
            current_paragraph.append(line)
            continue

        # Outside code block: blank line ends paragraph
        if not line.strip():
            if current_paragraph:
                paragraphs.append("\n".join(current_paragraph))
                current_paragraph = []
            continue

        # Outside code block, non-blank line: accumulate
        current_paragraph.append(line)

    # Don't forget the last paragraph
    if current_paragraph:
        paragraphs.append("\n".join(current_paragraph))

    return paragraphs


def _check_doc_references(manifests: list[Manifest]) -> None:
    """Every tool a summary/doc names must exist, or be marked not yet implemented.

    Checks both summary (visible to all MCP clients) and doc (reachable via
    describe_tool). Scans per-paragraph to survive rewrapping. Skips fenced
    code blocks where hypothetical tool names are safe.

    Without this, a doc that says "use run_x instead" keeps saying it after
    run_x ships under a different name, or before it ships at all — and the
    model acts on it either way.
    """
    known = {m.name for m in manifests}

    for manifest in manifests:
        # Check both summary (more visible) and doc (less visible but still exposed)
        for text_kind, text in [("summary", manifest.summary), ("doc", manifest.doc)]:
            paragraphs = _extract_paragraphs(text)
            for paragraph in paragraphs:
                # Skip paragraphs that are entirely inside fenced code blocks
                if paragraph.strip().startswith("```"):
                    continue

                for mentioned in _TOOL_MENTION_RE.findall(paragraph):
                    if mentioned in known or mentioned == manifest.name:
                        continue
                    if _UNAVAILABLE_MARKER in paragraph.lower():
                        continue
                    raise ManifestError(
                        f"{manifest.name}: {text_kind} names {mentioned!r}, which is not a "
                        "known tool. Either fix the name, or mark it "
                        f"'({_UNAVAILABLE_MARKER})' in the same paragraph."
                    )


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
    _check_doc_references(manifests)
    return sorted(manifests, key=lambda m: m.name)
