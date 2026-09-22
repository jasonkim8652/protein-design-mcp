"""Discover and load tool manifests from a directory."""

from __future__ import annotations

import collections
import os
import re
from pathlib import Path
from typing import NamedTuple

import yaml

from protein_design_mcp.manifest.schema import (
    TOOL_NAME_RE,
    Manifest,
    ManifestError,
    parse_manifest,
)

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


def _has_marker_for_mention(paragraph: str, mentioned: str) -> bool:
    """Check if the marker appears near a specific tool mention in the text.

    Looks for the marker directly following the mention (with optional backticks).
    E.g., `run_tool` (not yet implemented) or run_tool (not yet implemented).
    """
    # Pattern: optional backtick, mention, optional backtick, whitespace, marker in parens
    pattern = rf"[`]?{re.escape(mentioned)}[`]?\s*\([^)]*{re.escape(_UNAVAILABLE_MARKER)}[^)]*\)"
    return bool(re.search(pattern, paragraph, re.IGNORECASE))


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
                    if _has_marker_for_mention(paragraph, mentioned):
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


def _probe_name(path: Path) -> str | None:
    """Best-effort tool name for a manifest whose full load failed, used
    only to key its exclusion reason so ``resolve(name)`` can find it.

    Returns ``None`` when the file isn't valid YAML, isn't a mapping, or
    has no ``name`` that even looks like a real tool name — callers must
    fall back to the file's own name (or stem) in that case. That fallback
    is a naming CONVENTION this codebase happens to follow (every real
    manifest today is named after its tool), never a guarantee: nothing
    enforces filename == tool name, so a caller relying on it must be told
    honestly, not as if the tool is confirmed to exist.
    """
    try:
        data = yaml.safe_load(path.read_text())
    except yaml.YAMLError:
        return None
    if not isinstance(data, dict):
        return None
    name = data.get("name")
    if isinstance(name, str) and TOOL_NAME_RE.match(name):
        return name
    return None


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
    """Load every ``*.yaml`` manifest in ``directory``, sorted by tool name.

    All-or-nothing: one bad manifest raises ``ManifestError`` and aborts the
    whole load. This is what CI wants and is what every existing caller of
    this function already expects — it is deliberately left unchanged.
    ``load_manifests_resilient`` is the entry point that excludes failures
    individually instead of aborting; use that for anything that must keep
    serving the tools that DID load (i.e. the running server).
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise ManifestError(f"manifest directory not found: {directory}")

    manifests = [_load_one(p) for p in sorted(directory.glob("*.yaml"))]
    _check_unique(manifests)
    _check_sibling_docs(manifests)
    _check_doc_references(manifests)
    return sorted(manifests, key=lambda m: m.name)


class ManifestLoadResult(NamedTuple):
    """What ``load_manifests_resilient`` found.

    ``manifests`` is every manifest that is safe to serve. ``reasons`` maps
    an identifier to a model-facing explanation of why something else was
    excluded, keyed by whatever a caller would plausibly ``resolve()``:

    - the tool NAME whenever it is known — duplicate-name and
      bad-cross-reference exclusions always have a fully parsed, named
      Manifest behind them; a per-file failure (schema violation, etc.)
      that happened AFTER ``name:`` itself parsed is keyed by that name too.
    - the file's STEM (``run_mpnn.yaml`` -> ``run_mpnn``) when no name ever
      parsed at all (unreadable file, invalid YAML, not even a mapping).
      This is a naming convention this codebase's manifests happen to
      follow, never a guarantee, so that reason says the FILE failed to
      load rather than asserting a tool by that name exists.
    - the raw file name may ALSO be present (for the startup exclusion
      table) but is never the only key — nothing is ever resolvable only by
      a ``*.yaml``-suffixed identifier, since no tool name ever has one.
    """

    manifests: list[Manifest]
    reasons: dict[str, str]


def _duplicate_exclusions(pairs: list[tuple[Path, Manifest]]) -> dict[str, str]:
    """Every manifest name claimed by more than one file is excluded — all
    of them, not a "first file wins" pick. An arbitrary winner would be
    worse than an honest exclusion: it would silently change which code
    runs depending on directory listing order.
    """
    by_name: dict[str, list[Path]] = collections.defaultdict(list)
    for path, manifest in pairs:
        by_name[manifest.name].append(path)

    reasons: dict[str, str] = {}
    for name, paths in by_name.items():
        if len(paths) < 2:
            continue
        files = ", ".join(sorted(p.name for p in paths))
        reasons[name] = (
            f"{name} is unavailable: it is defined by {len(paths)} manifest "
            f"files at once ({files}), so none of the definitions can be "
            "trusted to be the intended one."
        )
    return reasons


def _sibling_doc_exclusions(manifests: list[Manifest]) -> dict[str, str]:
    """Same rule as ``_check_sibling_docs``, but excludes only the manifest
    missing the heading rather than aborting the whole category.
    """
    by_category: dict[str, list[Manifest]] = collections.defaultdict(list)
    for m in manifests:
        by_category[m.category].append(m)

    reasons: dict[str, str] = {}
    for category, members in by_category.items():
        if len(members) < 2:
            continue
        for m in members:
            if SIBLING_DOC_HEADING not in m.doc:
                reasons[m.name] = (
                    f"{m.name} is unavailable: it shares category {category!r} "
                    f"with {len(members) - 1} other tool(s), but its "
                    "documentation does not explain when to prefer it over "
                    "them, so which one to call cannot be determined "
                    "confidently."
                )
    return reasons


def _doc_reference_exclusions(manifests: list[Manifest]) -> dict[str, str]:
    """Same rule as ``_check_doc_references``, but excludes only the
    manifest carrying the bad reference, not the tool it names.
    """
    known = {m.name for m in manifests}
    reasons: dict[str, str] = {}

    for manifest in manifests:
        for text_kind, text in [("summary", manifest.summary), ("doc", manifest.doc)]:
            if manifest.name in reasons:
                break
            for paragraph in _extract_paragraphs(text):
                if paragraph.strip().startswith("```"):
                    continue
                for mentioned in _TOOL_MENTION_RE.findall(paragraph):
                    if mentioned in known or mentioned == manifest.name:
                        continue
                    if _has_marker_for_mention(paragraph, mentioned):
                        continue
                    reasons[manifest.name] = (
                        f"{manifest.name} is unavailable: its {text_kind} "
                        f"refers to {mentioned!r}, which is not a known "
                        "tool, so following that reference would call "
                        "something that doesn't exist."
                    )
                    break
                if manifest.name in reasons:
                    break
    return reasons


def load_manifests_resilient(directory: Path) -> ManifestLoadResult:
    """Load every ``*.yaml`` manifest in ``directory``, excluding failures
    individually rather than aborting the whole load.

    A manifest directory is meant to grow to ~25+ entries (the GPU engine
    plan). One malformed file, one name collision, or one stale
    cross-reference must not take every other, perfectly valid, manifest
    down with it — see the module docstring's defect writeup. Each failure
    is excluded on its own, with a reason carried out in the result instead
    of raised, so ``build_registry`` can keep serving everything that DID
    load and can tell the model *why* the rest didn't.

    ``STRICT_MANIFESTS`` is read HERE, at call time (not import time), so a
    test can set and unset it around a single call. When it is exactly
    ``"1"`` this delegates to ``load_manifests`` and lets its
    ``ManifestError`` propagate unchanged — the same all-or-nothing
    behaviour CI wants, byte-identical to today's error messages.
    """
    if os.environ.get("STRICT_MANIFESTS") == "1":
        return ManifestLoadResult(load_manifests(directory), {})

    directory = Path(directory)
    if not directory.is_dir():
        raise ManifestError(f"manifest directory not found: {directory}")

    reasons: dict[str, str] = {}
    pairs: list[tuple[Path, Manifest]] = []
    for path in sorted(directory.glob("*.yaml")):
        try:
            pairs.append((path, _load_one(path)))
        except ManifestError as exc:
            message = str(exc)
            probed_name = _probe_name(path)
            if probed_name is not None:
                # The name parsed before something later failed (bad
                # category, empty summary, etc.) — key by it so a caller
                # asking for THAT tool gets told why, not "unknown tool".
                reasons[probed_name] = (
                    f"{probed_name} is unavailable: its manifest file "
                    f"({path.name}) failed to load: {message}"
                )
            else:
                # No name ever parsed (unreadable file, invalid YAML, not
                # even a mapping). The filename stem is the only plausible
                # identifier a caller might use — but it's a convention,
                # not a fact, so say so rather than asserting the tool
                # exists.
                reasons[path.stem] = (
                    f"a manifest file ({path.name}) failed to load, so it "
                    "defines no tool right now. If a caller expected a "
                    f"tool named {path.stem!r} here — going only by the "
                    "filename, which this codebase does not guarantee "
                    f"matches the tool's actual name — that name is "
                    f"unavailable until the file is fixed: {message}"
                )
            # Keep the raw filename too (harmless, and useful for an
            # operator scanning the startup log by file rather than tool
            # name) but never as the ONLY key: resolve() is never called
            # with a ".yaml" suffix.
            reasons.setdefault(path.name, message)

    duplicate_reasons = _duplicate_exclusions(pairs)
    reasons.update(duplicate_reasons)
    manifests = [m for _, m in pairs if m.name not in duplicate_reasons]

    sibling_reasons = _sibling_doc_exclusions(manifests)
    reasons.update(sibling_reasons)
    manifests = [m for m in manifests if m.name not in sibling_reasons]

    reference_reasons = _doc_reference_exclusions(manifests)
    reasons.update(reference_reasons)
    manifests = [m for m in manifests if m.name not in reference_reasons]

    return ManifestLoadResult(sorted(manifests, key=lambda m: m.name), reasons)
