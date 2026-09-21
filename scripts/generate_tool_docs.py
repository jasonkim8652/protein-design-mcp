#!/usr/bin/env python3
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
