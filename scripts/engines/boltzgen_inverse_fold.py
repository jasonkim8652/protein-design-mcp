#!/usr/bin/env python
"""Build a fixed-structure BoltzGen spec inside the tool, then inverse-fold."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

import yaml


def build_redesign_spec(structure: str, design_chains: list[str]) -> dict:
    if not design_chains:
        raise ValueError("design_chains must name at least one chain")
    return {"entities": [{"file": {
        "path": str(Path(structure).resolve()),
        "design": [{"chain": {"id": chain}} for chain in design_chains],
    }}]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--structure")
    source.add_argument("--design-spec")
    parser.add_argument("--design-chains", default="")
    parser.add_argument("--passthrough", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()
    if args.structure:
        spec = build_redesign_spec(args.structure, [c for c in args.design_chains.split(",") if c])
    else:
        original = Path(args.design_spec).resolve()
        spec = yaml.safe_load(original.read_text())
        # Preserve relative file references when exporting into this call's
        # scratch directory for downstream tools.
        for entity in spec.get("entities", []):
            body = entity.get("file")
            if isinstance(body, dict) and body.get("path"):
                path = Path(body["path"])
                if not path.is_absolute():
                    body["path"] = str((original.parent / path).resolve())
    spec_path = Path.cwd() / "design_spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec, sort_keys=False))
    result = subprocess.run(["boltzgen", "run", str(spec_path), *args.passthrough])
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
