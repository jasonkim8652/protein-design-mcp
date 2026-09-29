#!/usr/bin/env python3
"""Normalize and deduplicate a completed staging tree into a Docker payload.

Run only after environment/asset staging and their audits have completed.
The staging tree is modified; original installed environments are never inputs.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rootfs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rootfs, output = args.rootfs.resolve(), args.output.resolve()
    if rootfs == Path("/") or output.is_relative_to(rootfs):
        parser.error("Use a separate staging tree and an output outside that tree")
    if not (rootfs / "opt/conda/envs").is_dir() or not (rootfs / "opt/models").is_dir():
        parser.error("The completed environment and model staging trees are required")
    if any(p.name not in {"opt", "usr", "alphafold3_venv"} for p in rootfs.iterdir()):
        parser.error("Unexpected rootfs content; external assets must not enter the image")
    packages = rootfs / "opt/conda/envs/BindCraft/lib/python3.10/site-packages"
    if any(packages.glob("pyrosetta*")) or (packages / "rosetta").exists():
        parser.error("Restricted PyRosetta/Rosetta payload found")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        parser.error("Output already exists; choose a new archive path")
    # All staging helpers make independent copies. Normalize before creating
    # image layers, avoiding a later chmod layer duplicating the large payload.
    subprocess.run(["chmod", "-R", "a+rX", str(rootfs)], check=True)
    subprocess.run(["hardlink", "-t", "-s", "1048576", str(rootfs)], check=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    subprocess.run(["tar", "--sort=name", "--numeric-owner", "--owner=0", "--group=0",
                    "-cf", str(temporary), "-C", str(rootfs), "."], check=True)
    temporary.replace(output)
    print(f"Prepared {output.stat().st_size} bytes at {output}", flush=True)


if __name__ == "__main__":
    main()
