#!/usr/bin/env python
"""Build a BoltzGen design spec from parameters, then run the design step.

Every other tool on this server takes typed parameters through its MCP schema
and the model fills them. The six run_boltzgen_* tools instead demanded a
`design_spec` YAML, and nothing here produced one -- so a planned workflow
could never reach them. A round planned
`run_epitope_scan -> run_rfdiffusion3_binder -> run_mpnn -> run_boltzgen_fold`,
cleared three steps, and the fourth refused for a file no earlier step could
have written.

The spec's content is structured parameters: which chain is designed and how
long, and which target chains to condition on. So this builds it, and writes it
into the working directory where the manifest declares it as an OUTPUT -- which
is what makes `run_boltzgen_fold`, `_analyze` and `_filter` reachable, since
they need the same spec the designs came from.

`--design-spec` remains, for a hand-written spec that says something this
parameter set cannot (multiple designed chains, ligands, templated motifs).
When it is given, nothing is built.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

SPEC_FILENAME = "design_spec.yaml"


def build_design_spec(*, target_structure: str, target_chains: list[str],
                      binder_length_min: int, binder_length_max: int,
                      binder_chain_id: str) -> dict[str, Any]:
    """BoltzGen's `entities` list for a binder against a structure's chains.

    The length is written `min..max` because that is BoltzGen's own grammar; a
    bare integer is a different token there, so a fixed length is still a range
    with equal ends.
    """
    if not target_chains:
        raise ValueError(
            "at least one target chain is required -- an empty include conditions "
            "on nothing, which is unconditional generation wearing a binder "
            "tool's name"
        )
    if binder_length_max < binder_length_min:
        raise ValueError(
            f"binder_length_max ({binder_length_max}) is below binder_length_min "
            f"({binder_length_min})"
        )
    if binder_chain_id in target_chains:
        raise ValueError(
            f"binder_chain_id {binder_chain_id!r} is already a target chain; two "
            "entities claiming one id is a spec BoltzGen accepts and then behaves "
            "unpredictably on"
        )
    return {
        "entities": [
            {"protein": {"id": binder_chain_id,
                         "sequence": f"{binder_length_min}..{binder_length_max}"}},
            {"file": {"path": target_structure,
                      "include": [{"chain": {"id": c}} for c in target_chains]}},
        ]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--design-spec", default=None,
                        help="a hand-written spec; when given, nothing is built")
    parser.add_argument("--target-structure")
    parser.add_argument("--target-chains", default="")
    parser.add_argument("--binder-length-min", type=int)
    parser.add_argument("--binder-length-max", type=int)
    parser.add_argument("--binder-chain-id", default="C")
    parser.add_argument("--passthrough", nargs=argparse.REMAINDER, default=[],
                        help="everything after this is handed to `boltzgen run`")
    args = parser.parse_args()

    workdir = Path.cwd()
    spec_path = workdir / SPEC_FILENAME

    if args.design_spec:
        # Copied rather than referenced: the manifest declares the spec as an
        # output of this step so later steps can consume it, and an output has
        # to be inside the working directory to be collected.
        shutil.copyfile(args.design_spec, spec_path)
    else:
        missing = [n for n, v in (("--target-structure", args.target_structure),
                                  ("--binder-length-min", args.binder_length_min),
                                  ("--binder-length-max", args.binder_length_max))
                   if v is None]
        if missing:
            sys.exit(f"without --design-spec these are required: {', '.join(missing)}")
        spec = build_design_spec(
            target_structure=args.target_structure,
            target_chains=[c for c in args.target_chains.split(",") if c],
            binder_length_min=args.binder_length_min,
            binder_length_max=args.binder_length_max,
            binder_chain_id=args.binder_chain_id,
        )
        spec_path.write_text(yaml.safe_dump(spec, sort_keys=False))

    proc = subprocess.run(["boltzgen", "run", str(spec_path), *args.passthrough],
                          capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
