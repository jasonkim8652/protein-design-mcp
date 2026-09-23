"""Adapter for Genie 2's unconditional sampler (`run_genie2`).

Wraps `python -m genie.sample_unconditional` directly -- a real, importable
console entry point (namespace package `genie`, confirmed live:
`python -c "import genie.sample_unconditional"` succeeds under the `genie2`
env). No wrapper script is needed: every parameter this tool exposes maps
straight onto one of the script's own argparse flags (see its `--help`,
quoted in the manifest doc).

`--rootdir` is hardcoded to the genie2 checkout's own `results/` directory
(absolute) rather than left at the script's relative default of `results` --
the dispatcher's cwd is a scratch workdir, not the genie2 checkout, so a
relative rootdir would never find `results/base/checkpoints/epoch=40.ckpt`.
`--outdir` is `output` (relative), landing inside that same scratch workdir,
which is where `run.outputs["backbones"]` (glob `output/pdbs/*.pdb`) expects
to find it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.schema import Manifest

_GENIE2_ROOTDIR = "/home/jk661/projects/genie2/results"


def build_args(manifest: Manifest, params: dict[str, Any]) -> list[str]:
    """Translate validated parameters into `sample_unconditional.py`'s argv."""
    del manifest
    args = [
        "--name",
        "base",
        "--epoch",
        "40",
        "--rootdir",
        _GENIE2_ROOTDIR,
        "--scale",
        str(params["scale"]),
        "--outdir",
        "output",
        "--num_samples",
        str(params["num_samples"]),
        "--batch_size",
        str(params["batch_size"]),
        "--min_length",
        str(params["min_length"]),
        "--max_length",
        str(params["max_length"]),
        "--length_step",
        str(params["length_step"]),
        "--num_devices",
        "1",
    ]
    if params["sequential_order"]:
        args.append("--sequential_order")
    return args


def _length_from_pdb(path: str) -> int:
    """Count CA atoms in a PDB file -- the sample's true residue count,
    rather than trusting it can always be parsed back out of the filename.
    """
    count = 0
    for line in Path(path).read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")) and line[12:16].strip() == "CA":
            count += 1
    return count


def parse_output(manifest: Manifest, run: CompletedRun) -> dict[str, Any]:
    """Read every generated backbone's length back out of its own PDB."""
    del manifest
    paths = run.outputs.get("backbones")
    if not paths:
        raise ValueError(
            "run_genie2's declared 'backbones' output was not collected -- "
            f"no PDB was found. run.outputs was: {run.outputs}"
        )
    if not isinstance(paths, list):
        paths = [paths]

    backbones = [
        {"id": Path(path).stem, "length": _length_from_pdb(path)}
        for path in sorted(paths)
    ]

    return {"backbones": backbones, "num_backbones": len(backbones)}
