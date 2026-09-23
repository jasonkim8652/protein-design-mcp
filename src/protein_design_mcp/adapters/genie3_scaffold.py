"""Adapter for Genie 3's unconditional sampler (`run_genie3_scaffold`).

Almost all translation work happens in the wrapper script
(``scripts/engines/genie3_scaffold.py``), which builds Genie 3's own
experiment YAML (its CLI takes a config file path, not key=value overrides).
This adapter only serializes parameters into the wrapper's argv and reads
the collected PDBs back.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.schema import Manifest


def _bool_str(value: bool) -> str:
    return "true" if value else "false"


def build_args(manifest: Manifest, params: dict[str, Any]) -> list[str]:
    """Translate validated parameters into the wrapper script's argv."""
    del manifest
    return [
        "--model-variant",
        str(params["model_variant"]),
        "--min-length",
        str(params["min_length"]),
        "--max-length",
        str(params["max_length"]),
        "--length-step",
        str(params["length_step"]),
        "--num-samples",
        str(params["num_samples"]),
        "--batch-size",
        str(params["batch_size"]),
        "--direction-scale",
        str(params["direction_scale"]),
        "--eta",
        str(params["eta"]),
        "--n-sample-step",
        str(params["n_sample_step"]),
        "--noise-scale",
        str(params["noise_scale"]),
        # Pinned false, not read from params: the schema no longer exposes it.
        # Genie 3's side-chain pass is guarded by three assertions, and the
        # third is `assert ...sampler.predict_sequence` -- which this tool
        # fixes to false, because sequence design is run_mpnn's job. Confirmed
        # by running it: predict_sidechain=true raises AssertionError on
        # workflow.py:204, AFTER the main stage has finished, discarding the
        # generation. The flag is still passed because the CLI expects it.
        "--predict-sidechain",
        _bool_str(False),
        "--seed",
        str(params["seed"]),
    ]


def _length_from_pdb(path: str) -> int:
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
            "run_genie3_scaffold's declared 'backbones' output was not "
            f"collected -- no PDB was found. run.outputs was: {run.outputs}"
        )
    if not isinstance(paths, list):
        paths = [paths]

    backbones = [
        {"id": Path(path).stem, "length": _length_from_pdb(path)}
        for path in sorted(paths)
    ]

    return {"backbones": backbones, "num_backbones": len(backbones)}
