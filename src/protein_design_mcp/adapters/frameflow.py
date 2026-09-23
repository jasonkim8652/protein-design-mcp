"""Adapter for FrameFlow's unconditional sampler (`run_frameflow`).

Builds Hydra command-line overrides directly (`key=value` positional args) --
no wrapper script is needed since every parameter this tool exposes maps
straight onto one of `inference_unconditional.yaml`'s own config keys (see
the manifest doc, and `docs/superpowers/reviews/2026-09-22-gpu-engine-survey.md`
#7 for the verified base invocation).

`inference.output_dir` is deliberately never set here -- confirmed live
(survey) to have no effect; see the manifest's "outputs" comment for why the
scratch-workdir-as-cwd contract is what actually contains the writes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.schema import Manifest

_CKPT_PATH = {
    "pdb": "/home/jk661/projects/frameflow/weights/pdb/published.ckpt",
    "pdb_amortization": "/home/jk661/projects/frameflow/weights/pdb_amortization/published.ckpt",
    "scope": "/home/jk661/projects/frameflow/weights/scope/published.ckpt",
}


def _hydra_bool(value: bool) -> str:
    return "true" if value else "false"


def build_args(manifest: Manifest, params: dict[str, Any]) -> list[str]:
    """Translate validated parameters into Hydra overrides."""
    del manifest
    checkpoint_variant = str(params["checkpoint_variant"])
    return [
        "-cn",
        "inference_unconditional",
        f"inference.ckpt_path={_CKPT_PATH[checkpoint_variant]}",
        f"inference.seed={params['seed']}",
        "inference.num_gpus=1",
        # inference_unconditional.yaml's OWN default sets
        # inference.samples.length_subset=[70, 100, 200, 300] -- confirmed
        # live to silently override min_length/max_length/length_step
        # otherwise (all four hardcoded lengths were sampled regardless of
        # what min/max were set to). Must be nulled out explicitly for this
        # tool's own length parameters to take effect at all.
        "inference.samples.length_subset=null",
        f"inference.samples.min_length={params['min_length']}",
        f"inference.samples.max_length={params['max_length']}",
        f"inference.samples.length_step={params['length_step']}",
        f"inference.samples.samples_per_length={params['samples_per_length']}",
        f"inference.interpolant.min_t={params['min_t']}",
        f"inference.interpolant.sampling.num_timesteps={params['num_timesteps']}",
        f"inference.interpolant.self_condition={_hydra_bool(params['self_condition'])}",
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
            "run_frameflow's declared 'backbones' output was not collected "
            f"-- no sample.pdb was found. run.outputs was: {run.outputs}"
        )
    if not isinstance(paths, list):
        paths = [paths]

    backbones = [
        {"id": Path(path).parent.name, "length": _length_from_pdb(path)}
        for path in sorted(paths)
    ]

    return {"backbones": backbones, "num_backbones": len(backbones)}
