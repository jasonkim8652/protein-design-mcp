"""Adapter for RFdiffusion 1.1.0 binder generation
(`scripts/engines/rfdiffusion.py`, env `SE3nv`).

Translates this tool's schema into RFdiffusion's own Hydra `key=value`
overrides. There is no job file to synthesise (unlike Boltz's YAML or
RFdiffusion3's JSON) -- the wrapper forwards these overrides to
`run_inference.py` almost unchanged (see that module's docstring for the
one exception: a relative `ckpt_override_path`).
"""

from __future__ import annotations

import re
from typing import Any

from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.schema import Manifest

_OUTPUT_PREFIX = "out/design"
_BETA_CKPT = "models/Complex_beta_ckpt.pt"

_CHECKPOINT_RE = re.compile(r"Reading checkpoint from (\S+)")
_CONTIG_RE = re.compile(r"Using contig:\s*(\[.*\])")


def build_args(manifest: Manifest, params: dict[str, Any]) -> list[str]:
    """Build RFdiffusion's Hydra overrides. ``manifest`` is unused (this
    adapter backs exactly one tool) but is part of every adapter's
    signature -- see ``adapters_discovery``.
    """
    del manifest

    args = [
        f"inference.input_pdb={params['target_pdb']}",
        f"inference.output_prefix={_OUTPUT_PREFIX}",
        f"inference.num_designs={params['num_designs']}",
        f"contigmap.contigs=[{params['contig']}]",
        f"diffuser.T={params['diffusion_steps']}",
        f"denoiser.noise_scale_ca={params['noise_scale_ca']}",
        f"denoiser.noise_scale_frame={params['noise_scale_frame']}",
        f"inference.deterministic={params['deterministic']}",
    ]

    hotspots = params.get("hotspot_res") or []
    if hotspots:
        args.append(f"ppi.hotspot_res=[{','.join(hotspots)}]")

    partial_t = params.get("partial_t")
    if partial_t is not None:
        args.append(f"diffuser.partial_T={partial_t}")

    provide_seq = params.get("provide_seq") or []
    if provide_seq:
        if partial_t is None:
            raise ValueError(
                "provide_seq requires partial_t to be set -- RFdiffusion "
                "only fixes sequence identity within a region it is "
                "partially (re-)diffusing, not during from-scratch "
                "generation. Set partial_t, or omit provide_seq."
            )
        args.append(f"contigmap.provide_seq=[{','.join(provide_seq)}]")

    ckpt_variant = params.get("ckpt_variant", "auto")
    if ckpt_variant == "beta":
        if provide_seq:
            raise ValueError(
                "ckpt_variant='beta' cannot be combined with provide_seq -- "
                "there is no beta+InpaintSeq checkpoint. Use "
                "ckpt_variant='auto' (RFdiffusion selects the InpaintSeq "
                "checkpoint automatically when provide_seq is set), or "
                "drop provide_seq."
            )
        args.append(f"inference.ckpt_override_path={_BETA_CKPT}")

    return args


def parse_output(manifest: Manifest, run: CompletedRun) -> dict[str, Any]:
    """Report design count plus what the engine itself echoed back for the
    checkpoint and contig actually used. ``manifest`` is unused (see
    ``build_args``).
    """
    del manifest

    structures = run.outputs.get("structures")
    if not structures:
        raise ValueError(
            "run_rfdiffusion_binder's declared 'structures' output was not "
            f"collected -- no backbone PDB was found. run.outputs was: "
            f"{run.outputs}"
        )
    if not isinstance(structures, list):
        structures = [structures]

    checkpoint_match = _CHECKPOINT_RE.search(run.stdout)
    contig_match = _CONTIG_RE.search(run.stdout)

    return {
        "num_structures": len(structures),
        "checkpoint_used": checkpoint_match.group(1) if checkpoint_match else None,
        "contig_used": contig_match.group(1) if contig_match else None,
        "sequence_caveat": (
            "Backbone only -- diffused (binder) residues carry no real "
            "sequence (written as poly-glycine placeholders). Run "
            "run_mpnn on these structures to design an actual sequence."
        ),
    }
