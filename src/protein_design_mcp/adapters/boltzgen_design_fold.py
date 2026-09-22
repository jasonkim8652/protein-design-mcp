"""Adapter for BoltzGen's ``design_folding`` pipeline step
(``run_boltzgen_design_fold``) -- the design refolded ALONE, target absent.

Same underlying Predict/``fold.yaml`` task as ``run_boltzgen_fold`` (see
that adapter's module docstring for the staging rationale, identical here),
with two extra config overrides BoltzGen's own CLI sets for this mode
(``writer.designfolding=true``, ``data.cfg.return_designfolding=true``) and
different output subdirectories (``refold_design_cif/``,
``fold_out_design_npz/`` instead of ``refold_cif/``, ``fold_out_npz/``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.schema import Manifest

# Target-relative fields (protein_iptm, target_ptm, ligand_iptm,
# design_to_target_iptm, design_residue_iptm, interaction PAEs) are omitted
# here -- the target is not present in this mode's input at all, so those
# columns are not meaningful (see the manifest's "What you get back").
_METRIC_KEYS = (
    "design_ptm",
    "design_iptm",
    "design_iiptm",
    "ptm",
    "iptm",
    "complex_plddt",
    "complex_iplddt",
    "complex_pde",
    "complex_ipde",
)


def build_args(manifest: Manifest, params: dict[str, Any]) -> list[str]:
    """Translate validated (and already-staged) parameters into
    ``boltzgen run``'s argv. ``manifest`` is unused -- part of every
    adapter's signature, see ``protein_design_mcp.adapters.boltz``.
    """
    del manifest
    generated_files = params["generated_files"]
    design_dir = str(Path(generated_files[0]).parent)

    return [
        str(params["design_spec"]),
        "--output",
        ".",
        "--steps",
        "design_folding",
        "--devices",
        "1",
        # checkpoint/moldir/use_kernels MUST go through the top-level flags,
        # not --config -- see run_boltzgen_fold's identical comment
        # (verified live: the same "Invalid moldir" failure hits this step
        # too, since it shares fold.yaml with run_boltzgen_fold).
        "--folding_checkpoint",
        str(params["folding_checkpoint"]),
        "--moldir",
        str(params["moldir"]),
        "--use_kernels",
        str(params["use_kernels"]),
        "--config",
        "design_folding",
        f"data.design_dir={design_dir}",
        f"output={design_dir}",
        f"data.cfg.num_workers={params['num_workers']}",
        "trainer.devices=1",
        f"recycling_steps={params['recycling_steps']}",
        f"sampling_steps={params['sampling_steps']}",
        f"diffusion_samples={params['diffusion_samples']}",
        "writer.designfolding=true",
        "data.cfg.return_designfolding=true",
    ]


def _best_sample_metrics(path: str) -> dict[str, Any]:
    """Read one design's fold_out_design_npz file and return the metrics of
    its highest-confidence sample (0.8*iptm + 0.2*ptm), matching
    ``FoldingWriter.write_on_batch_end``'s own selection.
    """
    data = np.load(path)
    iptm = data["iptm"]
    ptm = data["ptm"]
    confidence = 0.8 * iptm + 0.2 * ptm
    best_idx = int(np.argmax(confidence))

    metrics: dict[str, Any] = {}
    for key in _METRIC_KEYS:
        if key in data.files:
            metrics[key] = float(data[key][best_idx])
    metrics["num_samples"] = int(len(iptm))
    metrics["best_sample_index"] = best_idx
    return metrics


def parse_output(manifest: Manifest, run: CompletedRun) -> dict[str, Any]:
    """Read the best-sample metrics out of each design's
    fold_out_design_npz. ``manifest`` is unused (see ``build_args``).
    """
    del manifest
    metrics_paths = run.outputs.get("refold_metrics")
    if not metrics_paths:
        raise ValueError(
            "run_boltzgen_design_fold's declared 'refold_metrics' output "
            f"was not collected -- no metrics file was found. run.outputs "
            f"was: {run.outputs}"
        )
    if not isinstance(metrics_paths, list):
        metrics_paths = [metrics_paths]

    refolds = [
        {"id": Path(path).stem, **_best_sample_metrics(path)}
        for path in sorted(metrics_paths)
    ]

    return {"refolds": refolds, "num_refolds": len(refolds)}
