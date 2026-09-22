"""Adapter for BoltzGen's ``analysis`` pipeline step (``run_boltzgen_analyze``).

Reassembles up to THREE prior tool calls' outputs
(``run_boltzgen_design``/``run_boltzgen_inverse_fold``, ``run_boltzgen_fold``,
optionally ``run_boltzgen_design_fold``) into the specific directory tree
``boltzgen.task.analyze.analyze.Analyze`` expects (``design_dir`` itself for
the original files, ``design_dir/refold_cif``/``fold_out_npz`` for the
fold outputs, ``design_dir/refold_design_cif``/``fold_out_design_npz`` for
the optional design-fold outputs -- all hardcoded relative to ONE
``design_dir`` in ``analyze.py``'s own ``init_datasets``, not independently
configurable). ``manifest.engine.stage``/``stage_subdir`` do the actual
file placement; this adapter derives ``design_dir``'s path from the
already-staged ``generated_files`` the same way ``run_boltzgen_fold``'s
adapter does.

As with ``run_boltzgen_filter``, every one of ``Analyze.__init__``'s
parameters this tool exposes is always pushed through
``--config analysis <key>=<value>`` (not a top-level CLI flag) so a
protocol preset can never silently override it -- ``--protocol`` is
consequently never passed.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.schema import Manifest

_BOOLEAN_KEYS = (
    "affinity_metrics",
    "backbone_fold_metrics",
    "allatom_fold_metrics",
    "noncovalents_original",
    "noncovalents_refolded",
    "delta_sasa_original",
    "delta_sasa_refolded",
    "largest_hydrophobic",
    "largest_hydrophobic_refolded",
    "run_clustering",
    "liability_analysis",
    "designfolding_metrics",
    "use_design_mask_for_target",
    "compute_lddts",
)


def _bool_str(value: bool) -> str:
    return "true" if value else "false"


def build_args(manifest: Manifest, params: dict[str, Any]) -> list[str]:
    """Translate validated (and already-staged) parameters into
    ``boltzgen run``'s argv. ``manifest`` is unused -- part of every
    adapter's signature, see ``protein_design_mcp.adapters.boltz``.
    """
    del manifest
    generated_files = params["generated_files"]
    design_dir = str(Path(generated_files[0]).parent)

    config_overrides = [
        f"design_dir={design_dir}",
        f"data.cfg.num_workers={params['num_workers']}",
        f"num_processes={params['num_processes']}",
    ]
    for key in _BOOLEAN_KEYS:
        config_overrides.append(f"{key}={_bool_str(params[key])}")
    config_overrides.append(f"liability_modality={params['liability_modality']}")
    config_overrides.append(f"liability_peptide_type={params['liability_peptide_type']}")

    return [
        str(params["design_spec"]),
        "--output",
        ".",
        "--steps",
        "analysis",
        # moldir MUST go through the top-level flag, not --config -- see
        # run_boltzgen_fold's identical comment (BoltzGen's own CLI resolves
        # a "huggingface:repo:file" artifact reference to a real local path
        # BEFORE embedding it in every step's fixed args; a --config
        # override bypasses that and hands the engine the raw unresolved
        # string, which fails hard).
        "--moldir",
        str(params["moldir"]),
        "--config",
        "analysis",
        *config_overrides,
    ]


def parse_output(manifest: Manifest, run: CompletedRun) -> dict[str, Any]:
    """Count the rows analyze wrote to its own aggregate metrics CSV.
    ``manifest`` is unused (see ``build_args``).
    """
    del manifest
    csv_path = run.outputs.get("aggregate_metrics_csv")
    if not csv_path:
        raise ValueError(
            "run_boltzgen_analyze's declared 'aggregate_metrics_csv' output "
            f"was not collected -- no metrics file was found. run.outputs "
            f"was: {run.outputs}"
        )
    if isinstance(csv_path, list):
        csv_path = csv_path[0]

    with Path(csv_path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    return {"num_designs_analyzed": len(rows)}
