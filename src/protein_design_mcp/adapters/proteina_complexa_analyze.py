"""Adapter for Proteina-Complexa's ``analyze`` pipeline step
(``run_proteina_complexa_analyze``).

Proteina-Complexa's own `evaluate` step -- the one that would normally
write `binder_results_{config}_{job}.csv` for `analyze` to read -- is
excluded from this server by design (see WAVE-COMMON.md and the manifest's
doc). This adapter is therefore the one place in this tool that MATERIALISES
that CSV, from structured parameters, before invoking the real
`complexa analyze`. See the manifest doc's "Verified: diversity needs only
structures, not refolding metrics" section for why this degrades correctly
when only structures/sequences are supplied.

``build_args`` has no ``workdir`` parameter (every adapter's signature is
fixed at ``(manifest, params) -> list[str]``), so it cannot be handed the
scratch directory directly. It recovers it instead from
``params["structure_paths"]``, which by the time this runs has already been
STAGED by the dispatcher (``engine.stage: ["structure_paths"]`` in the
manifest) into ``workdir/structure_paths/<basename>`` -- so
``Path(staged_path).parent.parent`` IS the workdir, reliably, because
``staging.stage_inputs`` always uses exactly that ``workdir/<param name>/``
shape. The materialised CSV is written directly into that workdir (a
sibling directory, ``analyze_results/``), which is why this adapter -- unlike
every other one in this project -- has a real side effect (a disk write)
inside ``build_args`` rather than being a pure function of its inputs.
"""

from __future__ import annotations

import ast
import csv
import json
from pathlib import Path
from typing import Any

from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.schema import Manifest

_CONFIG_PATH = "/home/jk661/projects/proteina-complexa/configs/analyze.yaml"
_CONFIG_STEM = "analyze"
_RESULTS_DIR = "analyze_results"
_JOB_ID = 0
_RUN_NAME = "mcp_run"
_TASK_NAME = "mcp_analyze_set"

# Optional per-design metric -> materialised CSV column name. "external" is
# this adapter's fixed stand-in for a folding-model name -- see the
# manifest doc's naming section.
_OPTIONAL_METRIC_COLUMNS = {
    "designability_scrmsd_ca": "_res_scRMSD_ca_external",
    "codesignability_scrmsd_ca": "_res_co_scRMSD_ca_external",
    "codesignability_scrmsd_all_atom": "_res_co_scRMSD_all_atom_external",
    "interface_hbonds_tmol": "generated_n_interface_hbonds_tmol",
}


def _omega_scalar(value: Any) -> str:
    """Same rendering convention as ``adapters.boltzgen_filter._omega_scalar``."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value))


def _omega_list(values: list[Any]) -> str:
    return "[" + ",".join(_omega_scalar(v) for v in values) + "]"


def _omega_value(value: Any) -> str:
    """Render an arbitrary JSON-ish value (nested dict/list/scalar) as an
    OmegaConf/Hydra CLI override fragment -- used for the free-form
    threshold objects, which are nested (unlike every other object
    parameter in this project so far, e.g. boltzgen_filter's flat
    metrics_override)."""
    if isinstance(value, dict):
        items = ",".join(f"{key}:{_omega_value(val)}" for key, val in value.items())
        return "{" + items + "}"
    if isinstance(value, list):
        return "[" + ",".join(_omega_value(v) for v in value) + "]"
    return _omega_scalar(value)


def _require_matching_length(name: str, values: list[Any] | None, expected: int) -> None:
    if values is None:
        return
    if len(values) != expected:
        raise ValueError(
            f"{name} has {len(values)} entries but structure_paths has "
            f"{expected}; every optional per-design array must cover every "
            "design when supplied (see the manifest doc)."
        )


def _write_materialized_csv(csv_path: Path, params: dict[str, Any]) -> None:
    structure_paths = params["structure_paths"]
    sequences = params["sequences"]
    n = len(structure_paths)

    if len(sequences) != n:
        raise ValueError(
            f"sequences has {len(sequences)} entries but structure_paths has "
            f"{n}; they must be the same length, matched by index."
        )
    for param_name in _OPTIONAL_METRIC_COLUMNS:
        _require_matching_length(param_name, params.get(param_name), n)

    fieldnames = ["pdb_path", "sequence", "run_name", "task_name", "id_gen"]
    active_optional = [name for name in _OPTIONAL_METRIC_COLUMNS if params.get(name) is not None]
    fieldnames.extend(_OPTIONAL_METRIC_COLUMNS[name] for name in active_optional)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(n):
            row = {
                "pdb_path": str(structure_paths[index]),
                "sequence": sequences[index],
                "run_name": _RUN_NAME,
                "task_name": _TASK_NAME,
                "id_gen": index,
            }
            for name in active_optional:
                row[_OPTIONAL_METRIC_COLUMNS[name]] = params[name][index]
            writer.writerow(row)


def build_args(manifest: Manifest, params: dict[str, Any]) -> list[str]:
    """Materialise ``binder_results_{config}_{job}.csv`` into the scratch
    workdir, then translate the remaining parameters into ``complexa
    analyze``'s argv. ``manifest`` is unused -- part of every adapter's
    signature, see ``protein_design_mcp.adapters.boltz`` for why.
    """
    del manifest

    structure_paths = params["structure_paths"]
    if not structure_paths:
        raise ValueError("structure_paths must contain at least one entry.")
    # Recover the scratch workdir from the STAGED first structure path --
    # see this module's own docstring for why this is reliable.
    workdir = Path(structure_paths[0]).parent.parent

    csv_path = workdir / _RESULTS_DIR / f"binder_results_{_CONFIG_STEM}_{_JOB_ID}.csv"
    _write_materialized_csv(csv_path, params)

    overrides = [
        f"++results_dir={_RESULTS_DIR}",
        f"++base_config_name={_CONFIG_STEM}",
        f"++result_type={_omega_scalar(params['result_type'])}",
        f"++aggregation.compute_diversity={_omega_scalar(params['compute_foldseek_diversity'])}",
        f"++aggregation.compute_mmseqs_diversity={_omega_scalar(params['compute_mmseqs_diversity'])}",
        f"++aggregation.mmseqs_min_seq_id={_omega_scalar(params['mmseqs_min_seq_id'])}",
        f"++aggregation.mmseqs_coverage={_omega_scalar(params['mmseqs_coverage'])}",
        f"++aggregation.analysis_modes={_omega_list(params['analysis_modes'])}",
        f"++aggregation.require_all_thresholds={_omega_scalar(params['require_all_thresholds'])}",
    ]

    for key in (
        "success_thresholds",
        "designability_thresholds",
        "ca_codesignability_thresholds",
        "allatom_codesignability_thresholds",
    ):
        value = params.get(key)
        if value is not None:
            overrides.append(f"++aggregation.{key}={_omega_value(value)}")

    return [_CONFIG_PATH, *overrides, "--verbose"]


def _parse_diversity_cell(raw: str | None) -> dict[str, float | int | None] | None:
    """Parse a ``(score, num_clusters, num_samples)`` tuple cell -- pandas
    writes a Python tuple's ``repr()`` into a plain CSV cell, since neither
    ``compute_foldseek_diversity`` nor ``compute_mmseqs_diversity`` unpacks
    its own return value into separate columns before saving (verified by
    reading ``result_analysis/compute_diversity.py``). ``ast.literal_eval``
    is used rather than a bare ``eval`` because this text originates from a
    file THIS adapter wrote and complexa's own numeric formatting -- never
    from an arbitrary caller-controlled expression."""
    if raw is None or raw == "" or raw.lower() == "none":
        return None
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return None
    if not isinstance(parsed, (tuple, list)) or len(parsed) != 3:
        return None
    score, num_clusters, num_samples = parsed
    return {
        "score": float(score),
        "num_clusters": int(num_clusters),
        "num_samples": int(num_samples),
    }


def _read_all_generated_row(paths: list[str] | str | None) -> dict[str, str] | None:
    if not paths:
        return None
    candidates = paths if isinstance(paths, list) else [paths]
    for path in candidates:
        if "all_generated" not in Path(path).name:
            continue
        with Path(path).open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if rows:
            return rows[0]
    return None


def parse_output(manifest: Manifest, run: CompletedRun) -> dict[str, Any]:
    """Read the "all_generated" diversity rows ``analyze.py`` itself wrote.
    ``manifest`` is unused (see ``build_args``)."""
    del manifest

    foldseek_row = _read_all_generated_row(run.outputs.get("diversity_foldseek_csv"))
    mmseqs_row = _read_all_generated_row(run.outputs.get("diversity_mmseqs_csv"))

    foldseek_diversity = None
    if foldseek_row is not None:
        for key, value in foldseek_row.items():
            if key.startswith("_res_diversity_foldseek_"):
                foldseek_diversity = _parse_diversity_cell(value)
                break

    mmseqs_diversity = None
    if mmseqs_row is not None:
        for key, value in mmseqs_row.items():
            if key.startswith("_res_diversity_mmseqs_"):
                mmseqs_diversity = _parse_diversity_cell(value)
                break

    num_designs = None
    combined_paths = run.outputs.get("combined_results_csv")
    if combined_paths:
        combined_path = combined_paths[0] if isinstance(combined_paths, list) else combined_paths
        with Path(combined_path).open(newline="") as handle:
            num_designs = sum(1 for _ in csv.DictReader(handle))

    return {
        "foldseek_diversity": foldseek_diversity,
        "mmseqs_diversity": mmseqs_diversity,
        "num_designs": num_designs,
    }
