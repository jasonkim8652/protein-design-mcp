"""Adapter for BoltzGen's ``filtering`` pipeline step (``run_boltzgen_filter``).

This step runs no model -- it is pure dataframe filtering and ranking over
columns BoltzGen's own ``analysis`` step already wrote to
``design_dir/aggregate_metrics_*.csv`` (see ``boltzgen.task.filter.filter.Filter``).
This adapter wraps ``boltzgen run <design_spec> --steps filtering`` and pushes
every one of ``Filter.__init__``'s real parameters through
``--config filtering <key>=<value> ...`` rather than the top-level CLI flags
(``--budget``, ``--alpha``, ...) BoltzGen's own CLI also offers for a subset of
them. This is a deliberate choice, not an oversight: BoltzGen's own
``BinderDesignPipeline.__init__`` builds the filtering step's argv as
``filter_args + protocol_preset_args + user_config_args`` (in that order), and
OmegaConf's dotlist merge lets a LATER duplicate key win -- so a protocol
preset (e.g. ``peptide-anything`` setting ``alpha=0.01``) can silently
override an explicit ``--alpha`` the caller passed, because the top-level
flag is applied BEFORE the preset in that concatenation (verified by reading
``cli/boltzgen.py``'s ``BinderDesignPipeline.__init__``). Routing every
parameter through ``--config filtering`` instead means this tool's own values
are always the LAST word, regardless of ``--protocol`` (which this tool
therefore hardcodes to ``protein-anything``, a genuine no-op for an isolated
``filtering`` step -- verified live by diffing ``boltzgen configure``'s
resolved ``filtering.yaml`` across all six protocols with everything else
held fixed: only the fields this adapter already overrides differ).
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.schema import Manifest

# Filter.__init__ parameters this tool exposes and always sets explicitly
# via --config filtering, in addition to design_dir/outdir/metrics_override/
# additional_filters/size_buckets (built up separately below).
_SIMPLE_KEYS = (
    "budget",
    "top_budget",
    "use_affinity",
    "filter_cysteine",
    "from_inverse_folded",
    "filter_designfolding",
    "filter_bindingsite",
    "filter_target_aligned",
    "filter_biased",
    "refolding_rmsd_threshold",
    "modality",
    "peptide_type",
    "alpha",
    "random_state",
    "num_liability_plots",
    "plot_seq_logos",
)


def _omega_scalar(value: Any) -> str:
    """Render one Python value as an OmegaConf/YAML flow-style scalar.

    Verified live (2026-09-22): ``boltzgen configure`` round-trips exactly
    this shape (``null`` for None, lowercase ``true``/``false`` for bools,
    a bare number for int/float, a double-quoted string otherwise) back into
    the same Python value when it resolves ``config/filtering.yaml``.
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value))


def _omega_mapping(mapping: dict[str, Any]) -> str:
    items = ", ".join(f"{key}: {_omega_scalar(val)}" for key, val in mapping.items())
    return "{" + items + "}"


def _require_keys(entry: Any, label: str) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise ValueError(f"{label} entry {entry!r} must be an object.")
    return entry


def _render_additional_filters(filters: list[Any]) -> str:
    rendered = []
    for entry in filters:
        entry = _require_keys(entry, "additional_filters")
        feature = entry.get("feature")
        threshold = entry.get("threshold")
        lower_is_better = entry.get("lower_is_better")
        if not isinstance(feature, str) or not feature:
            raise ValueError(
                f"additional_filters entry {entry!r} is missing a non-empty "
                "string 'feature'."
            )
        if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
            raise ValueError(
                f"additional_filters entry {entry!r} is missing a numeric "
                "'threshold'."
            )
        if not isinstance(lower_is_better, bool):
            raise ValueError(
                f"additional_filters entry {entry!r} is missing a boolean "
                "'lower_is_better'."
            )
        rendered.append(
            "{"
            f"feature: {_omega_scalar(feature)}, threshold: {_omega_scalar(threshold)}, "
            f"lower_is_better: {_omega_scalar(lower_is_better)}"
            "}"
        )
    return "[" + ", ".join(rendered) + "]"


def _render_size_buckets(buckets: list[Any]) -> str:
    rendered = []
    for entry in buckets:
        entry = _require_keys(entry, "size_buckets")
        values: dict[str, Any] = {}
        for field_name in ("min", "max", "num_designs"):
            value = entry.get(field_name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(
                    f"size_buckets entry {entry!r} is missing an integer "
                    f"{field_name!r}."
                )
            values[field_name] = value
        rendered.append(
            "{"
            f"min: {values['min']}, max: {values['max']}, "
            f"num_designs: {values['num_designs']}"
            "}"
        )
    return "[" + ", ".join(rendered) + "]"


def build_args(manifest: Manifest, params: dict[str, Any]) -> list[str]:
    """Translate validated parameters into ``boltzgen run``'s argv.

    ``manifest`` is unused here -- part of every adapter's signature, see
    ``protein_design_mcp.adapters.boltz`` for why.
    """
    del manifest

    config_overrides = [f"design_dir={params['design_dir']}", "outdir=."]
    for key in _SIMPLE_KEYS:
        config_overrides.append(f"{key}={_omega_scalar(params[key])}")

    metrics_override = params.get("metrics_override")
    if metrics_override is not None:
        if not isinstance(metrics_override, dict):
            raise ValueError(
                "metrics_override must be an object mapping metric name to "
                "a number or null."
            )
        config_overrides.append(f"metrics_override={_omega_mapping(metrics_override)}")

    additional_filters = params.get("additional_filters")
    if additional_filters:
        config_overrides.append(
            f"additional_filters={_render_additional_filters(additional_filters)}"
        )

    size_buckets = params.get("size_buckets")
    if size_buckets:
        config_overrides.append(f"size_buckets={_render_size_buckets(size_buckets)}")

    return [
        str(params["design_spec"]),
        "--output",
        ".",
        "--protocol",
        "protein-anything",
        "--steps",
        "filtering",
        "--config",
        "filtering",
        *config_overrides,
    ]


def _to_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _to_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


_TOTAL_RE = re.compile(r"Total number of designs:\s*(\d+)")
_REMAINING_RE = re.compile(r"Remaining designs:\s*(\d+)")


def parse_output(manifest: Manifest, run: CompletedRun) -> dict[str, Any]:
    """Read the quality+diversity-selected CSV and the filter-pass counts
    BoltzGen printed to stdout. ``manifest`` is unused (see ``build_args``).
    """
    del manifest
    csv_paths = run.outputs.get("ranked_metrics_csv")
    if not csv_paths:
        raise ValueError(
            "run_boltzgen_filter's declared 'ranked_metrics_csv' output was "
            f"not collected -- no ranked CSV was found. run.outputs was: "
            f"{run.outputs}"
        )
    path = csv_paths[0] if isinstance(csv_paths, list) else csv_paths

    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    selected_designs = [
        {
            "id": row.get("id"),
            "final_rank": _to_int(row.get("final_rank")),
            "designed_sequence": row.get("designed_sequence"),
            "designed_chain_sequence": row.get("designed_chain_sequence"),
            "num_design": _to_int(row.get("num_design")),
            "design_to_target_iptm": _to_float(row.get("design_to_target_iptm")),
            "design_ptm": _to_float(row.get("design_ptm")),
            "min_design_to_target_pae": _to_float(row.get("min_design_to_target_pae")),
        }
        for row in rows
    ]

    total_match = _TOTAL_RE.search(run.stdout)
    remaining_matches = _REMAINING_RE.findall(run.stdout)

    return {
        "selected_designs": selected_designs,
        "num_selected": len(selected_designs),
        "num_total_designs": int(total_match.group(1)) if total_match else None,
        "num_passing_all_filters": (
            int(remaining_matches[-1]) if remaining_matches else None
        ),
    }
