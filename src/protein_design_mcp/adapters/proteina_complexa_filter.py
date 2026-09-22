"""Adapter for Proteina-Complexa's ``filter`` pipeline step
(``run_proteina_complexa_filter``).

Wraps ``complexa filter <config> ++overrides... --verbose``. The one input
(``rewards_csv``) is staged into ``workdir/rewards_csv/`` by the dispatcher
(``engine.stage`` in the manifest); this adapter points Proteina-Complexa's
own ``root_path`` override AT that same staged directory, which sidesteps
two problems at once:

1. ``filter.py``'s own ``setup()`` (only called when ``cfg.root_path`` is
   unset) asserts ``torch.cuda.is_available()`` -- inherited, unused,
   dead-code-looking copy from ``generate.py``'s own ``setup()`` (verified
   by reading ``filter.py``: nothing after ``setup()`` returns uses a GPU).
   Supplying ``root_path`` explicitly skips that function entirely.
2. ``filter.py`` looks for files matching
   ``rewards_{config_name}_*.csv`` inside ``root_path`` by literal
   ``os.listdir`` + prefix/suffix match (``filter.py:114-117``) -- the
   staged copy keeps its ORIGINAL basename, so as long as this tool's fixed
   config resolves to the SAME ``config_name`` a
   ``run_proteina_complexa_generate`` call used (both point at
   ``configs/search_binder_local_pipeline.yaml``), the staged file already
   matches with no renaming needed.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.schema import Manifest

_CONFIG_PATH = "/home/jk661/projects/proteina-complexa/configs/search_binder_local_pipeline.yaml"

# Matches the destination subdirectory engine.stage's staging convention
# creates for a parameter named "rewards_csv" (workdir/rewards_csv/...) --
# used here as a RELATIVE root_path override, resolved against the
# subprocess's own cwd (the dispatcher's scratch workdir).
_ROOT_PATH = "rewards_csv"


def _omega_scalar(value: Any) -> str:
    """Same rendering convention as ``adapters.boltzgen_filter._omega_scalar``."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value))


def build_args(manifest: Manifest, params: dict[str, Any]) -> list[str]:
    """Translate validated parameters into ``complexa filter``'s argv.

    ``manifest`` is unused here -- part of every adapter's signature, see
    ``protein_design_mcp.adapters.boltz`` for why. ``params["rewards_csv"]``
    is already the STAGED path by the time this runs (app.py stages before
    calling build_args) -- unused directly here since root_path/complexa's
    own file-discovery finds it by filename, but its presence is what makes
    root_path non-empty.
    """
    del manifest
    overrides = [
        f"++root_path={_ROOT_PATH}",
        "++base_config_name=search_binder_local_pipeline",
        f"++generation.filter.filter_samples_limit={_omega_scalar(params['filter_samples_limit'])}",
        f"++generation.filter.dedup_sequence={_omega_scalar(params['dedup_sequence'])}",
        f"++generation.filter.reward_threshold={_omega_scalar(params.get('reward_threshold'))}",
    ]
    return [_CONFIG_PATH, *overrides, "--verbose"]


def _to_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


_INITIAL_RE = re.compile(r"Initial samples:\s*(\d+)")
_FINAL_RE = re.compile(r"Final top samples:\s*(\d+)")


def parse_output(manifest: Manifest, run: CompletedRun) -> dict[str, Any]:
    """Read the top-samples CSV ``filter.py`` itself wrote. ``manifest`` is
    unused (see ``build_args``)."""
    del manifest
    csv_paths = run.outputs.get("top_samples_csv")
    if not csv_paths:
        raise ValueError(
            "run_proteina_complexa_filter's declared 'top_samples_csv' output "
            f"was not collected -- no top-samples CSV was found. run.outputs was: {run.outputs}"
        )
    path = csv_paths[0] if isinstance(csv_paths, list) else csv_paths

    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    selected_designs = [
        {**row, "total_reward": _to_float(row.get("total_reward"))} for row in rows
    ]

    all_rewards_paths = run.outputs.get("all_rewards_csv")
    num_total_designs = None
    if all_rewards_paths:
        all_path = all_rewards_paths[0] if isinstance(all_rewards_paths, list) else all_rewards_paths
        with Path(all_path).open(newline="") as handle:
            num_total_designs = sum(1 for _ in csv.DictReader(handle))

    if num_total_designs is None:
        final_match = _FINAL_RE.search(run.stdout)
        num_total_designs = int(final_match.group(1)) if final_match else None
        del final_match

    return {
        "selected_designs": selected_designs,
        "num_selected": len(selected_designs),
        "num_total_designs": num_total_designs,
    }
