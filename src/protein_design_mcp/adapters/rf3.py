"""Adapter for RoseTTAFold3 (`rf3`, RosettaCommons' `foundry` monorepo).

RF3's own JSON input format (``rf3/utils/inference.py::InferenceInput.
from_json_dict``) takes a ``components`` list plus a top-level ``msa_paths``
dict keyed by ``chain_id`` -- both must be written into a JSON FILE the
``rf3 fold`` CLI reads via ``inputs=<path>``. Building that file needs a
working directory (see ``protein_design_mcp.dispatch.env.EnvDispatcher.
build_command`` -- ``build_args`` never gets one), so the wrapper script
(``scripts/engines/rf3.py``, which runs with cwd set to the scratch
directory) does the actual file writing; this adapter only serializes
``chains``/``msa`` to JSON strings for its argv and parses the confidence
JSON RF3 writes back out.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.schema import Manifest


def _resolve_msa_paths(msa: dict[str, Any] | None) -> dict[str, Any] | None:
    """Resolve each non-null msa path to an absolute path, the same way
    ``app._resolve_path_params`` does for a top-level ``format: path``
    parameter -- ``msa`` here is a nested object, so that central resolution
    never sees its individual values."""
    if msa is None:
        return None
    resolved: dict[str, Any] = {}
    for chain_id, value in msa.items():
        resolved[chain_id] = str(Path(value).resolve()) if value is not None else None
    return resolved


def build_args(manifest: Manifest, params: dict[str, Any]) -> list[str]:
    """Translate validated parameters into the wrapper script's argv.

    ``manifest`` is unused here -- RF3 backs exactly one tool -- but is part
    of every adapter's signature (see ``protein_design_mcp.app.ADAPTERS``) so
    a future adapter module backing several tools on one engine repo can
    branch on ``manifest.name`` without changing the call site.
    """
    del manifest
    chains = params["chains"]
    if not isinstance(chains, list) or not chains:
        raise ValueError(
            f"run_rf3.chains must be a non-empty list of chain objects, got {chains!r}"
        )
    chain_ids: list[str] = []
    for entry in chains:
        if not isinstance(entry, dict) or "chain_id" not in entry or "sequence" not in entry:
            raise ValueError(
                "run_rf3.chains entries must each have 'chain_id' and "
                f"'sequence', got {entry!r}"
            )
        chain_ids.append(str(entry["chain_id"]))
    if len(set(chain_ids)) != len(chain_ids):
        raise ValueError(
            f"run_rf3.chains has duplicate chain_id value(s): {chain_ids}"
        )

    msa = params["msa"]
    if msa is not None:
        if not isinstance(msa, dict):
            raise ValueError(
                f"run_rf3.msa must be null or an object, got {type(msa).__name__}"
            )
        missing = set(chain_ids) - set(msa)
        extra = set(msa) - set(chain_ids)
        if missing or extra:
            raise ValueError(
                "run_rf3.msa must have exactly one key per chain_id in "
                f"'chains' ({sorted(chain_ids)}); missing: {sorted(missing)}, "
                f"unexpected: {sorted(extra)}"
            )

    resolved_msa = _resolve_msa_paths(msa)

    return [
        "--chains",
        json.dumps(chains),
        "--msa",
        json.dumps(resolved_msa),
        "--n-recycles",
        str(params["n_recycles"]),
        "--diffusion-batch-size",
        str(params["diffusion_batch_size"]),
        "--num-steps",
        str(params["num_steps"]),
    ]


def parse_output(manifest: Manifest, run: CompletedRun) -> dict[str, Any]:
    """Read RF3's own summary_confidences JSON back out. ``manifest`` is
    unused (see ``build_args``)."""
    del manifest
    conf_path = run.outputs.get("summary_confidences_json")
    if not conf_path:
        raise ValueError(
            "RF3's declared 'summary_confidences_json' output was not "
            f"collected -- no summary file was found. run.outputs was: {run.outputs}"
        )
    try:
        return json.loads(Path(conf_path).read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"RF3's summary_confidences_json at {conf_path} was not valid JSON: {exc}"
        ) from exc
