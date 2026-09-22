"""Adapter for Promera (github.com/bjing2016/promera).

Promera's own CLI (``python -m promera input=<dir> output=<dir>
msa_dir=<dir> ...``) takes a DIRECTORY of target schema JSON files plus a
directory of MSA files keyed by a SHA-256 hash of each chain's sequence
(``tinyprot.msa.hash_sequence``/``load_msa_from_dir``, read from source).
Building that hash-keyed directory needs the exact sequence text alongside
each caller-supplied a3m path, which ``build_args`` alone cannot do (it has
no access to the scratch working directory -- see
``protein_design_mcp.dispatch.env.EnvDispatcher.build_command``). The
wrapper script (``scripts/engines/promera.py``, which DOES run with cwd set
to that scratch directory) does the actual file staging and hashing; this
adapter only serializes ``chains``/``msa`` to JSON strings for its argv and
parses the confidence JSON it writes back out.
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

    ``manifest`` is unused here -- Promera backs exactly one tool -- but is
    part of every adapter's signature (see ``protein_design_mcp.app.ADAPTERS``)
    so a future adapter module backing several tools on one engine repo can
    branch on ``manifest.name`` without changing the call site.
    """
    del manifest
    chains = params["chains"]
    if not isinstance(chains, dict) or not chains:
        raise ValueError(
            "run_promera.chains must be a non-empty object mapping chain "
            f"labels to schema entries, got {chains!r}"
        )

    msa = params["msa"]
    if msa is not None:
        if not isinstance(msa, dict):
            raise ValueError(
                f"run_promera.msa must be null or an object, got {type(msa).__name__}"
            )
        missing = set(chains) - set(msa)
        extra = set(msa) - set(chains)
        if missing or extra:
            raise ValueError(
                "run_promera.msa must have exactly one key per chain in "
                f"'chains' ({sorted(chains)}); missing: {sorted(missing)}, "
                f"unexpected: {sorted(extra)}"
            )

    resolved_msa = _resolve_msa_paths(msa)

    return [
        "--schema",
        json.dumps(chains),
        "--msa",
        json.dumps(resolved_msa),
        "--recycling-steps",
        str(params["recycling_steps"]),
        "--diffusion-samples",
        str(params["diffusion_samples"]),
        "--diffusion-steps",
        str(params["diffusion_steps"]),
        "--num-seeds",
        str(params["num_seeds"]),
    ]


def parse_output(manifest: Manifest, run: CompletedRun) -> dict[str, Any]:
    """Read Promera's own confidence JSON back out. ``manifest`` is unused
    (see ``build_args``)."""
    del manifest
    conf_path = run.outputs.get("confidence_json")
    if not conf_path:
        raise ValueError(
            "Promera's declared 'confidence_json' output was not collected "
            f"-- no confidence file was found. run.outputs was: {run.outputs}"
        )
    try:
        return json.loads(Path(conf_path).read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Promera's confidence_json at {conf_path} was not valid JSON: {exc}"
        ) from exc
