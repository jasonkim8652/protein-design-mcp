"""Adapter for AlphaFold2-Multimer via ColabFold (`colabfold_batch`).

The remote-MSA-server restriction (see the manifest's summary/doc) is
enforced structurally here, not merely documented: this tool has no
``msa_mode`` parameter at all (see ``run_alphafold2_multimer.yaml``'s
schema), so ``build_args`` can only ever emit ``--msa-mode single_sequence``
(when ``msa`` is ``null``) or a bare ``--msa-path`` for the wrapper to use as
ColabFold's input file in place of a FASTA (when ``msa`` is a path) --
neither code path can construct one of ColabFold's ``mmseqs2_*`` flags.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.schema import Manifest


def build_args(manifest: Manifest, params: dict[str, Any]) -> list[str]:
    """Translate validated parameters into the wrapper script's argv.

    ``manifest`` is unused here -- this engine backs exactly one tool -- but
    is part of every adapter's signature (see
    ``protein_design_mcp.app.ADAPTERS``) so a future adapter module backing
    several tools on one engine repo can branch on ``manifest.name`` without
    changing the call site.
    """
    del manifest
    sequences = params["sequences"]
    if not isinstance(sequences, list) or not sequences:
        raise ValueError(
            f"run_alphafold2_multimer.sequences must be a non-empty list, got {sequences!r}"
        )

    args = [
        "--sequences",
        json.dumps(sequences),
        "--num-recycle",
        str(params["num_recycle"]),
        "--num-models",
        str(params["num_models"]),
        "--num-seeds",
        str(params["num_seeds"]),
        "--random-seed",
        str(params["random_seed"]),
        "--num-ensemble",
        str(params["num_ensemble"]),
        "--pair-mode",
        str(params["pair_mode"]),
        "--pair-strategy",
        str(params["pair_strategy"]),
        "--rank",
        str(params["rank"]),
        "--stop-at-score",
        str(params["stop_at_score"]),
    ]
    if params["use_dropout"]:
        args.append("--use-dropout")

    msa = params["msa"]
    if msa is None:
        # The ONLY MSA-related flag this adapter can ever emit for a null
        # msa. There is no parameter through which a caller can reach one
        # of ColabFold's mmseqs2_* (remote-server) modes.
        args += ["--msa-mode", "single_sequence"]
    else:
        args += ["--msa-path", str(Path(msa).resolve())]
    return args


def parse_output(manifest: Manifest, run: CompletedRun) -> dict[str, Any]:
    """Read ColabFold's own top-ranked scores JSON back out. ``manifest`` is
    unused (see ``build_args``)."""
    del manifest
    scores_path = run.outputs.get("scores_json")
    if not scores_path:
        raise ValueError(
            "ColabFold's declared 'scores_json' output was not collected -- "
            f"no scores file was found. run.outputs was: {run.outputs}"
        )
    try:
        return json.loads(Path(scores_path).read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"ColabFold's scores_json at {scores_path} was not valid JSON: {exc}"
        ) from exc
