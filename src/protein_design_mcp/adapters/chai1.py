"""Adapter for Chai-1 (``scripts/engines/chai1.py``, env ``chai1`` -- plain
PyPI ``chai_lab==0.6.1``, not editable).

See ``adapters/boltz.py``'s module docstring for why ``chains`` is
hand-validated here rather than by the manifest schema (this project's
validator only checks scalar constraints on array items, not nested object
properties) and why a relative ``msa`` path is resolved here rather than by
``app._resolve_path_params`` (which only resolves top-level ``format: path``
schema entries, not paths nested inside an array).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.schema import Manifest

_SEQUENCE_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWYXBZJUO]+$")
_MAX_CHAINS = 12
_MAX_COPIES = 20


def _normalize_chain(index: int, entry: Any) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise ValueError(
            f"chains[{index}] must be an object with 'sequence' and 'msa' "
            f"keys, got {type(entry).__name__}"
        )

    sequence = entry.get("sequence")
    if not isinstance(sequence, str) or not sequence:
        raise ValueError(
            f"chains[{index}].sequence must be a non-empty string of "
            "uppercase amino acid codes."
        )
    if not _SEQUENCE_RE.fullmatch(sequence):
        raise ValueError(
            f"chains[{index}].sequence = {sequence!r} contains characters "
            "outside the standard 20 amino acids plus the ambiguity codes "
            "X/B/Z/J/U/O, uppercase only."
        )

    if "msa" not in entry:
        raise ValueError(
            f"chains[{index}] is missing 'msa'. Pass null to run this chain "
            "MSA-free (a deliberate choice), or a path to the plain "
            "'unpaired_a3m' output of run_mmseqs_search for this chain's "
            "sequence. There is no default -- see the manifest's msa policy."
        )
    msa = entry["msa"]
    if msa is not None:
        if not isinstance(msa, str) or not msa:
            raise ValueError(
                f"chains[{index}].msa must be null or a non-empty path "
                f"string, got {msa!r}."
            )
        msa = str(Path(msa).expanduser().resolve())

    copies = entry.get("copies", 1)
    if not isinstance(copies, int) or isinstance(copies, bool) or copies < 1:
        raise ValueError(
            f"chains[{index}].copies must be a positive integer, got {copies!r}."
        )
    if copies > _MAX_COPIES:
        raise ValueError(
            f"chains[{index}].copies = {copies} exceeds the maximum of "
            f"{_MAX_COPIES} identical copies in one assembly."
        )

    return {"sequence": sequence, "msa": msa, "copies": copies}


def _normalize_chains(chains: Any) -> list[dict[str, Any]]:
    if not isinstance(chains, list) or not chains:
        raise ValueError("chains must be a non-empty array of chain objects.")
    normalized = [_normalize_chain(i, c) for i, c in enumerate(chains)]
    total_copies = sum(c["copies"] for c in normalized)
    if total_copies > _MAX_CHAINS:
        raise ValueError(
            f"chains describes {total_copies} total chain instances "
            f"(entries plus copies), which exceeds the maximum of "
            f"{_MAX_CHAINS} chains in one assembly."
        )
    return normalized


def build_args(manifest: Manifest, params: dict[str, Any]) -> list[str]:
    """Serialise validated parameters to one JSON argv token for the
    wrapper script (see its module docstring for why file synthesis has to
    happen inside the engine's own environment). ``manifest`` is unused
    here -- this adapter backs exactly one tool.
    """
    del manifest
    job = {
        "chains": _normalize_chains(params["chains"]),
        "num_trunk_recycles": params["num_trunk_recycles"],
        "num_diffn_timesteps": params["num_diffn_timesteps"],
        "num_diffn_samples": params["num_diffn_samples"],
        "num_trunk_samples": params["num_trunk_samples"],
        "use_esm_embeddings": params["use_esm_embeddings"],
        "low_memory": params["low_memory"],
        "seed": params["seed"],
    }
    return [json.dumps(job)]


def _rank(path: str) -> int:
    match = re.search(r"model_idx_(\d+)\.", path)
    return int(match.group(1)) if match else 10**9


def parse_output(manifest: Manifest, run: CompletedRun) -> dict[str, Any]:
    """Read the rank-0 sample's scores.npz plus the collected output paths.
    ``manifest`` is unused (see ``build_args``).
    """
    del manifest
    score_paths = run.outputs.get("scores")
    if not score_paths:
        raise ValueError(
            "run_chai1's declared 'scores' output was not collected -- no "
            f"scores.model_idx_*.npz file was found. run.outputs was: {run.outputs}"
        )
    if not isinstance(score_paths, list):
        score_paths = [score_paths]

    best_path = min(score_paths, key=_rank)
    with np.load(best_path) as data:
        scores = {key: data[key] for key in data.files}

    structures = run.outputs.get("structures")
    num_structures = len(structures) if isinstance(structures, list) else (
        1 if structures else 0
    )

    return {
        # .item() -- not float()/bool() -- on purpose: chai_lab's own
        # scores.model_idx_*.npz stores these as size-1 (not 0-d) arrays.
        # float()/bool() on a non-0-d array is numpy's own long-deprecated
        # implicit scalar conversion (DeprecationWarning since numpy 1.25);
        # numpy 2.4.6 (this image's unpinned `pip install .` resolves) turns
        # that into a hard TypeError, while numpy 2.2.6 (the host dev env)
        # still only warns -- see task-13-report.md. .item() is the
        # version-stable, explicitly-supported way to pull a single element
        # out of an array regardless of whether it is 0-d or size-1, and
        # works identically on every numpy version.
        "aggregate_score": scores["aggregate_score"].item(),
        "ptm": scores["ptm"].item(),
        "iptm": scores["iptm"].item(),
        "per_chain_ptm": scores["per_chain_ptm"].tolist(),
        "per_chain_pair_iptm": scores["per_chain_pair_iptm"].tolist(),
        "has_inter_chain_clashes": bool(scores["has_inter_chain_clashes"].item()),
        "num_structures": num_structures,
        "caveat": (
            "Chai-1's CLI does not write a PAE matrix to disk, so run_ipsae "
            "cannot score this tool's output -- use per_chain_pair_iptm "
            "here, or predict with run_boltz/run_protenix/run_openfold3 "
            "instead when a PAE-based ipSAE score is required."
        ),
    }
