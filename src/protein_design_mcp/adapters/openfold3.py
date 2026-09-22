"""Adapter for OpenFold3 (``scripts/engines/openfold3.py``, env
``openfold3`` -- plain PyPI install, not editable).

See ``adapters/boltz.py``'s module docstring for why ``chains`` is
hand-validated here rather than by the manifest schema, and why a relative
``msa`` path is resolved here rather than by ``app._resolve_path_params``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

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


def _normalize_seeds(seeds: Any) -> list[int]:
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("seeds must be a non-empty array of integers.")
    for seed in seeds:
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError(f"seeds must all be integers, got {seed!r}.")
    return list(seeds)


def build_args(manifest: Manifest, params: dict[str, Any]) -> list[str]:
    """Serialise validated parameters to one JSON argv token for the
    wrapper script. ``manifest`` is unused here -- this adapter backs
    exactly one tool.
    """
    del manifest
    job = {
        "chains": _normalize_chains(params["chains"]),
        "num_diffusion_samples": params["num_diffusion_samples"],
        "seeds": _normalize_seeds(params["seeds"]),
    }
    return [json.dumps(job)]


def parse_output(manifest: Manifest, run: CompletedRun) -> dict[str, Any]:
    """Read the lowest (seed, sample) pair's aggregated confidence plus the
    collected output paths. ``manifest`` is unused (see ``build_args``).
    """
    del manifest
    confidence_paths = run.outputs.get("confidence_aggregated_json")
    if not confidence_paths:
        raise ValueError(
            "run_openfold3's declared 'confidence_aggregated_json' output "
            f"was not collected -- no aggregated confidence file was "
            f"found. run.outputs was: {run.outputs}"
        )
    if not isinstance(confidence_paths, list):
        confidence_paths = [confidence_paths]

    def _sort_key(path: str) -> tuple[int, int]:
        seed_match = re.search(r"seed_(\d+)", path)
        sample_match = re.search(r"_sample_(\d+)_", path)
        seed = int(seed_match.group(1)) if seed_match else 10**9
        sample = int(sample_match.group(1)) if sample_match else 10**9
        return (seed, sample)

    best_path = min(confidence_paths, key=_sort_key)
    summary = json.loads(Path(best_path).read_text())

    structures = run.outputs.get("structures")
    num_structures = len(structures) if isinstance(structures, list) else (
        1 if structures else 0
    )

    return {
        "avg_plddt": summary.get("avg_plddt"),
        "ptm": summary.get("ptm"),
        "iptm": summary.get("iptm"),
        "gpde": summary.get("gpde"),
        "has_clash": summary.get("has_clash"),
        "sample_ranking_score": summary.get("sample_ranking_score"),
        "chain_ptm": summary.get("chain_ptm"),
        "chain_pair_iptm": summary.get("chain_pair_iptm"),
        "num_structures": num_structures,
        "num_diffusion_samples_cap": 5,
    }
