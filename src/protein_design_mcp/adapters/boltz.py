"""Adapter for Boltz-2 (``scripts/engines/boltz.py``, env ``boltz`` — the
user's own editable fork at ``~/projects/lightning-boltz-dev``, not upstream
Boltz; see the manifest's doc for why that is stated to the caller).

The manifest's ``chains`` parameter is ``type: array, items: {type: object}``
because this project's validator (``protein_design_mcp.validation``) only
checks scalar constraints on array items, not nested object properties (see
``validation._check_value``) -- so each chain object's own shape is
validated HERE, by hand, with the same "clear message a retrying model can
act on" standard ``validation.py`` itself uses.

``msa`` inside a chain object is not a top-level ``format: path`` schema
entry, so ``app._resolve_path_params`` never sees it and a relative path
would otherwise resolve against the dispatcher's scratch workdir instead of
the caller's intended location. This adapter resolves it itself, the same
way (``Path(...).resolve()`` against the server's own cwd), before the
wrapper script ever runs.
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


def build_args(manifest: Manifest, params: dict[str, Any]) -> list[str]:
    """Serialise validated parameters to one JSON argv token for the wrapper
    script (see its own module docstring for why: Boltz takes a YAML job
    file, not a flat argv, and file synthesis has to happen inside the
    engine's own environment/cwd, which this function -- running in the
    server process, before dispatch -- does not have). ``manifest`` is
    unused here -- this adapter backs exactly one tool -- but is part of
    every adapter's signature (see ``adapters_discovery``).
    """
    del manifest
    job = {
        "chains": _normalize_chains(params["chains"]),
        "recycling_steps": params["recycling_steps"],
        "sampling_steps": params["sampling_steps"],
        "diffusion_samples": params["diffusion_samples"],
        "step_scale": params["step_scale"],
        "use_potentials": params["use_potentials"],
        "output_format": params["output_format"],
        "max_msa_seqs": params["max_msa_seqs"],
        "subsample_msa": params["subsample_msa"],
        "num_subsampled_msa": params["num_subsampled_msa"],
        "seed": params["seed"],
    }
    return [json.dumps(job)]


def parse_output(manifest: Manifest, run: CompletedRun) -> dict[str, Any]:
    """Read the confidence summary Boltz wrote for the rank-0 model plus the
    collected output paths. ``manifest`` is unused (see ``build_args``).
    """
    del manifest
    confidence_paths = run.outputs.get("confidence_json")
    if not confidence_paths:
        raise ValueError(
            "run_boltz's declared 'confidence_json' output was not "
            f"collected -- no confidence file was found. run.outputs was: "
            f"{run.outputs}"
        )
    if not isinstance(confidence_paths, list):
        confidence_paths = [confidence_paths]

    # Boltz names each model's file "..._model_<rank>.json" with rank 0 the
    # highest-confidence sample (writer.py sorts by confidence_score before
    # assigning ranks) -- picking the lexicographically-sorted first path
    # is not reliable once diffusion_samples > 1 digit (e.g. "_model_10"
    # would sort before "_model_2"), so the rank is read out of the
    # filename explicitly instead.
    def _rank(path: str) -> int:
        match = re.search(r"_model_(\d+)\.json$", path)
        return int(match.group(1)) if match else 10**9

    best_path = min(confidence_paths, key=_rank)
    summary = json.loads(Path(best_path).read_text())

    structures = run.outputs.get("structures")
    num_structures = len(structures) if isinstance(structures, list) else (
        1 if structures else 0
    )

    return {
        "confidence_score": summary.get("confidence_score"),
        "ptm": summary.get("ptm"),
        "iptm": summary.get("iptm"),
        "protein_iptm": summary.get("protein_iptm"),
        "complex_plddt": summary.get("complex_plddt"),
        "complex_iplddt": summary.get("complex_iplddt"),
        "complex_pde": summary.get("complex_pde"),
        "complex_ipde": summary.get("complex_ipde"),
        "num_structures": num_structures,
        "fork_notice": (
            "Ran the user's own editable fork of Boltz at "
            "~/projects/lightning-boltz-dev, not upstream boltz-community/boltz."
        ),
    }
