"""Adapter for AlphaFold 3 (``scripts/engines/alphafold3.py``, dispatched
through ``engine.prefix: /alphafold3_venv`` -- an extracted venv mounted
like any other GPU engine's conda environment, not a sibling Docker
container -- see the manifest's own ``engine:`` comment and "How this tool
is dispatched" doc section for the full reasoning).

The manifest's ``chains`` parameter is ``type: array, items: {type:
object}`` for the same reason ``run_boltz``'s is (see that adapter's own
docstring): this project's validator only checks scalar constraints on
array items, not nested object properties, so each chain object's own
shape -- including the ``unpaired_msa``/``paired_msa`` coupling rule -- is
validated HERE, by hand.

``unpaired_msa``/``paired_msa`` are not top-level ``format: path`` schema
entries (they are nested inside ``chains`` objects), so
``app._resolve_path_params`` never sees them; this adapter resolves and
READS each one itself (embedding the actual a3m TEXT inline in AlphaFold
3's JSON, AlphaFold 3's own "expert option" combination -- see the
manifest's doc) rather than passing a path into the engine's own
subprocess, which would need its own extra mount for an arbitrary caller
path.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.schema import Manifest

_SEQUENCE_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWYXBZJUO]+$")
_MAX_CHAINS = 30
_MAX_COPIES = 20


def _read_msa(label: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty path string, got {value!r}.")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"{label} = {value!r} does not resolve to an existing file.")
    return path.read_text()


def _normalize_chain(index: int, entry: Any) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise ValueError(
            f"chains[{index}] must be an object with 'sequence', "
            f"'unpaired_msa' and 'paired_msa' keys, got {type(entry).__name__}"
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

    if "unpaired_msa" not in entry or "paired_msa" not in entry:
        raise ValueError(
            f"chains[{index}] is missing 'unpaired_msa' and/or 'paired_msa'. "
            "Both are required on every entry -- pass null for both to run "
            "this chain MSA-free, or a path for both (e.g. "
            "run_mmseqs_search's unpaired_a3m/paired_a3m outputs) to supply "
            "real alignments. There is no default -- see the manifest's MSA "
            "policy."
        )
    unpaired_msa = entry["unpaired_msa"]
    paired_msa = entry["paired_msa"]
    if (unpaired_msa is None) != (paired_msa is None):
        raise ValueError(
            f"chains[{index}]: unpaired_msa and paired_msa must be BOTH "
            f"null or BOTH a path -- got unpaired_msa={unpaired_msa!r}, "
            f"paired_msa={paired_msa!r}. One null and one a path is not a "
            "valid AlphaFold 3 combination (see the manifest's MSA policy)."
        )

    if unpaired_msa is None:
        # AlphaFold 3's OWN "equivalent to running completely MSA-free"
        # encoding is "" / "" -- NEVER JSON null/null, which instead means
        # "AlphaFold 3, build both MSAs yourself" (exactly the self-search
        # behaviour run_data_pipeline=false exists to prevent). See the
        # manifest's "MSA is optional" section.
        unpaired_msa_content = ""
        paired_msa_content = ""
    else:
        unpaired_msa_content = _read_msa(f"chains[{index}].unpaired_msa", unpaired_msa)
        paired_msa_content = _read_msa(f"chains[{index}].paired_msa", paired_msa)

    copies = entry.get("copies", 1)
    if not isinstance(copies, int) or isinstance(copies, bool) or copies < 1:
        raise ValueError(
            f"chains[{index}].copies must be a positive integer, got {copies!r}."
        )
    if copies > _MAX_COPIES:
        raise ValueError(
            f"chains[{index}].copies = {copies} exceeds the maximum of "
            f"{_MAX_COPIES} identical copies of one chain."
        )

    return {
        "sequence": sequence,
        "unpaired_msa": unpaired_msa_content,
        "paired_msa": paired_msa_content,
        "copies": copies,
    }


def _normalize_chains(chains: Any) -> list[dict[str, Any]]:
    if not isinstance(chains, list) or not chains:
        raise ValueError("chains must be a non-empty array of chain objects.")
    normalized = [_normalize_chain(i, c) for i, c in enumerate(chains)]
    total = sum(c["copies"] for c in normalized)
    if total > _MAX_CHAINS:
        raise ValueError(
            f"chains describes {total} total chain instances (entries plus "
            f"copies), which exceeds the maximum of {_MAX_CHAINS} chains in "
            "one assembly."
        )
    return normalized


def build_args(manifest: Manifest, params: dict[str, Any]) -> list[str]:
    """Serialise validated parameters to one JSON argv token for the
    wrapper script -- same reasoning as ``boltz.build_args``: the job
    specification has to be built server-side (file reads, validation) and
    handed across the process boundary as one blob, this time to a wrapper
    that shells out to a sibling ``docker run`` rather than a conda-mounted
    binary. ``manifest`` is unused -- this adapter backs exactly one tool --
    but is part of every adapter's signature (see ``adapters_discovery``).
    """
    del manifest
    job = {
        "chains": _normalize_chains(params["chains"]),
        "seeds": list(params["seeds"]),
        "num_recycles": params["num_recycles"],
        "num_diffusion_samples": params["num_diffusion_samples"],
        "max_template_date": params["max_template_date"],
        "resolve_msa_overlaps": params["resolve_msa_overlaps"],
        "flash_attention_implementation": params["flash_attention_implementation"],
        "save_embeddings": params["save_embeddings"],
        "save_distogram": params["save_distogram"],
        "buckets": list(params["buckets"]),
        "conformer_max_iterations": params["conformer_max_iterations"],
    }
    return [json.dumps(job)]


def parse_output(manifest: Manifest, run: CompletedRun) -> dict[str, Any]:
    """Read the top-ranked sample's summary confidences plus the collected
    output paths. ``manifest`` is unused (see ``build_args``)."""
    del manifest
    summary_path = run.outputs.get("summary_confidences_json")
    if not summary_path:
        raise ValueError(
            "run_alphafold3's declared 'summary_confidences_json' output "
            f"was not collected. run.outputs was: {run.outputs}"
        )
    summary = json.loads(Path(str(summary_path)).read_text())

    per_sample_cifs = run.outputs.get("per_sample_cifs")
    if isinstance(per_sample_cifs, list):
        num_samples = len(per_sample_cifs)
    elif per_sample_cifs:
        num_samples = 1
    else:
        num_samples = 0

    return {
        "ranking_score": summary.get("ranking_score"),
        "ptm": summary.get("ptm"),
        "iptm": summary.get("iptm"),
        "fraction_disordered": summary.get("fraction_disordered"),
        "has_clash": summary.get("has_clash"),
        "num_samples": num_samples,
    }
