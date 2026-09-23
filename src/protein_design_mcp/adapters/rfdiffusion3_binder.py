"""Adapter for RFdiffusion3's binder (PPI) path
(`scripts/engines/rfd3.py`, env `foundry`).

Builds the one JSON argv the shared wrapper expects: a `DesignInputSpecification`-
shaped `"job"` dict (RFdiffusion3's own pydantic conditioning schema -- see
`rfd3.inference.input_parsing.DesignInputSpecification`, which is
`extra="forbid"`, so only its recognised field names may appear) plus an
`"engine"` dict of diffusion-sampling knobs that are separate Hydra overrides,
not part of that JSON object at all.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.schema import Manifest


def build_args(manifest: Manifest, params: dict[str, Any]) -> list[str]:
    """``manifest`` is unused (this adapter backs exactly one tool) but is
    part of every adapter's signature -- see ``adapters_discovery``.
    """
    del manifest

    job: dict[str, Any] = {
        "input": params["target_pdb"],
        "contig": params["contig"],
        "redesign_motif_sidechains": params["redesign_motif_sidechains"],
    }
    if params.get("select_hotspots") is not None:
        job["select_hotspots"] = params["select_hotspots"]
    if params.get("length") is not None:
        job["length"] = params["length"]
    if params.get("partial_t") is not None:
        job["partial_t"] = params["partial_t"]

    engine = {
        "diffusion_batch_size": params["diffusion_batch_size"],
        "num_timesteps": params["num_timesteps"],
        "step_scale": params["step_scale"],
        "seed": params.get("seed"),
    }

    return [json.dumps({"job": job, "engine": engine})]


_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V", "UNK": "X",
}


def _sequences_from_cif(path: Path) -> dict[str, str] | None:
    """``{chain: one-letter sequence}`` from a (gzipped) mmCIF, or None.

    Column positions are read from the ``_atom_site.`` loop header rather than
    assumed: a fixed-index parse of this exact file silently produced 504
    one-residue "chains" because the columns are not in PDB order.

    Returns None rather than raising. The sequence is a bonus on top of the
    structure the caller asked for, and failing the whole call because one CIF
    did not parse would throw away a finished GPU run.
    """
    try:
        opener = gzip.open if path.suffix == ".gz" else open
        columns: list[str] = []
        chains: dict[str, list[str]] = {}
        seen: set[tuple[str, str]] = set()
        with opener(path, "rt") as handle:  # type: ignore[operator]
            for raw in handle:
                line = raw.strip()
                if line.startswith("_atom_site."):
                    columns.append(line.split(".", 1)[1])
                    continue
                if not columns or not line.startswith(("ATOM", "HETATM")):
                    continue
                parts = line.split()
                if len(parts) != len(columns):
                    continue
                row = dict(zip(columns, parts))
                chain = row.get("auth_asym_id") or row.get("label_asym_id")
                number = row.get("auth_seq_id") or row.get("label_seq_id")
                comp = row.get("label_comp_id")
                if not chain or number is None or comp is None:
                    continue
                key = (chain, number)
                if key in seen:
                    continue
                seen.add(key)
                chains.setdefault(chain, []).append(_THREE_TO_ONE.get(comp, "X"))
        if not chains:
            return None
        return {c: "".join(v) for c, v in chains.items()}
    except Exception:
        return None


def parse_output(manifest: Manifest, run: CompletedRun) -> dict[str, Any]:
    """Read the rank-0 (first) sample's metadata sidecar plus the collected
    output paths. ``manifest`` is unused (see ``build_args``).
    """
    del manifest

    metadata_paths = run.outputs.get("metadata_json")
    if not metadata_paths:
        raise ValueError(
            "run_rfdiffusion3_binder's declared 'metadata_json' output was "
            f"not collected -- no metadata sidecar was found. run.outputs "
            f"was: {run.outputs}"
        )
    if not isinstance(metadata_paths, list):
        metadata_paths = [metadata_paths]

    structures = run.outputs.get("structure_cif")
    num_structures = len(structures) if isinstance(structures, list) else (
        1 if structures else 0
    )

    first = json.loads(Path(sorted(metadata_paths)[0]).read_text())

    cif_list = structures if isinstance(structures, list) else ([structures] if structures else [])
    return {
        "num_structures": num_structures,
        "metrics": first.get("metrics"),
        "diffused_index_map": first.get("diffused_index_map"),
        "ckpt_path": first.get("ckpt_path"),
        # One {chain: sequence} per structure, in structure_cif's order.
        # RFdiffusion3 CO-GENERATES a real sequence -- a live run's designed
        # chain A came back with 18 distinct residue types, not the
        # poly-alanine placeholder the other generators here emit. Leaving it
        # inside a gzipped CIF meant a caller could not tell it existed, and
        # the predictable next move was to run run_mpnn to get one: discarding
        # a sequence the model had already produced and, without
        # chains_to_design, redesigning the target along with it.
        "sequences": [_sequences_from_cif(Path(p)) for p in cif_list],
    }
