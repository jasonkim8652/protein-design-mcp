"""Adapter for Protpardelle-1c's multi-chain (binder) sampler
(`run_protpardelle`).

Almost all translation work happens in the wrapper script
(``scripts/engines/protpardelle.py``), which builds Protpardelle-1c's own
sampling-config YAML (its CLI takes a config file path, not key=value
overrides). This adapter serializes parameters into the wrapper's argv,
validates ``hotspots``/``total_lengths`` structurally (the manifest schema
cannot express "null or array-of-pattern-matched-strings" -- see
``adapters.rf3`` for the same class of precedent with ``msa``), and reads
the collected samples back, grouped by generated PDB with every chain's own
sequence reported (target chain(s) included, unchanged).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1

from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.schema import Manifest

_HOTSPOT_RE = re.compile(r"^[A-Za-z]\d+[A-Za-z]?$")
_PDB_PARSER = PDBParser(QUIET=True)


def _validate_hotspots(hotspots: Any) -> list[str] | None:
    if hotspots is None:
        return None
    if not isinstance(hotspots, list) or not all(isinstance(h, str) for h in hotspots):
        raise ValueError(
            f"hotspots must be null or a list of strings, got {hotspots!r}"
        )
    bad = [h for h in hotspots if not _HOTSPOT_RE.match(h)]
    if bad:
        raise ValueError(
            f"hotspots contains malformed tag(s) {bad!r}; expected "
            "'{chain_id}{residue_index}', e.g. 'A19'"
        )
    return hotspots


def _validate_total_lengths(total_lengths: Any, contig: str) -> list[list[int]]:
    if not isinstance(total_lengths, list) or not total_lengths:
        raise ValueError(f"total_lengths must be a non-empty list, got {total_lengths!r}")
    for entry in total_lengths:
        if (
            not isinstance(entry, list)
            or len(entry) != 2
            or not all(isinstance(x, int) for x in entry)
            or entry[0] > entry[1]
        ):
            raise ValueError(
                f"total_lengths entry {entry!r} must be a [min, max] pair of "
                "integers with min <= max"
            )
    # contig's own chain-break count ("/" segments) is one less than the
    # number of chains it describes.
    num_chains = contig.split(";").count("/") + 1
    if len(total_lengths) != num_chains:
        raise ValueError(
            f"total_lengths has {len(total_lengths)} entries but contig "
            f"{contig!r} describes {num_chains} chain(s) (one more than its "
            "'/' chain-break count) -- these must match"
        )
    return total_lengths


def build_args(manifest: Manifest, params: dict[str, Any]) -> list[str]:
    """Translate validated parameters into the wrapper script's argv."""
    del manifest
    contig = str(params["contig"])
    hotspots = _validate_hotspots(params["hotspots"])
    total_lengths = _validate_total_lengths(params["total_lengths"], contig)

    args = [
        "--target-pdb",
        str(params["target_pdb"]),
        "--contig",
        contig,
        "--total-lengths",
        json.dumps(total_lengths),
        "--hotspots",
        json.dumps(hotspots),
        "--model",
        str(params["model"]),
        "--step-scale",
        str(params["step_scale"]),
        "--schurn",
        str(params["schurn"]),
        "--crop-cond-start",
        str(params["crop_cond_start"]),
        "--translation",
        json.dumps(list(params["translation"])),
        "--num-samples",
        str(params["num_samples"]),
        "--batch-size",
        str(params["batch_size"]),
    ]
    seed = params.get("seed")
    if seed is not None:
        args += ["--seed", str(seed)]
    return args


def _chains_from_pdb(path: str) -> list[dict[str, Any]]:
    structure = _PDB_PARSER.get_structure(Path(path).stem, path)
    model = next(structure.get_models())
    chains = []
    for chain in model.get_chains():
        resnames = [residue.resname for residue in chain.get_residues()]
        sequence = seq1("".join(resnames))
        chains.append({"chain_id": chain.id, "length": len(sequence)})
    return chains


def parse_output(manifest: Manifest, run: CompletedRun) -> dict[str, Any]:
    """Read every chain's length out of each generated PDB."""
    del manifest
    paths = run.outputs.get("samples")
    if not paths:
        raise ValueError(
            "run_protpardelle's declared 'samples' output was not "
            f"collected -- no PDB was found. run.outputs was: {run.outputs}"
        )
    if not isinstance(paths, list):
        paths = [paths]

    samples = [
        {"id": Path(path).stem, "chains": _chains_from_pdb(path)}
        for path in sorted(paths)
    ]

    return {"samples": samples, "num_samples": len(samples)}
