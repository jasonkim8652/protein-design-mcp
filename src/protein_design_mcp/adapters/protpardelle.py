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
from protein_design_mcp.validation import ToolInputError

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


def assert_break_between_segments(contig: str) -> None:
    """Every "/" must have a segment on BOTH sides.

    Settled by a live matrix against the engine, not by reading one of its
    parsers:

        B2-505;/;80-120   ->  two chains (A:504, B:108)
        B2-505;80-120     ->  ONE fused 613-mer -- runs, but not a binder
        B2-505;/          ->  ValueError: invalid literal for int(): '/'

    A dangling or leading "/" is a token with nothing to separate; it reaches
    the engine's scaffold branch, which calls int("/"). The chain COUNT is
    unaffected (segment groups, see below), which is why counting alone let
    this through.
    """
    groups = contig.split("/")
    if len(groups) == 1:
        return
    empty = [
        i for i, group in enumerate(groups)
        if not [token for token in group.split(";") if token]
    ]
    if empty:
        raise ToolInputError(
            f"run_protpardelle.contig = {contig!r} has a '/' with no segment on "
            "one side of it. A chain break separates two segments -- "
            "'B2-505;/;80-120' is the binder shape (target chain, break, "
            "diffused length range). A trailing or leading '/' reaches the "
            "engine's scaffold parser, which fails on int('/')."
        )


def contig_chain_count(contig: str) -> int:
    """How many chains a contig describes, counted the way Protpardelle counts.

    Protpardelle counts SEGMENT GROUPS, not separators. Counting
    ``contig.count("/") + 1`` instead made a DANGLING break -- ``"B2-505;/"``,
    with nothing after it -- read as two chains, so a two-entry
    ``total_lengths`` passed validation here and was then refused from inside
    Protpardelle's own sampler:

        AssertionError: Contig B2-505;/ has 1 chains but length ranges
        specify 2 chains.

    A live round sent exactly that. Our count has to agree with the engine's,
    or validation waves malformed input through to an assertion.
    """
    groups = [
        [token for token in group.split(";") if token]
        for group in contig.split("/")
    ]
    return max(1, sum(1 for group in groups if group))


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
    assert_break_between_segments(contig)
    num_chains = contig_chain_count(contig)
    if len(total_lengths) != num_chains:
        raise ValueError(
            f"total_lengths has {len(total_lengths)} entries but contig "
            f"{contig!r} describes {num_chains} chain(s) -- these must match. "
            "A chain is a SEGMENT, not a separator: '/' between two segments "
            "makes two chains ('B2-505;/;80-120'), while a '/' with nothing "
            "after it adds none ('B2-505;/' is one chain)."
        )
    return total_lengths


#: Checkpoints that produce SIDE CHAINS, not just a backbone. An all-atom
#: model runs ProteinMPNN to assign the sequence, so it needs ProteinMPNN's
#: own weights -- which the backbone-only checkpoints never touch. Only cc94
#: qualifies among the four this tool exposes: the manifest doc calls cc83 and
#: cc95 backbone-only and cc78 experimental, and cc94 is the one observed
#: loading ProteinMPNN live.
ALL_ATOM_MODELS = {"cc94"}

#: Where Protpardelle-1c looks for those weights. Not bundled with its own
#: checkpoints, and absent on this host -- a live round spent ~35s of GPU
#: sampling 500 backbone steps and then died on a bare FileNotFoundError
#: naming this path, which no caller had chosen and no message explained.
MPNN_WEIGHTS = Path(
    "/home/jk661/projects/protpardelle-1c/model_params/ProteinMPNN"
    "/vanilla_model_weights/v_48_020.pt"
)


def build_args(manifest: Manifest, params: dict[str, Any]) -> list[str]:
    """Translate validated parameters into the wrapper script's argv."""
    del manifest
    model = str(params.get("model") or "")
    if model in ALL_ATOM_MODELS and not MPNN_WEIGHTS.exists():
        raise ToolInputError(
            f"run_protpardelle.model = {model!r} is the all-atom checkpoint: it "
            "assigns side chains with ProteinMPNN, whose weights are not "
            f"installed on this host ({MPNN_WEIGHTS} is missing). Use a "
            "backbone-only checkpoint instead -- cc83 (the default), cc95 or "
            "cc78 -- and design the sequence with run_mpnn, or install "
            "ProteinMPNN's v_48_020 weights at that path."
        )
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
