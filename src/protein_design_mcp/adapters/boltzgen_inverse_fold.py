"""Adapter for BoltzGen's ``inverse_folding`` step run standalone
(``run_boltzgen_inverse_fold``), via ``--only_inverse_fold``.

BoltzGen's own inverse-folding head (``boltzgen1_ifold.ckpt``), NOT
ProteinMPNN -- see ``run_mpnn`` (a different tool, a different engine) for
that. As with ``run_boltzgen_design``, ``--protocol`` is never passed:
``cli/boltzgen.py``'s ``protocol_configs`` dict never has an
``"inverse_folding"`` key, so a protocol name has no config-level effect on
this step. The ONE place a protocol name would otherwise matter --
``inverse_fold_avoid``'s default ("C" for peptide/nanobody/antibody, ""
otherwise, computed from ``args.protocol`` inside
``BinderDesignPipeline.__init__``) -- is fully subsumed by this tool's own
``avoid_residues`` parameter, which this adapter always passes explicitly
(even when empty), so the result never depends on which protocol BoltzGen's
CLI would have defaulted to.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from Bio.PDB import MMCIFParser
from Bio.SeqUtils import seq1

from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.schema import Manifest
from protein_design_mcp.validation import ToolInputError

_CIF_PARSER = MMCIFParser(QUIET=True)

#: A BoltzGen entity whose ``sequence`` is a LENGTH RANGE ("80..140") is a
#: chain to be generated, not one that exists.
_LENGTH_RANGE = re.compile(r"^\s*\d+\s*\.\.\s*\d+\s*$")


def _refuse_generative_spec(spec: Path) -> None:
    """Refuse a spec that describes a chain with no coordinates yet.

    ``--only_inverse_fold`` passes ``data.cfg.yaml_path=[spec]`` and
    inverse-folds THE STRUCTURE THE SPEC DESCRIBES; it takes no structure
    directory. Handed run_boltzgen_design's own output spec -- whose binder
    entity reads ``sequence: 80..140`` -- there are no coordinates for that
    chain to read, and BoltzGen returned a 94-residue sequence of nothing but
    T and G. Every later step then scored it as a design, and
    ``is_placeholder`` could not see it: two distinct residues is not one.

    Unreadable or unparseable specs pass through -- this is a precondition,
    not a YAML validator, and the engine's own error is the honest one then.
    """
    try:
        import yaml

        data = yaml.safe_load(spec.read_text(errors="replace"))
    except (OSError, ValueError):
        return
    if not isinstance(data, dict):
        return
    for entity in data.get("entities") or []:
        if not isinstance(entity, dict):
            continue
        for kind, body in entity.items():
            if kind == "file" or not isinstance(body, dict):
                continue
            sequence = body.get("sequence")
            if isinstance(sequence, str) and _LENGTH_RANGE.match(sequence):
                raise ToolInputError(
                    f"run_boltzgen_inverse_fold.design_spec = {str(spec)!r} "
                    f"describes chain {body.get('id')!r} as {sequence!r} -- a "
                    "LENGTH RANGE, so that chain has no coordinates yet. This "
                    "is a run_boltzgen_design spec (a design to generate), and "
                    "inverse folding needs a structure that already exists. "
                    "Generate the design first and inverse-fold its output, or "
                    "pass a spec whose every protein entity has a real "
                    "sequence or comes from a file."
                )


def build_args(manifest: Manifest, params: dict[str, Any]) -> list[str]:
    """Translate validated parameters into ``boltzgen run``'s argv.

    ``manifest`` is unused here -- part of every adapter's signature, see
    ``protein_design_mcp.adapters.boltz`` for why.
    """
    del manifest
    _refuse_generative_spec(Path(str(params["design_spec"])))

    return [
        str(params["design_spec"]),
        "--output",
        ".",
        "--steps",
        "inverse_folding",
        "--only_inverse_fold",
        "--inverse_fold_num_sequences",
        str(params["inverse_fold_num_sequences"]),
        "--inverse_fold_checkpoint",
        str(params["inverse_fold_checkpoint"]),
        "--inverse_fold_avoid",
        str(params["avoid_residues"]),
        "--use_kernels",
        str(params["use_kernels"]),
        "--moldir",
        str(params["moldir"]),
        "--num_workers",
        str(params["num_workers"]),
        "--devices",
        "1",
    ]


def _chains_from_cif(path: str) -> list[dict[str, Any]]:
    structure = _CIF_PARSER.get_structure(Path(path).stem, path)
    model = next(structure.get_models())
    chains = []
    for chain in model.get_chains():
        resnames = [residue.resname for residue in chain.get_residues()]
        sequence = seq1("".join(resnames))
        chains.append(
            {"chain_id": chain.id, "sequence": sequence, "length": len(sequence)}
        )
    return chains


def parse_output(manifest: Manifest, run: CompletedRun) -> dict[str, Any]:
    """Read every chain's sequence out of each inverse-folded ``.cif``.

    No chain is singled out as "the design" -- see the manifest's doc for
    why: the caller already knows which chain(s) they marked ``design:`` in
    design_spec, and every chain (including the ones held fixed) is
    reported so a caller can confirm what did and did not change.
    ``manifest`` is unused (see ``build_args``).

    ``inverse_folded_designs`` collects both each design's ``.cif`` AND its
    companion ``.npz`` (see the manifest's ``outputs:`` comment) -- the
    ``.npz`` is real, required output, just not something this payload's
    ``designs`` describes a chain sequence for, so it is skipped here.
    """
    del manifest
    all_paths = run.outputs.get("inverse_folded_designs")
    if not all_paths:
        raise ValueError(
            "run_boltzgen_inverse_fold's declared 'inverse_folded_designs' "
            f"output was not collected -- no design file was found. "
            f"run.outputs was: {run.outputs}"
        )
    if not isinstance(all_paths, list):
        all_paths = [all_paths]
    cif_paths = [path for path in all_paths if Path(path).suffix == ".cif"]

    designs = [
        {"id": Path(path).stem, "chains": _chains_from_cif(path)}
        for path in sorted(cif_paths)
    ]

    return {"designs": designs, "num_designs": len(designs)}
