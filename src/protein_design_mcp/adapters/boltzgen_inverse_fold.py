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

from pathlib import Path
from typing import Any

from Bio.PDB import MMCIFParser
from Bio.SeqUtils import seq1

from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.schema import Manifest

_CIF_PARSER = MMCIFParser(QUIET=True)


def build_args(manifest: Manifest, params: dict[str, Any]) -> list[str]:
    """Translate validated parameters into ``boltzgen run``'s argv.

    ``manifest`` is unused here -- part of every adapter's signature, see
    ``protein_design_mcp.adapters.boltz`` for why.
    """
    del manifest

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
    """
    del manifest
    cif_paths = run.outputs.get("inverse_folded_designs")
    if not cif_paths:
        raise ValueError(
            "run_boltzgen_inverse_fold's declared 'inverse_folded_designs' "
            f"output was not collected -- no design file was found. "
            f"run.outputs was: {run.outputs}"
        )
    if not isinstance(cif_paths, list):
        cif_paths = [cif_paths]

    designs = [
        {"id": Path(path).stem, "chains": _chains_from_cif(path)}
        for path in sorted(cif_paths)
    ]

    return {"designs": designs, "num_designs": len(designs)}
