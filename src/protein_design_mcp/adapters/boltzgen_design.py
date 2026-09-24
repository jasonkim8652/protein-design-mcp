"""Adapter for BoltzGen's ``design`` pipeline step (``run_boltzgen_design``).

Wraps ``boltzgen run <design_spec> --steps design`` directly. Unlike
``run_boltzgen_filter``, there is no protocol-preset-ordering hazard to route
around here: ``cli/boltzgen.py``'s ``protocol_configs`` dict never has a
``"design"`` key in any of its six protocols, and this was confirmed live by
diffing ``boltzgen configure --steps design``'s resolved ``design.yaml``
across all six -- byte-identical except the ``--output`` path itself. So
``--protocol`` is simply never passed (BoltzGen's own CLI default,
``protein-anything``, applies and is a genuine no-op for this step).
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

    # The spec is BUILT by the wrapper from the parameters below, or copied
    # from `design_spec` when a hand-written one is given. Everything after
    # --passthrough is what the wrapper hands to `boltzgen run`.
    args: list[str] = [
        "--target-structure", str(params["target_structure"]),
        "--target-chains", ",".join(str(c) for c in params["target_chains"]),
        "--binder-length-min", str(params["binder_length_min"]),
        "--binder-length-max", str(params["binder_length_max"]),
        "--binder-chain-id", str(params["binder_chain_id"]),
    ]
    args += [
        "--passthrough",
        "--output",
        ".",
        "--steps",
        "design",
        "--num_designs",
        str(params["num_designs"]),
        "--design_checkpoints",
        *[str(c) for c in params["design_checkpoints"]],
        "--use_kernels",
        str(params["use_kernels"]),
        "--moldir",
        str(params["moldir"]),
        "--num_workers",
        str(params["num_workers"]),
        "--devices",
        "1",
    ]

    diffusion_batch_size = params.get("diffusion_batch_size")
    if diffusion_batch_size is not None:
        args.extend(["--diffusion_batch_size", str(diffusion_batch_size)])

    step_scale = params.get("step_scale")
    if step_scale is not None:
        args.extend(["--step_scale", str(step_scale)])

    noise_scale = params.get("noise_scale")
    if noise_scale is not None:
        args.extend(["--noise_scale", str(noise_scale)])

    return args


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
    """Read every chain's sequence out of each generated ``.cif``.

    No chain is singled out as "the design" -- see the manifest's doc for
    why: this step alone has no design-mask bookkeeping this adapter can
    read back reliably, so every chain (target included, unchanged) is
    reported, and the caller -- who wrote design_spec -- already knows which
    one(s) they marked for design. ``manifest`` is unused (see
    ``build_args``).

    ``generated_designs`` collects both each design's ``.cif`` AND its
    companion ``.npz`` (see the manifest's ``outputs:`` comment) -- the
    ``.npz`` is real, required output (BoltzGen's own downstream fold/
    analyze steps need it), just not something THIS payload's ``designs``
    describes a chain sequence for, so it is skipped here rather than
    fed to the mmCIF parser.
    """
    del manifest
    all_paths = run.outputs.get("generated_designs")
    if not all_paths:
        raise ValueError(
            "run_boltzgen_design's declared 'generated_designs' output was "
            f"not collected -- no design file was found. run.outputs was: "
            f"{run.outputs}"
        )
    if not isinstance(all_paths, list):
        all_paths = [all_paths]
    cif_paths = [path for path in all_paths if Path(path).suffix == ".cif"]

    designs = [
        {"id": Path(path).stem, "chains": _chains_from_cif(path)}
        for path in sorted(cif_paths)
    ]

    return {"designs": designs, "num_designs": len(designs)}
