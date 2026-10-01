"""Adapter for BoltzGen's ``analysis`` pipeline step (``run_boltzgen_analyze``).

Reassembles up to THREE prior tool calls' outputs
(``run_boltzgen_design``/``run_boltzgen_inverse_fold``, ``run_boltzgen_fold``,
optionally ``run_boltzgen_fold` with `with_target: false``) into the specific directory tree
``boltzgen.task.analyze.analyze.Analyze`` expects (``design_dir`` itself for
the original files, ``design_dir/refold_cif``/``fold_out_npz`` for the
fold outputs, ``design_dir/refold_design_cif``/``fold_out_design_npz`` for
the optional design-fold outputs -- all hardcoded relative to ONE
``design_dir`` in ``analyze.py``'s own ``init_datasets``, not independently
configurable). ``manifest.engine.stage``/``stage_subdir`` do the actual
file placement; this adapter derives ``design_dir``'s path from the
already-staged ``generated_files`` the same way ``run_boltzgen_fold``'s
adapter does.

As with ``run_boltzgen_filter``, every one of ``Analyze.__init__``'s
parameters this tool exposes is always pushed through
``--config analysis <key>=<value>`` (not a top-level CLI flag) so a
protocol preset can never silently override it -- ``--protocol`` is
consequently never passed.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.schema import Manifest
from protein_design_mcp.validation import ToolInputError

_BOOLEAN_KEYS = (
    "affinity_metrics",
    "backbone_fold_metrics",
    "allatom_fold_metrics",
    "noncovalents_original",
    "noncovalents_refolded",
    "delta_sasa_original",
    "delta_sasa_refolded",
    "largest_hydrophobic",
    "largest_hydrophobic_refolded",
    "run_clustering",
    "liability_analysis",
    "designfolding_metrics",
    "use_design_mask_for_target",
    "compute_lddts",
)


def _bool_str(value: bool) -> str:
    return "true" if value else "false"


# BoltzGen data.const.tokens: the on-disk res_type one-hot vocabulary.
_TOKENS = ("<pad> - ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET "
           "PHE PRO SER THR TRP TYR VAL UNK A G C U N DA DG DC DT DN").split()
_TOKEN_IDS = {name: i for i, name in enumerate(_TOKENS)}
_POLYMER_TYPES = {"polypeptide(L)": 0, "polydeoxyribonucleotide": 1,
                  "polyribonucleotide": 2}
_COMPLEX_HINT = (
    "Required refold_structures/refold_metrics must match the complete original complex "
    "(all chains, in order, with identical sequences). Design-only folds belong in optional "
    "design_refold_structures/design_refold_metrics with designfolding_metrics=true; "
    "they do not replace the required complete-complex inputs. Supply matching outputs "
    "for the same generated designs; no inputs have been changed."
)


def _index_files(paths: list[str], field: str, suffix: str) -> dict[str, Path]:
    result = {}
    for raw in paths:
        path = Path(raw)
        if path.suffix != suffix:
            raise ToolInputError(f"{field}: expected {suffix} files, got {path.name!r}.")
        if path.stem in result:
            raise ToolInputError(f"{field}: duplicate design IDs: {path.stem!r}.")
        result[path.stem] = path
    return result


def _chains(path: Path, field: str) -> list[tuple[str, str, tuple[str, ...]]]:
    """Read full declared sequences, including residues without coordinates.

    Entity IDs are local to a CIF; label chain IDs and their ordered sequences
    carry the cross-file identity. Nonpolymer identities are retained too.
    """
    try:
        data = MMCIF2Dict(str(path))
        types = dict(zip(data.get("_entity_poly.entity_id", []),
                         data.get("_entity_poly.type", []), strict=True))
        sequences: dict[str, list[str]] = {}
        for entity, residue in zip(data.get("_entity_poly_seq.entity_id", []),
                                   data.get("_entity_poly_seq.mon_id", []), strict=True):
            sequences.setdefault(entity, []).append(residue)
        for entity, residue in zip(data.get("_pdbx_entity_nonpoly.entity_id", []),
                                   data.get("_pdbx_entity_nonpoly.comp_id", []), strict=True):
            sequences[entity] = [residue]
        chains = [(chain, types.get(entity, "nonpolymer"), tuple(sequences[entity]))
                  for chain, entity in zip(data["_struct_asym.id"],
                                           data["_struct_asym.entity_id"], strict=True)]
        if not chains or any(not seq for _, _, seq in chains):
            raise ValueError("no complete chain sequences")
        if len({chain for chain, _, _ in chains}) != len(chains):
            raise ValueError("duplicate chain IDs")
        return chains
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ToolInputError(f"{field}: cannot read complete chain sequences from {path}: {exc}") from exc


def _check_metrics(path: Path, field: str, chains: list, expected_mol_type=None) -> None:
    """Check tensor dimensions and canonical polymer identities without pickle.

    Noncanonical residues can be atom-tokenized by the engine, so their exact
    tensor expansion is not reconstructed here. Their full CIF identities and
    the complete generated mol_type vector are still checked.
    """
    try:
        with np.load(path, allow_pickle=False) as data:
            mol_type = data["mol_type"]
            res_type = data["res_type"]
        if mol_type.ndim == 2 and mol_type.shape[0] == 1:
            mol_type = mol_type[0]
        if res_type.ndim == 3 and res_type.shape[0] == 1:
            res_type = res_type[0]
        if mol_type.ndim != 1 or res_type.shape != (len(mol_type), len(_TOKENS)):
            raise ValueError("invalid mol_type/res_type tensor dimensions")
        if not np.all((res_type == 0) | (res_type == 1)) or not np.all(res_type.sum(axis=1) == 1):
            raise ValueError("res_type must contain one-hot residue identities")
        if expected_mol_type is not None and not np.array_equal(mol_type, expected_mol_type):
            raise ValueError("token count or molecule types differ from generated_files")
        actual = res_type.argmax(axis=1)
        for poly_type, kind in _POLYMER_TYPES.items():
            residues = [res for _, chain_type, seq in chains if chain_type == poly_type for res in seq]
            # A noncanonical residue may expand into several tokens. Keep the
            # CIF comparison authoritative for that polymer type.
            if any(res not in _TOKEN_IDS for res in residues):
                continue
            expected = np.array([_TOKEN_IDS[res] for res in residues], dtype=int)
            if not np.array_equal(actual[mol_type == kind], expected):
                raise ValueError(f"res_type sequences differ from the {poly_type} CIF chains")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ToolInputError(f"{field}: incompatible metrics for design {path.stem!r}: {exc}. {_COMPLEX_HINT}") from exc


def _validate_handoff(params: dict[str, Any]) -> None:
    optional = (bool(params.get("design_refold_structures")), bool(params.get("design_refold_metrics")))
    if optional[0] != optional[1] or (params["designfolding_metrics"] and not all(optional)):
        raise ToolInputError("Supply design_refold_structures and design_refold_metrics together; "
                             "both are required when designfolding_metrics=true.")
    generated = params["generated_files"]
    unexpected = [p for p in generated if Path(p).suffix not in {".cif", ".npz"}]
    if unexpected:
        raise ToolInputError("generated_files must contain the original .cif and .npz outputs only.")
    groups = {
        "generated CIF": _index_files([p for p in generated if Path(p).suffix == ".cif"], "generated_files", ".cif"),
        "generated NPZ": _index_files([p for p in generated if Path(p).suffix == ".npz"], "generated_files", ".npz"),
        "refold_structures": _index_files(params["refold_structures"], "refold_structures", ".cif"),
        "refold_metrics": _index_files(params["refold_metrics"], "refold_metrics", ".npz"),
    }
    if all(optional):
        for field, suffix in [("design_refold_structures", ".cif"), ("design_refold_metrics", ".npz")]:
            groups[field] = _index_files(params[field], field, suffix)
    ids = set(groups["generated CIF"])
    for field, indexed in groups.items():
        if not ids or set(indexed) != ids:
            raise ToolInputError(f"{field}: design IDs must match generated_files exactly; "
                                 f"missing={sorted(ids - set(indexed))}, extra={sorted(set(indexed) - ids)}.")
    for design_id in sorted(ids):
        original = _chains(groups["generated CIF"][design_id], "generated_files")
        refold = _chains(groups["refold_structures"][design_id], "refold_structures")
        if original != refold:
            original_lengths = [(chain, len(seq)) for chain, _, seq in original]
            refold_lengths = [(chain, len(seq)) for chain, _, seq in refold]
            raise ToolInputError(f"refold_structures: design {design_id!r} chain sequences differ "
                                 f"(generated chain lengths={original_lengths}, refold={refold_lengths}). {_COMPLEX_HINT}")
        try:
            with np.load(groups["generated NPZ"][design_id], allow_pickle=False) as data:
                mol_type = data["mol_type"]
            if mol_type.ndim != 1:
                raise ValueError("mol_type must be a one-dimensional token vector")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ToolInputError(f"generated_files: cannot read token metadata for {design_id!r}: {exc}") from exc
        _check_metrics(groups["refold_metrics"][design_id], "refold_metrics", original, mol_type)
        if all(optional):
            design_chains = _chains(groups["design_refold_structures"][design_id], "design_refold_structures")
            _check_metrics(groups["design_refold_metrics"][design_id], "design_refold_metrics", design_chains)


def build_args(manifest: Manifest, params: dict[str, Any]) -> list[str]:
    """Translate validated (and already-staged) parameters into
    ``boltzgen run``'s argv. ``manifest`` is unused -- part of every
    adapter's signature, see ``protein_design_mcp.adapters.boltz``.
    """
    del manifest
    _validate_handoff(params)
    generated_files = params["generated_files"]
    design_dir = str(Path(generated_files[0]).parent)

    config_overrides = [
        f"design_dir={design_dir}",
        f"data.cfg.num_workers={params['num_workers']}",
        f"num_processes={params['num_processes']}",
        # foldseek_binary is a plain host path, not a "huggingface:..."
        # artifact reference, so (unlike moldir/checkpoint) it needs no
        # top-level-flag resolution and can go through --config directly.
        # Passed unconditionally: Analyze only USES it when run_clustering
        # is true, so passing it when clustering is off is harmless.
        f"foldseek_binary={params['foldseek_binary']}",
    ]
    for key in _BOOLEAN_KEYS:
        config_overrides.append(f"{key}={_bool_str(params[key])}")
    config_overrides.append(f"liability_modality={params['liability_modality']}")
    config_overrides.append(f"liability_peptide_type={params['liability_peptide_type']}")

    return [
        str(params["design_spec"]),
        "--output",
        ".",
        "--steps",
        "analysis",
        # moldir MUST go through the top-level flag, not --config -- see
        # run_boltzgen_fold's identical comment (BoltzGen's own CLI resolves
        # a "huggingface:repo:file" artifact reference to a real local path
        # BEFORE embedding it in every step's fixed args; a --config
        # override bypasses that and hands the engine the raw unresolved
        # string, which fails hard).
        "--moldir",
        str(params["moldir"]),
        "--config",
        "analysis",
        *config_overrides,
    ]


def parse_output(manifest: Manifest, run: CompletedRun) -> dict[str, Any]:
    """Count the rows analyze wrote to its own aggregate metrics CSV.
    ``manifest`` is unused (see ``build_args``).
    """
    del manifest
    csv_path = run.outputs.get("aggregate_metrics_csv")
    if not csv_path:
        raise ValueError(
            "run_boltzgen_analyze's declared 'aggregate_metrics_csv' output "
            f"was not collected -- no metrics file was found. run.outputs "
            f"was: {run.outputs}"
        )
    if isinstance(csv_path, list):
        csv_path = csv_path[0]

    with Path(csv_path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    return {"num_designs_analyzed": len(rows)}
