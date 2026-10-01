"""Adapter for BoltzGen's ``folding`` pipeline step (``run_boltzgen_fold``).

BoltzGen's own ``folding`` step reads each design's ``.cif``+``.npz`` and
writes its own refold output (``refold_cif/``, ``fold_out_npz/``) BESIDE
those inputs, into the same ``design_dir`` -- the ipSAE "writes beside its
input" pattern, generalised to a whole directory of paired files (see
``manifest.schema.EngineSpec.stage``, ``protein_design_mcp.staging``). The
manifest's ``engine.stage: ["generated_files"]`` copies the caller's whole
file list into one shared ``<workdir>/generated_files/`` directory before
this engine runs; this adapter derives that directory's path from the
already-staged ``params["generated_files"]`` (every item shares the same
parent by construction) and points BoltzGen's own ``data.design_dir`` (and
``output``, harmlessly) at it.

As with ``run_boltzgen_design``, ``--protocol`` is never passed:
``cli/boltzgen.py``'s ``protocol_configs`` dict's only ``"folding"`` entry
belongs to ``protein-redesign`` (a mode this tool does not support), so a
protocol name has no effect on the path this wrapper actually takes --
verified by diffing ``boltzgen configure --steps folding``'s resolved
``fold.yaml`` across all six protocols with everything else held fixed.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np
from Bio.PDB import MMCIFParser, PDBParser
from Bio.SeqUtils import seq1
from protein_design_mcp.validation import ToolInputError

from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.schema import Manifest
from protein_design_mcp.staging import validate_source_file

# boltzgen.data.const.eval_keys_confidence, minus the ones that duplicate
# ligand_iptm and minus the raw tensors (coords, res_type, ...) eval_keys
# also carries -- those aren't per-sample scalars and aren't useful here.
_METRIC_KEYS = (
    "design_ptm",
    "design_iptm",
    "design_iiptm",
    "design_to_target_iptm",
    "design_residue_iptm",
    "target_ptm",
    "ptm",
    "iptm",
    "protein_iptm",
    "ligand_iptm",
    "min_interaction_pae",
    "min_design_to_target_pae",
    "interaction_pae",
    "complex_plddt",
    "complex_iplddt",
    "complex_pde",
    "complex_ipde",
)


def build_args(manifest: Manifest, params: dict[str, Any]) -> list[str]:
    """Translate validated (and already-staged) parameters into
    ``boltzgen run``'s argv. ``manifest`` is unused -- part of every
    adapter's signature, see ``protein_design_mcp.adapters.boltz``.
    """
    del manifest
    structure = params.get("structure")
    native = params.get("generated_files") or params.get("design_spec")
    if bool(structure) == bool(native):
        raise ToolInputError("Supply exactly one input mode: structure + design_chains + designed_sequences, or design_spec + generated_files.")
    if structure:
        chains = params.get("design_chains") or []
        sequences = params.get("designed_sequences") or {}
        if not chains or len(chains) != len(set(chains)) or set(sequences) != set(chains):
            raise ToolInputError("designed_sequences must map exactly the unique design_chains to their designed amino acid sequences.")
        parser = PDBParser(QUIET=True) if str(structure).lower().endswith((".pdb", ".pdb.gz")) else MMCIFParser(QUIET=True)
        try:
            with (gzip.open(structure, "rt") if str(structure).endswith(".gz") else open(structure)) as handle:
                models = list(parser.get_structure("input", handle).get_models())
            if len(models) != 1:
                raise ValueError("exactly one structure model is required")
            model = models[0]
        except Exception as exc:
            raise ToolInputError(f"Cannot read folding structure: {exc}") from exc
        if any(r.id[0] != " " or seq1(r.resname) == "X" for r in model.get_residues()):
            raise ToolInputError("External folding supports canonical protein chains only; use native mode for ligands or modified residues.")
        for chain in chains:
            sequence = sequences[chain]
            if not isinstance(sequence, str) or not sequence or set(sequence) - set("ACDEFGHIKLMNPQRSTVWY"):
                raise ToolInputError(f"designed_sequences[{chain!r}] must contain canonical uppercase amino acids.")
            if chain not in model:
                raise ToolInputError(f"design_chains names missing chain {chain!r}; available: {[c.id for c in model]}")
            residues = [r for r in model[chain] if r.id[0] == " "]
            if len(residues) != len(sequence):
                raise ToolInputError(f"Sequence length for {chain!r} is {len(sequence)}, but structure contains {len(residues)} residues.")
            if any(not all(atom in r for atom in ("N", "CA", "C", "O")) for r in residues):
                raise ToolInputError(f"Chain {chain!r} requires a complete N/CA/C/O backbone; rebuild a CA-only trace before folding.")
        source_args = ["--structure", str(structure), "--design-chains", json.dumps(chains), "--designed-sequences", json.dumps(sequences)]
        design_dir = "generated_files"
    else:
        if params.get("design_chains") or params.get("designed_sequences"):
            raise ToolInputError("design_chains and designed_sequences apply only with structure.")
        generated_files = params.get("generated_files")
        if not generated_files or not params.get("design_spec"):
            raise ToolInputError("Native mode requires both design_spec and generated_files containing complete .cif/.npz pairs.")
        pairs = {}
        for value in generated_files:
            path = Path(value)
            if path.suffix not in (".cif", ".npz") or path.suffix in pairs.setdefault(path.stem, set()):
                raise ToolInputError("generated_files must contain unique .cif/.npz pairs.")
            pairs[path.stem].add(path.suffix)
        if any(suffixes != {".cif", ".npz"} for suffixes in pairs.values()) or len({Path(p).parent for p in generated_files}) != 1:
            raise ToolInputError("generated_files must contain complete matching .cif/.npz pairs in one staged directory.")
        design_dir = str(Path(generated_files[0]).parent)
        source_args = [str(validate_source_file(params["design_spec"], "design_spec"))]
    # ONE name, used for both --steps and --config. `--config <step>` binds
    # the overrides to that step alone, and `folding` stays a VALID step name
    # even when it is not the step being run -- so hardcoding it here sent
    # every override to a step that never ran, in silence, and
    # `design_folding` fell back to BoltzGen's own relative
    # `intermediate_designs_inverse_folded` default, which does not exist.
    step = "folding" if params["with_target"] else "design_folding"

    return source_args + [
        "--passthrough",
        "--output",
        ".",
        "--steps",
        # `folding` refolds in complex, `design_folding` refolds the design
        # alone. One engine task with a mode, which is why these were merged
        # from two tools that differed by nothing else.
        step,
        "--devices",
        "1",
        # checkpoint/moldir/use_kernels MUST go through the top-level flags,
        # not --config: BoltzGen's own CLI resolves a "huggingface:repo:file"
        # artifact reference (via get_artifact_path/hf_hub_download) and
        # "auto"/"true"/"false" use_kernels (via a device-capability check)
        # BEFORE embedding the resolved LOCAL path / actual bool into every
        # step's fixed args -- a --config override bypasses that resolution
        # entirely and hands the engine the raw unresolved string, which
        # fails hard (verified live: "Invalid moldir. Expected directory or
        # zip file: huggingface:boltzgen/inference-data:mols.zip").
        "--folding_checkpoint",
        str(params["folding_checkpoint"]),
        "--moldir",
        str(params["moldir"]),
        "--use_kernels",
        str(params["use_kernels"]),
        "--config",
        step,
        f"data.design_dir={design_dir}",
        f"output={design_dir}",
        f"data.cfg.num_workers={params['num_workers']}",
        "trainer.devices=1",
        f"recycling_steps={params['recycling_steps']}",
        f"sampling_steps={params['sampling_steps']}",
        f"diffusion_samples={params['diffusion_samples']}",
    ]


def _best_sample_metrics(path: str) -> dict[str, Any]:
    """Read one design's fold_out_npz file and return the metrics of its
    highest-confidence sample (0.8*iptm + 0.2*ptm), matching
    ``FoldingWriter.write_on_batch_end``'s own selection -- the sample this
    same tool's ``refolded_structures`` output actually wrote.
    """
    data = np.load(path)
    iptm = data["iptm"]
    ptm = data["ptm"]
    confidence = 0.8 * iptm + 0.2 * ptm
    best_idx = int(np.argmax(confidence))

    metrics: dict[str, Any] = {}
    for key in _METRIC_KEYS:
        if key in data.files:
            metrics[key] = float(data[key][best_idx])
    metrics["num_samples"] = int(len(iptm))
    metrics["best_sample_index"] = best_idx
    return metrics


def parse_output(manifest: Manifest, run: CompletedRun) -> dict[str, Any]:
    """Read the best-sample metrics out of each design's fold_out_npz.
    ``manifest`` is unused (see ``build_args``).
    """
    del manifest
    metrics_paths = run.outputs.get("refold_metrics")
    if not metrics_paths:
        raise ValueError(
            "run_boltzgen_fold's declared 'refold_metrics' output was not "
            f"collected -- no metrics file was found. run.outputs was: "
            f"{run.outputs}"
        )
    if not isinstance(metrics_paths, list):
        metrics_paths = [metrics_paths]

    refolds = [
        {"id": Path(path).stem, **_best_sample_metrics(path)}
        for path in sorted(metrics_paths)
    ]

    return {"refolds": refolds, "num_refolds": len(refolds)}
