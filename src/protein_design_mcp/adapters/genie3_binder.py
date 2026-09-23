"""Adapter for Genie 3's binder-design generator (`run_genie3_binder`).

Almost all translation work happens in the wrapper script
(``scripts/engines/genie3_binder.py``), which builds the minimal problem
JSON Genie 3's target-conditioned dataset actually reads plus its own
experiment YAML. This adapter serializes parameters into the wrapper's argv
and reads the collected binder PDBs back, plus the always-written
``interface_conditioning.json`` (see the wrapper), which reports whether
``extended`` interface expansion ran and what it expanded to.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.schema import Manifest


def _bool_str(value: bool) -> str:
    return "true" if value else "false"


def build_args(manifest: Manifest, params: dict[str, Any]) -> list[str]:
    """Translate validated parameters into the wrapper script's argv."""
    del manifest
    return [
        "--target-pdb",
        str(params["target_pdb"]),
        "--hotspot-residues",
        json.dumps(list(params["hotspot_residues"])),
        "--binder-min-length",
        str(params["binder_min_length"]),
        "--binder-max-length",
        str(params["binder_max_length"]),
        "--num-samples",
        str(params["num_samples"]),
        "--model-variant",
        str(params["model_variant"]),
        "--direction-scale",
        str(params["direction_scale"]),
        "--eta",
        str(params["eta"]),
        "--n-sample-step",
        str(params["n_sample_step"]),
        "--noise-scale",
        str(params["noise_scale"]),
        # Pinned false, not read from params: the schema no longer exposes it.
        # Genie 3's side-chain pass is a second stage guarded by
        # `assert config.dataset.source == "unconditional"`, and binder
        # generation runs with source == "target", so true can only raise an
        # AssertionError -- and it does so AFTER the main stage completes,
        # discarding the generation that was just paid for. The flag is still
        # passed because the CLI expects it.
        "--predict-sidechain",
        _bool_str(False),
        "--seed",
        str(params["seed"]),
        "--expand-interface",
        _bool_str(params["expand_interface"]),
        "--interface-cutoff-angstrom",
        str(params["interface_cutoff_angstrom"]),
        "--interface-rsa-threshold",
        str(params["interface_rsa_threshold"]),
        "--interface-abs-sasa-threshold",
        str(params["interface_abs_sasa_threshold"]),
    ]


def _chain_lengths(path: str) -> dict[str, int]:
    """CA-atom count per chain ID, in first-seen order."""
    counts: dict[str, int] = {}
    for line in Path(path).read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")) and line[12:16].strip() == "CA":
            chain_id = line[21]
            counts[chain_id] = counts.get(chain_id, 0) + 1
    return counts


def parse_output(manifest: Manifest, run: CompletedRun) -> dict[str, Any]:
    """Read each generated file's PER-CHAIN lengths back out of its own PDB.

    Genie 3 writes the FULL COMPLEX for a target-conditioned sample -- the
    generated binder chain AND the target chain(s) from target_pdb, all in
    one file (confirmed live: a 20/20-residue binder request against a
    5-residue target produced a file with a 20-residue chain A and a
    5-residue chain B). `create_np_features_from_target_config` always
    concatenates the binder's features BEFORE the target's (read from
    source), and Genie 3's own PDB writer assigns chain IDs in that same
    concatenation order -- so chain A is the generated binder and every
    other chain is target, confirmed by the live example above. Reporting
    only a combined atom count would have silently conflated the two.
    """
    del manifest
    paths = run.outputs.get("binders")
    if not paths:
        raise ValueError(
            "run_genie3_binder's declared 'binders' output was not "
            f"collected -- no PDB was found. run.outputs was: {run.outputs}"
        )
    if not isinstance(paths, list):
        paths = [paths]

    binders = []
    for path in sorted(paths):
        chain_lengths = _chain_lengths(path)
        if not chain_lengths:
            raise ValueError(f"{path} has no CA atoms in any chain")
        # dict preserves first-insertion order in Python 3.7+, and lines are
        # read in file order, so the first key is chain A / the binder.
        binder_chain_id = next(iter(chain_lengths))
        binders.append(
            {
                "id": Path(path).stem,
                "length": chain_lengths[binder_chain_id],
                "binder_chain_id": binder_chain_id,
                "chain_lengths": chain_lengths,
            }
        )

    result: dict[str, Any] = {"binders": binders, "num_binders": len(binders)}

    conditioning_path = run.outputs.get("interface_conditioning")
    if conditioning_path:
        conditioning = json.loads(Path(conditioning_path).read_text())
        result["cond_strategy"] = conditioning["cond_strategy"]
        result["extended_interface_residues"] = conditioning["extended_interface_residues"]

    return result
