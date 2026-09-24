"""Reconstruct N, C and O from a CA trace, and make the result designable.

Runs in the `mpnn` env, which already carries PULCHRA's three dependencies
(biopython, numpy, scipy) at versions LigandMPNN is pinned against -- adding
`pulchra` there moves none of them (verified in the image, 2026-09-24).

Two steps, in this order:

1. PULCHRA rebuilds the backbone. Side-chain rebuilding is deliberately OFF:
   the chain is about to be redesigned, so invented side chains would be
   thrown away, and ProteinMPNN derives a virtual CB from N/CA/C anyway.
2. Placeholder residue names are rewritten. PULCHRA preserves `UNK`, and
   ProDy -- which LigandMPNN parses with -- matches `protein` by residue
   NAME, so a UNK chain still reads as "no protein here" however complete
   its backbone is.

Chains that already have a full backbone pass through PULCHRA too; it leaves
measured atoms alone, and the adapter refuses a structure where EVERY chain
is already complete so this is never run over a real structure wholesale.
"""

from __future__ import annotations

import argparse
from pathlib import Path

OUTPUT_NAME = "rebuilt.pdb"

# Deliberately self-contained. This runs in the `mpnn` environment, which has
# PULCHRA and LigandMPNN but not the MCP server stack -- importing
# `protein_design_mcp.adapters.rebuild_backbone` pulls in the package
# `__init__`, hence `server`, hence `mcp`, and the engine died on the import
# before PULCHRA ran. Every other engine wrapper here is standalone for the
# same reason. `tests/test_adapter_rebuild_backbone.py` asserts this copy and
# the adapter's agree, so the duplication cannot drift.
PLACEHOLDER_RESNAMES = {"UNK", "GLX", "XAA"}
_STAND_IN = "GLY"


def normalise_designable_residues(lines: list[str]) -> list[str]:
    """Rename placeholder residues so a ProDy-based reader sees protein."""
    out: list[str] = []
    for line in lines:
        if line.startswith(("ATOM", "HETATM")) and line[17:20].strip() in PLACEHOLDER_RESNAMES:
            line = line[:17] + _STAND_IN.ljust(3) + line[20:]
        out.append(line)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("structure")
    parser.add_argument("--out-dir", default=".")
    args = parser.parse_args()

    from pulchra import Pulchra

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rebuilt = out_dir / OUTPUT_NAME

    Pulchra(
        rebuild_backbone=True,
        # See the module docstring: invented side chains are discarded by the
        # very next tool, and cost time to place.
        rebuild_sidechains=False,
        verbose=False,
    ).reconstruct(str(Path(args.structure).resolve()), str(rebuilt))

    lines = rebuilt.read_text(errors="replace").splitlines()
    rebuilt.write_text("\n".join(normalise_designable_residues(lines)) + "\n")
    print(f"rebuilt backbone -> {rebuilt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
