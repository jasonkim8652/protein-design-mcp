"""Adapter for run_rebuild_backbone (PULCHRA).

Turns a CA trace into a structure a sequence designer can read. See
``tests/test_adapter_rebuild_backbone.py`` for the two conditions this
exists to satisfy and how each was established live.

The engine is PULCHRA (PyPI ``pulchra`` 1.0.1, the published CA-trace
reconstruction method), driven by ``scripts/engines/rebuild_backbone.py``.
Nothing here computes geometry; the wrapper calls PULCHRA and then renames
the residues PULCHRA leaves as ``UNK``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.schema import Manifest
from protein_design_mcp.validation import ToolInputError

#: ProDy's ``protein`` selection is matched by residue name, and ``UNK`` is
#: not in it. A chain of UNK therefore reads as "no protein here" to every
#: ProDy-based consumer, ``run_mpnn`` included.
PLACEHOLDER_RESNAMES = {"UNK", "GLX", "XAA"}

#: What ProteinMPNN needs per residue to build its frame.
BACKBONE_ATOMS = ("N", "CA", "C", "O")

_STAND_IN = "GLY"


def _atom_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.startswith(("ATOM", "HETATM"))]


def normalise_designable_residues(lines: list[str]) -> list[str]:
    """Rename placeholder residues so a ProDy-based reader sees protein.

    Only the residue-name columns (18-20) change, and only for a placeholder
    name -- a real residue is the target's, and rewriting it would change
    what ``run_mpnn`` holds fixed.
    """
    out: list[str] = []
    for line in lines:
        if line.startswith(("ATOM", "HETATM")) and line[17:20].strip() in PLACEHOLDER_RESNAMES:
            line = line[:17] + _STAND_IN.ljust(3) + line[20:]
        out.append(line)
    return out


def _chains(lines: list[str]) -> dict[str, set[str]]:
    chains: dict[str, set[str]] = {}
    for line in lines:
        if len(line) > 21:
            chains.setdefault(line[21], set()).add(line[12:16].strip())
    return chains


def build_args(manifest: Manifest, params: dict[str, Any]) -> list[str]:
    """PULCHRA's argv, after checking there is something to rebuild."""
    del manifest
    structure = Path(str(params["structure"]))
    try:
        text = structure.read_text(errors="replace")
    except OSError as exc:
        raise ToolInputError(
            f"run_rebuild_backbone.structure = {str(structure)!r} could not be "
            f"read ({exc.strerror})."
        ) from None

    lines = _atom_lines(text)
    if not lines:
        raise ToolInputError(
            f"run_rebuild_backbone.structure = {str(structure)!r} has no ATOM "
            "records."
        )
    chains = _chains(lines)
    if all(set(BACKBONE_ATOMS) <= atoms for atoms in chains.values()):
        raise ToolInputError(
            f"run_rebuild_backbone.structure = {str(structure)!r} already has a "
            f"complete backbone in every chain ({', '.join(sorted(chains))}). "
            "This tool reconstructs N, C and O from a CA trace; running it over "
            "measured atoms would replace them with approximations. Pass the "
            "structure straight to the sequence designer instead."
        )
    return [str(structure)]


def parse_output(manifest: Manifest, run: CompletedRun) -> dict[str, Any]:
    """Report what came back, including whether it is actually designable."""
    del manifest
    produced = (run.outputs or {}).get("structure_pdb")
    if not produced:
        raise ToolInputError(
            "run_rebuild_backbone produced no structure; PULCHRA wrote nothing "
            "to rebuild from."
        )
    # A list only when the manifest says `multiple: true`. This one declares a
    # single file, so the dispatcher hands over a plain string -- taking [0]
    # of that is the character "/", and the adapter failed on `Is a directory`.
    first = produced[0] if isinstance(produced, (list, tuple)) else produced
    path = Path(str(first))
    lines = _atom_lines(path.read_text(errors="replace"))
    chains = _chains(lines)
    residues = {(line[21], line[22:26]) for line in lines if len(line) > 26}
    return {
        "structure_pdb": str(path),
        "chains": sorted(chains),
        "residues_rebuilt": len(residues),
        # Stated rather than assumed: a partial backbone here becomes an
        # unreadable crash one tool later.
        "backbone_complete": all(
            set(BACKBONE_ATOMS) <= atoms for atoms in chains.values()),
    }
