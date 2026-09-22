"""Adapter for dauparas/LigandMPNN (PyPI `ligandmpnn`).

One codebase serves ProteinMPNN, SolubleMPNN and LigandMPNN; `model_type`
selects the checkpoint family.

The engine writes FASTA whose FIRST record is the input sequence rather than a
design. Returning it as a design is a real bug this server previously shipped —
a caller trusting ``designs[0]`` got back what it put in. Verified against the
upstream ``ligandmpnn/run.py`` (PyPI ``ligandmpnn==0.1.2``, entry_points.txt
maps its console scripts to ``ligandmpnn.run:main`` and the module itself has
an ``if __name__ == "__main__": main()`` guard, so
``python -m ligandmpnn.run`` is a genuine, working invocation): the native
record's header has no ``id=`` field at all (it carries ``num_res=``,
``model_path=``, etc. instead), while every design record does, e.g.
``id=1, T=0.1, seed=37, overall_confidence=0.5310, ligand_confidence=0.0,
seq_rec=0.8824``. Dropping any record lacking ``id=`` is therefore exactly
right, not merely defensive.

SETTLED LIVE (Task 7, protein-design-mcp:envs image, CPU, real ``ligandmpnn``
weights bundled in the PyPI wheel): the engine does NOT echo the FASTA to
stdout. Reading ``ligandmpnn/run.py`` end to end confirms every ``print(...)``
call is a progress/status message (``"CUDA not available... using CPU"``,
``"Designing protein from this path: ..."``, etc.) — the sequence records
(both the native/input record and every ``id=``-bearing design record) are
written ONLY via ``with open(output_fasta, "w") as f: f.write(...)`` to
``<out_folder>/seqs/<name>.fa``. A live run confirmed ``run.stdout`` contains
no FASTA content at all, so the previous stdout-parsing adapter could never
have worked against the real engine. This adapter instead reads the
manifest's declared ``designs_fasta`` output — a LIST because the manifest
sets ``multiple: true`` — via ``run.outputs["designs_fasta"]``, which the
dispatcher populates by collecting files matching ``seqs/*.fa`` out of the
engine's scratch working directory (confirmed live: ``--out_folder .`` with
the dispatcher's per-run ``cwd`` does put the file where that pattern finds
it). The input-record-dropping rule (no ``id=`` means it's the native
sequence, not a design) is unchanged.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.schema import Manifest

CHECKPOINT_FOR = {
    "protein": "protein_mpnn",
    "soluble": "soluble_mpnn",
    "ligand": "ligand_mpnn",
}

_ID_RE = re.compile(r"\bid=(\d+)")
_CONF_RE = re.compile(r"\boverall_confidence=([\d.]+)")


def build_args(manifest: Manifest, params: dict[str, Any]) -> list[str]:
    """Translate validated parameters into the engine's argv."""
    del manifest
    return [
        "--model_type",
        CHECKPOINT_FOR[str(params["model_type"])],
        "--pdb_path",
        str(params["backbone_pdb"]),
        "--out_folder",
        ".",
        "--batch_size",
        str(params["num_sequences"]),
        "--temperature",
        str(params["sampling_temp"]),
        "--seed",
        str(params["seed"]),
    ]


def _records(text: str) -> list[tuple[str, str]]:
    """Split FASTA into (header, sequence) pairs, preserving order."""
    records: list[tuple[str, str]] = []
    header: str | None = None
    chunks: list[str] = []
    for line in text.splitlines():
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(chunks)))
            header, chunks = line[1:], []
        elif header is not None and line.strip():
            chunks.append(line.strip())
    if header is not None:
        records.append((header, "".join(chunks)))
    return records


def parse_output(manifest: Manifest, run: CompletedRun) -> dict[str, Any]:
    """Extract designs from the collected FASTA file(s), dropping the input
    record the engine writes first.

    ``run.outputs["designs_fasta"]`` is always a list: the manifest's
    ``designs_fasta`` output spec sets ``multiple: true``, and
    ``dispatch.env`` / ``results.collect_outputs`` only ever return a bare
    string for a single-valued (``multiple: false``) spec.
    """
    del manifest
    fasta_paths = run.outputs.get("designs_fasta")
    if not fasta_paths:
        raise ValueError(
            "MPNN's declared 'designs_fasta' output was not collected — no "
            f"seqs/*.fa file was found. run.outputs was: {run.outputs}"
        )

    designs = []
    for fasta_path in fasta_paths:
        text = Path(fasta_path).read_text()
        for header, sequence in _records(text):
            id_match = _ID_RE.search(header)
            if id_match is None:
                # No id= means this is the native input sequence, not a design.
                continue
            conf = _CONF_RE.search(header)
            designs.append(
                {
                    "id": int(id_match.group(1)),
                    "sequence": sequence,
                    "overall_confidence": float(conf.group(1)) if conf else None,
                }
            )

    if not designs:
        raise ValueError(
            "MPNN produced no designs — only the input record was present in "
            f"{fasta_paths}."
        )

    return {"designs": designs, "num_designs": len(designs)}
