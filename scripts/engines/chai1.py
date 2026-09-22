"""Wrapper for Chai-1 (``run_chai1``). Runs inside the ``chai1`` environment
(plain PyPI install of ``chai_lab==0.6.1``, not editable).

Chai-1 takes two inputs neither of which is the plain a3m this server's MSA
tool produces:

1. A FASTA file whose headers name an entity type and a unique label
   (``>protein|name=<label>``, verified against
   ``chai_lab/data/dataset/inference_dataset.py:read_inputs``), one record
   per chain INSTANCE (Chai-1 has no "id: [A, B]" homo-oligomer shorthand
   the way Boltz does -- copies are separate, uniquely-named records with
   the same sequence).
2. A directory of per-sequence ``<sha256(seq.upper())>.aligned.pqt`` files
   (``chai_lab/data/parsing/msas/aligned_pqt.py:expected_basename`` /
   ``AlignedParquetModel``), not a plain a3m -- Chai-1 never reads a3m text
   directly for inference, only for its own (server-backed) MSA generation
   path, which this wrapper never calls. This wrapper converts each chain's
   supplied plain a3m into that parquet format itself, following the exact
   column construction ``chai_lab/data/dataset/msas/colabfold.py:
   generate_colabfold_msas`` uses for its own server-fetched a3ms (query
   row first, source_database/pairing_key/comment columns), with one
   necessary approximation: a merged, deduplicated a3m from
   run_mmseqs_search has no per-row database provenance left, so this
   mirrors that same module's own fallback (header starts with "UniRef" ->
   uniref90, else -> bfd_uniclust) rather than inventing a new rule.

Never passes ``--use-msa-server`` -- every chain's alignment (or deliberate
absence) is already fully specified via the directory built here.

Reads one argv: a JSON object (see ``adapters/chai1.py`` for its shape).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

INPUT_FASTA = "input.fasta"
OUTPUT_DIR = "outputs"
MSA_DIR = "msas"


def _hash_sequence(seq: str) -> str:
    return hashlib.sha256(seq.encode()).hexdigest()


def _read_a3m(text: str) -> list[tuple[str, str]]:
    """Parse a plain a3m/FASTA file's text into (header, sequence) pairs.
    Header is everything after '>' on its own line, up to end of line."""
    records: list[tuple[str, str]] = []
    header = None
    seq_lines: list[str] = []
    for line in text.splitlines():
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(seq_lines)))
            header = line[1:].strip()
            seq_lines = []
        elif header is not None:
            seq_lines.append(line.strip())
    if header is not None:
        records.append((header, "".join(seq_lines)))
    return records


def _source_for_header(header: str) -> str:
    """Best-effort source-database label for a hit row, since a merged,
    deduplicated a3m (run_mmseqs_search's unpaired_a3m) carries no per-row
    database provenance. Mirrors chai_lab's own colabfold.py fallback
    exactly (see this module's docstring) rather than inventing a new rule.
    """
    return "uniref90" if header.startswith("UniRef") else "bfd_uniclust"


def _write_aligned_pqt(sequence: str, a3m_text: str, msa_dir: Path) -> None:
    records = _read_a3m(a3m_text)
    if not records:
        # A query-only / zero-hit a3m from run_mmseqs_search is still a
        # single record (the query itself) -- an empty file would mean the
        # caller handed us something that isn't a3m at all.
        raise ValueError(
            f"msa file for sequence {sequence[:20]}... contained no a3m "
            "records (expected at least the query row)"
        )

    all_sequences = [seq for _, seq in records]
    source_databases = ["query"] + [
        _source_for_header(header) for header, _ in records[1:]
    ]
    pairing_keys = [""] * len(records)  # unpaired a3m: no cross-chain pairing
    aligned_df = pd.DataFrame(
        {
            "sequence": all_sequences,
            "source_database": source_databases,
            "pairing_key": pairing_keys,
            "comment": [""] * len(records),
        }
    )
    out_path = msa_dir / f"{_hash_sequence(sequence.upper())}.aligned.pqt"
    aligned_df.to_parquet(out_path)


def _fasta_records(chains: list[dict]) -> list[tuple[str, str]]:
    """One (header, sequence) FASTA record per chain INSTANCE -- ``copies``
    duplicates the record under a distinct label (Chai-1 requires unique
    entity names; see this module's docstring)."""
    records = []
    for chain_index, chain in enumerate(chains):
        for copy_index in range(chain["copies"]):
            label = f"chain{chain_index}_{copy_index}"
            records.append((f"protein|name={label}", chain["sequence"]))
    return records


def main() -> None:
    job = json.loads(sys.argv[1])
    chains = job["chains"]

    fasta_text = "".join(
        f">{header}\n{seq}\n" for header, seq in _fasta_records(chains)
    )
    Path(INPUT_FASTA).write_text(fasta_text)

    msa_dir = Path(MSA_DIR)
    msa_dir.mkdir(exist_ok=True)
    any_msa = False
    written_hashes: set[str] = set()
    for chain in chains:
        if chain["msa"] is None:
            continue
        any_msa = True
        seq_hash = _hash_sequence(chain["sequence"].upper())
        if seq_hash in written_hashes:
            continue  # identical sequence already converted once
        a3m_text = Path(chain["msa"]).read_text()
        _write_aligned_pqt(chain["sequence"], a3m_text, msa_dir)
        written_hashes.add(seq_hash)

    cmd = [
        "chai-lab",
        "fold",
        "--num-trunk-recycles",
        str(job["num_trunk_recycles"]),
        "--num-diffn-timesteps",
        str(job["num_diffn_timesteps"]),
        "--num-diffn-samples",
        str(job["num_diffn_samples"]),
        "--num-trunk-samples",
        str(job["num_trunk_samples"]),
        "--seed",
        str(job["seed"]),
    ]
    cmd.append("--use-esm-embeddings" if job["use_esm_embeddings"] else "--no-use-esm-embeddings")
    # Never --use-msa-server: every chain's alignment is already fully
    # specified above (policy: no tool here may build its own MSA).
    if any_msa:
        cmd += ["--msa-directory", str(msa_dir)]
    cmd += [INPUT_FASTA, OUTPUT_DIR]

    result = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
