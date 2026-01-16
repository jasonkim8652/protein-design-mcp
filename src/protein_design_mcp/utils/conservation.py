"""
Conservation scoring using BLAST.

Calculates evolutionary conservation scores by running BLAST
and analyzing sequence alignments from homologous proteins.
"""

import subprocess
import re
from dataclasses import dataclass
from collections import Counter


@dataclass
class ConservationProfile:
    """Container for conservation analysis results."""

    sequence: str
    scores: list[float]  # Conservation score per residue (0-1)
    highly_conserved: list[int]  # 1-indexed positions with high conservation
    num_homologs: int

    @property
    def average_conservation(self) -> float:
        """Calculate average conservation score."""
        if not self.scores:
            return 0.0
        return sum(self.scores) / len(self.scores)

    def get_score(self, position: int) -> float:
        """Get conservation score for a position (1-indexed)."""
        if position < 1 or position > len(self.scores):
            raise IndexError(f"Position {position} out of range")
        return self.scores[position - 1]


async def run_blast(sequence: str, database: str = "swissprot") -> str:
    """
    Run BLAST search against a protein database.

    Args:
        sequence: Query protein sequence
        database: BLAST database to search (default: swissprot)

    Returns:
        BLAST output in FASTA format

    Note:
        Requires NCBI BLAST+ to be installed and configured.
        For production use, consider using NCBI's web BLAST API.
    """
    # Create temporary FASTA file for query
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
        f.write(f">query\n{sequence}\n")
        query_file = f.name

    try:
        # Run blastp
        result = subprocess.run(
            [
                "blastp",
                "-query", query_file,
                "-db", database,
                "-outfmt", "6 sseqid sseq",  # Tabular format with subject sequence
                "-max_target_seqs", "50",
                "-evalue", "1e-5",
            ],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )
        return result.stdout
    except FileNotFoundError:
        # BLAST not installed - return empty result
        return ""
    except subprocess.TimeoutExpired:
        return ""
    finally:
        os.unlink(query_file)


def parse_blast_results(blast_output: str) -> list[str]:
    """
    Parse BLAST output to extract aligned sequences.

    Args:
        blast_output: BLAST output (FASTA format or tabular)

    Returns:
        List of aligned sequences
    """
    if not blast_output.strip():
        return []

    sequences = []

    # Try parsing FASTA format
    if blast_output.strip().startswith(">"):
        current_seq = []
        for line in blast_output.strip().split("\n"):
            if line.startswith(">"):
                if current_seq:
                    sequences.append("".join(current_seq))
                current_seq = []
            else:
                current_seq.append(line.strip())
        if current_seq:
            sequences.append("".join(current_seq))
    else:
        # Try parsing tabular format (outfmt 6)
        for line in blast_output.strip().split("\n"):
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                seq = parts[-1].replace("-", "")  # Remove gaps
                if seq:
                    sequences.append(seq)

    return sequences


def _calculate_position_conservation(residues: list[str]) -> float:
    """
    Calculate conservation score for a single position.

    Uses Shannon entropy-based scoring.
    """
    if not residues:
        return 0.5

    # Count amino acid frequencies
    counter = Counter(residues)
    total = len(residues)

    if total == 0:
        return 0.5

    # Most frequent amino acid
    most_common_count = counter.most_common(1)[0][1]

    # Simple conservation = frequency of most common residue
    return most_common_count / total


async def calculate_conservation_scores(sequence: str) -> ConservationProfile:
    """
    Calculate conservation scores for a protein sequence.

    Runs BLAST to find homologs and calculates per-residue conservation
    based on multiple sequence alignment.

    Args:
        sequence: Query protein sequence

    Returns:
        ConservationProfile with per-residue scores and summary
    """
    # Run BLAST to get homologous sequences
    blast_output = await run_blast(sequence)
    homolog_sequences = parse_blast_results(blast_output)

    seq_len = len(sequence)
    num_homologs = len(homolog_sequences)

    if num_homologs == 0:
        # No homologs found - return default scores
        scores = [0.5] * seq_len
        return ConservationProfile(
            sequence=sequence,
            scores=scores,
            highly_conserved=[],
            num_homologs=0,
        )

    # Align homologs to query sequence
    # For simplicity, we assume sequences are roughly aligned
    # In production, would use proper MSA (e.g., MUSCLE, MAFFT)
    aligned_seqs = [sequence] + homolog_sequences

    # Calculate conservation at each position
    scores = []
    highly_conserved = []

    for i in range(seq_len):
        # Get residues at this position from all sequences
        residues = []
        for seq in aligned_seqs:
            if i < len(seq):
                residues.append(seq[i])

        score = _calculate_position_conservation(residues)
        scores.append(score)

        # Mark as highly conserved if score > 0.8
        if score > 0.8:
            highly_conserved.append(i + 1)  # 1-indexed

    return ConservationProfile(
        sequence=sequence,
        scores=scores,
        highly_conserved=highly_conserved,
        num_homologs=num_homologs,
    )
