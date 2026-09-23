"""The parsing chain_proof's assertions rest on.

The script itself is the real check -- it makes live tool calls -- but its
verdicts are only as good as how it reads a structure. A wrong parse here
would turn a real handoff failure into a green run.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from chain_proof import chain_sequences, is_placeholder  # noqa: E402

THREE = {"A": "ALA", "G": "GLY", "K": "LYS", "W": "TRP"}


def _pdb(path: Path, spec: list[tuple[str, str, int]]) -> Path:
    lines = [
        f"ATOM  {i:5d}  CA  {THREE[aa]} {ch}{num:4d}    "
        f"{i:8.3f}{i:8.3f}{i:8.3f}  1.00  1.00           C"
        for i, (ch, aa, num) in enumerate(spec, start=1)
    ]
    path.write_text("\n".join(lines) + "\n")
    return path


def test_sequences_are_split_by_chain(tmp_path):
    p = _pdb(tmp_path / "a.pdb", [("A", "G", 1), ("A", "W", 2), ("B", "K", 1)])
    assert chain_sequences(p) == {"A": "GW", "B": "K"}


def test_only_the_first_model_is_read(tmp_path):
    """A multi-model file would otherwise concatenate every model into one
    sequence and make every length assertion nonsense."""
    p = _pdb(tmp_path / "b.pdb", [("A", "G", 1)])
    p.write_text(p.read_text() + "ENDMDL\n" + p.read_text())
    assert chain_sequences(p) == {"A": "G"}


def test_each_residue_is_counted_once_however_many_atoms_it_has(tmp_path):
    p = tmp_path / "c.pdb"
    p.write_text(
        "ATOM      1  N   GLY A   1       1.000   1.000   1.000  1.00  1.00           N\n"
        "ATOM      2  CA  GLY A   1       1.000   1.000   1.000  1.00  1.00           C\n"
        "ATOM      3  C   GLY A   1       1.000   1.000   1.000  1.00  1.00           C\n"
    )
    assert chain_sequences(p) == {"A": "G"}


def test_an_unknown_residue_name_is_not_silently_dropped(tmp_path):
    """Dropping it would shorten the sequence and make a length comparison
    pass when the chains actually differ."""
    p = tmp_path / "d.pdb"
    p.write_text(
        "ATOM      1  CA  XYZ A   1       1.000   1.000   1.000  1.00  1.00           C\n"
    )
    assert chain_sequences(p) == {"A": "?"}


def test_placeholder_detection_catches_the_three_real_cases():
    """UNK, poly-alanine and poly-glycine are what the generators here actually
    emit when they have not designed a sequence."""
    assert is_placeholder("X" * 80)
    assert is_placeholder("A" * 80)
    assert is_placeholder("G" * 70)


def test_a_real_sequence_is_not_a_placeholder():
    assert not is_placeholder("GDSFKAILDNAGNIVEVPAELRDSVLKNLHLG")


def test_an_empty_chain_counts_as_a_placeholder():
    """Nothing designed is not a design."""
    assert is_placeholder("")
