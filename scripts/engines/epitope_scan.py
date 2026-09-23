"""Scan an unbound target for candidate surface (epitope) residues, ranked
by exposure alone, with per-residue exposure AND conservation evidence
exposed separately -- never combined into one opaque score (see the
manifest doc's "why this replaces suggest_hotspots").

Runs inside the ``protein-design-mcp`` environment itself, the same as
``interface_residues.py``: this wires already-tested, already-in-tree
analysis (``protein_design_mcp.utils.sasa.calculate_sasa`` for exposure,
``protein_design_mcp.utils.conservation._calculate_position_conservation``
for per-column conservation) to validated CLI arguments. No network access:
this script never fetches a structure or queries UniProt -- ``target_pdb``
is caller-supplied, and conservation (if wanted) comes from a caller-supplied
alignment file, exactly like every ``msa`` parameter elsewhere in this
server. There is no "auto".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from protein_design_mcp.exceptions import InvalidPDBError
from protein_design_mcp.utils.conservation import _calculate_position_conservation
from protein_design_mcp.utils.pdb import parse_pdb
from protein_design_mcp.utils.sasa import calculate_sasa


def _parse_a3m(text: str) -> list[str]:
    """Return every record's sequence (header lines stripped), in file
    order -- the first record is always the query in an a3m file (it is
    what every other row was aligned TO). No alignment-library dependency:
    a3m is just FASTA with lowercase insertion-state letters, and this only
    needs the raw per-record sequence string.
    """
    sequences: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if not line:
            continue
        if line.startswith(">"):
            if current:
                sequences.append("".join(current))
                current = []
            continue
        current.append(line.strip())
    if current:
        sequences.append("".join(current))
    return sequences


def _match_state_columns(sequences: list[str]) -> list[list[str]]:
    """Return one list of aligned residues per QUERY COLUMN.

    a3m's convention: the query (first record) is the plain, ungapped input
    sequence and defines the columns; every other row uses uppercase
    letters and '-' for the "match states" aligned 1:1 to query columns, and
    lowercase letters for "insert states" -- residues that don't correspond
    to any query position at all. Dropping the lowercase characters from
    every row (the query has none) leaves exactly one aligned column per
    query residue, which is what a per-residue conservation score needs.
    """
    if not sequences:
        return []
    query = sequences[0]
    if any(c.islower() for c in query):
        raise ValueError(
            "malformed a3m: the query row (first record) contains lowercase "
            "(insert-state) characters -- the query is expected to be the "
            "plain, ungapped input sequence"
        )
    n_columns = len(query)
    match_rows = []
    for index, seq in enumerate(sequences):
        matched = "".join(c for c in seq if not c.islower())
        if len(matched) != n_columns:
            raise ValueError(
                f"malformed a3m: record {index} has {len(matched)} match-state "
                f"column(s), expected {n_columns} (the query's length)"
            )
        match_rows.append(matched)
    columns = []
    for j in range(n_columns):
        residues = [row[j] for row in match_rows if row[j] not in ("-", ".")]
        columns.append(residues)
    return columns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-pdb", required=True)
    parser.add_argument("--chain", required=True)
    parser.add_argument("--msa", default=None, help="path to an a3m alignment; omit for none")
    parser.add_argument("--exposure-threshold-a2", type=float, required=True)
    parser.add_argument("--conserved-threshold", type=float, required=True)
    parser.add_argument("--top-n-hotspots", type=int, required=True)
    args = parser.parse_args()

    structure = parse_pdb(args.target_pdb)
    present = {c.chain_id for c in structure.chains}
    if args.chain not in present:
        raise InvalidPDBError(
            f"chain {args.chain!r} not found in {args.target_pdb}; "
            f"chains present: {sorted(present)}"
        )
    chain_obj = next(c for c in structure.chains if c.chain_id == args.chain)

    sasa_per_residue = calculate_sasa(args.target_pdb).per_residue

    conservation_scores: list[float | None]
    num_aligned_sequences = 0
    msa_provided = args.msa is not None
    if msa_provided:
        sequences = _parse_a3m(Path(args.msa).read_text())
        if not sequences:
            raise ValueError(f"msa file {args.msa} contains no records")
        query = sequences[0]
        if query != chain_obj.sequence:
            raise ValueError(
                f"msa's query row does not match chain {args.chain}'s own "
                f"sequence from {args.target_pdb}. "
                f"msa query ({len(query)} aa): {query}\n"
                f"chain {args.chain} ({len(chain_obj.sequence)} aa): "
                f"{chain_obj.sequence}\n"
                "Supply the alignment run_mmseqs_search built for THIS exact "
                "sequence (its unpaired_a3m), not a different chain's."
            )
        num_aligned_sequences = len(sequences) - 1  # exclude the query itself
        if num_aligned_sequences == 0:
            # A real, valid state (run_mmseqs_search always writes a
            # query-only a3m record for a zero-hit search) -- not an error,
            # but there is no diversity to score conservation from. Report
            # it honestly as "no score" rather than a fabricated number.
            conservation_scores = [None] * len(chain_obj.sequence)
        else:
            columns = _match_state_columns(sequences)
            conservation_scores = [
                _calculate_position_conservation(column) for column in columns
            ]
    else:
        conservation_scores = [None] * len(chain_obj.sequence)

    residues = []
    for index, residue in enumerate(chain_obj.residues):
        key = f"{args.chain}{residue.residue_number}"
        sasa = sasa_per_residue.get(key, 0.0)
        is_exposed = sasa > args.exposure_threshold_a2
        conservation = conservation_scores[index] if index < len(conservation_scores) else None
        is_conserved = conservation is not None and conservation > args.conserved_threshold
        residues.append(
            {
                "chain": args.chain,
                "residue_number": residue.residue_number,
                "residue_name": residue.residue_name,
                "sasa_a2": round(sasa, 3),
                "is_exposed": is_exposed,
                "conservation_score": (
                    round(conservation, 4) if conservation is not None else None
                ),
                "is_conserved": is_conserved,
                "hotspot_tag": key,
            }
        )

    # Ranked by exposure ALONE, among exposed residues -- burial is a hard
    # structural precondition for being bindable at all, so filtering on it
    # is not a value judgment. Conservation is NOT folded into this order:
    # whether a conserved or a variable epitope is "better" depends on the
    # caller's goal (cross-reactivity vs specificity), so that choice is
    # left to the caller by exposing conservation_score per residue instead
    # of baking a weight into a combined score (see the manifest doc).
    hotspot_tags = [
        r["hotspot_tag"]
        for r in sorted(
            (r for r in residues if r["is_exposed"]),
            key=lambda r: (-r["sasa_a2"], r["residue_number"]),
        )
    ][: args.top_n_hotspots]

    result = {
        "chain": args.chain,
        "exposure_threshold_a2": args.exposure_threshold_a2,
        "conserved_threshold": args.conserved_threshold,
        "msa_provided": msa_provided,
        "num_aligned_sequences": num_aligned_sequences,
        "residues": residues,
        "hotspot_tags": hotspot_tags,
        "n_exposed_residues": sum(1 for r in residues if r["is_exposed"]),
        "n_conserved_residues": sum(1 for r in residues if r["is_conserved"]),
    }
    (Path.cwd() / "results.json").write_text(json.dumps(result, indent=2))
    print(f"n_exposed_residues: {result['n_exposed_residues']}")


if __name__ == "__main__":
    main()
