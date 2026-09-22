"""Build an AlphaFold-3-shaped alignment with MMseqs2. Runs inside the
`mmseqs` environment and shells out to the standalone ``mmseqs`` binary
declared in ``engine.mounts`` (see ``run_mmseqs_search.yaml``).

Command sequence per database (createdb -> search -> result2msa -> unpackdb)
follows ``~/projects/af3-mmseqs-gpu/src/alphafold3/data/tools/mmseqs.py`` and
``mmseqs_template.py`` — the user's own working reference implementation, the
ground truth this wrapper is verified against (see the manifest and the
task-3 report for the specific points where this diverges from the MMseqs2
manual, which happens twice: see ``_run_search``'s ``-a`` note and the
template-header note below).

Three outputs, always written, never merely "absent":

- ``unpaired.a3m`` — every selected unpaired database (default: uniref90,
  mgnify, small_bfd), concatenated and deduplicated by exact aligned
  sequence, matching ``Msa.from_multiple_msas(..., deduplicate=True)`` in
  the reference. Valid for every co-folding tool's single-a3m ``msa``
  parameter (run_rf3, run_promera, Chai/Boltz/Protenix/OpenFold3), and
  together with paired.a3m for AlphaFold 3's ``unpairedMsa``.
- ``paired.a3m`` — UniProt, NOT deduplicated (matches the reference: pairing
  needs one row per source record, not a collapsed set), so AlphaFold 3's
  own per-chain taxonomy-based pairing (msa_pairing.py) has full multiplicity
  to work with. Only AlphaFold 3 consumes this field.
  **Single-chain (monomer) queries**: pairing only means something once a
  second chain's UniProt hits exist to pair against, which this tool — one
  sequence per call — never has itself. Rather than silently skip the
  search only when a caller "looks like" a monomer (this tool has no way to
  know how many chains the eventual complex has), this is the explicit
  ``pair`` parameter: True (default) runs the UniProt search anyway, since
  the caller may be about to fold a multimer one chain at a time and each
  chain's own paired.a3m is what AlphaFold 3's pairing logic needs later.
  False skips the ~78GB search for a caller who already knows this chain
  is going into a monomer prediction. EITHER way this file is written: an
  empty-of-hits search and ``pair=False`` produce the exact same shape (a
  single query-only record — see the corner-case note below), so a
  consumer never has to distinguish "no hits" from "not searched" by
  content; it can only find out by reading ``paired_hit_count`` /
  ``pair_searched`` in this tool's JSON reply.
- ``templates.a3m`` — pdb_seqres, optional via ``search_templates``. Headers
  are mmseqs' own raw shape (``>101m_A mol:protein length:154``), NOT
  reformatted into AlphaFold 3's internal templates.py hit-parser shape
  (``>101m_A/1-154 [subseq from] mol:protein length:154``, which needs an
  extra ``convertalis`` pass for alignment coordinates — see
  ``mmseqs_template.py``). A caller feeding this straight to AF3's own
  template parser must apply that reformatting itself; this tool hands back
  the hits, not a bespoke AF3-internal encoding.

Corner case — empty hit set: confirmed live (not assumed) that
``result2msa``/``unpackdb`` against a padded database with ZERO passing
hits still emit a valid single-record a3m containing only the query
(mmseqs' own fallback, not something this script has to invent). This
script relies on that: a zero-hit search and a skipped search
(``pair=False`` / ``search_templates=False``) are represented identically,
by construction, as the query-only single record — this file always
exists, is always valid a3m, and is never "absent".

Corner case — rejected characters: MMseqs2's own ``createdb`` was found,
empirically, NOT to reject digits, lowercase, embedded spaces, or `*` in a
sequence (verified live against a throwaway database) — it silently
accepts anything and encodes it. That is worse than a clean rejection: a
malformed input would search silently on garbage. The manifest's
``sequence`` pattern therefore rejects anything outside the standard
20 amino acids plus the common ambiguity codes (X, B, Z, J, U, O),
uppercase only, BEFORE this script (and mmseqs) ever sees it — enforced by
``protein_design_mcp.validation`` against the manifest schema, not by this
script.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

_AA_HEADER = "query"

_DB_FILENAMES = {
    "uniref90": "uniref90_padded",
    "mgnify": "mgnify_padded",
    "small_bfd": "small_bfd_padded",
}
_PAIR_DB_FILENAME = "uniprot_padded"
_TEMPLATE_DB_FILENAME = "pdb_seqres_padded"


class MmseqsStepError(RuntimeError):
    """One mmseqs subcommand exited non-zero."""


def _run(binary: str, args: list[str], step: str) -> None:
    cmd = [binary, *args]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise MmseqsStepError(
            f"mmseqs {step} exited {proc.returncode}.\n"
            f"command: {' '.join(cmd)}\n"
            f"stderr (tail):\n{proc.stderr.strip()[-2000:]}"
        )


def _parse_a3m(text: str) -> list[tuple[str, str]]:
    """Split a3m text into (header, aligned_sequence) pairs, in order."""
    records: list[tuple[str, str]] = []
    header: str | None = None
    chunks: list[str] = []
    for line in text.splitlines():
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(chunks)))
            header = line[1:]
            chunks = []
        elif header is not None:
            chunks.append(line.strip())
    if header is not None:
        records.append((header, "".join(chunks)))
    return records


def _write_a3m(records: list[tuple[str, str]]) -> str:
    return "".join(f">{header}\n{seq}\n" for header, seq in records)


def _query_only_a3m(sequence: str) -> str:
    return _write_a3m([(_AA_HEADER, sequence)])


def _search_one_database(
    *,
    binary: str,
    query_db: Path,
    target_db: str,
    workdir: Path,
    sensitivity: float,
    e_value: float,
    max_seqs: int,
    coverage: float,
    coverage_mode: int,
    min_seq_id: float,
    num_iterations: int,
    threads: int,
    use_gpu: bool,
) -> str:
    """Run one full search -> result2msa -> unpackdb cycle. Returns raw a3m text.

    Always returns SOME a3m text: mmseqs itself falls back to a query-only
    record when nothing passes the search thresholds (verified live — see
    the module docstring's empty-hit-set note), so this never raises for
    "no hits", only for a genuine mmseqs failure.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    result_db = workdir / "resultDB"
    tmp_dir = workdir / "tmp"
    tmp_dir.mkdir()

    search_args = [
        "search",
        str(query_db),
        target_db,
        str(result_db),
        str(tmp_dir),
        "-a",  # mandatory for alignment backtraces -> a3m generation
        "-s",
        str(sensitivity),
        "-e",
        str(e_value),
        "--threads",
        str(threads),
        "--max-seqs",
        str(max_seqs),
        "-c",
        str(coverage),
        "--cov-mode",
        str(coverage_mode),
        "--min-seq-id",
        str(min_seq_id),
        "--num-iterations",
        str(num_iterations),
    ]
    if use_gpu:
        search_args += ["--gpu", "1"]
    _run(binary, search_args, step=f"search ({target_db})")

    msa_db = workdir / "msaDB"
    _run(
        binary,
        [
            "result2msa",
            str(query_db),
            target_db,
            str(result_db),
            str(msa_db),
            "--msa-format-mode",
            "5",
            "--threads",
            str(threads),
        ],
        step=f"result2msa ({target_db})",
    )

    unpacked = workdir / "unpacked"
    unpacked.mkdir()
    _run(binary, ["unpackdb", str(msa_db), str(unpacked)], step=f"unpackdb ({target_db})")

    a3m_file = unpacked / "0"
    if a3m_file.exists():
        return a3m_file.read_text()
    # mmseqs was observed, live, to always write "0" even for a search that
    # passes zero hits (see module docstring) -- it is unpackdb's own
    # fallback, not something this script invents. A missing file here means
    # mmseqs' own behaviour changed underneath this wrapper; raise loudly
    # rather than silently fabricate a3m content.
    raise MmseqsStepError(
        f"mmseqs unpackdb ({target_db}) produced no file named '0' in "
        f"{unpacked} -- expected at least the query-only fallback record"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sequence")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mmseqs-binary", required=True)
    parser.add_argument("--db-root", required=True)
    parser.add_argument("--unpaired-database", action="append", default=[], dest="unpaired_databases")
    parser.add_argument("--pair", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--search-templates", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--use-gpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sensitivity", type=float, required=True)
    parser.add_argument("--e-value", type=float, required=True)
    parser.add_argument("--max-sequences", type=int, required=True)
    parser.add_argument("--coverage", type=float, required=True)
    parser.add_argument("--coverage-mode", type=int, required=True)
    parser.add_argument("--min-seq-id", type=float, required=True)
    parser.add_argument("--num-iterations", type=int, required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--template-e-value", type=float, required=True)
    parser.add_argument("--max-template-hits", type=int, required=True)
    args = parser.parse_args()

    start = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scratch = Path("scratch")
    scratch.mkdir(exist_ok=True)

    sequence = args.sequence

    query_fasta = scratch / "query.fasta"
    query_fasta.write_text(f">{_AA_HEADER}\n{sequence}\n")
    query_db = scratch / "queryDB"
    _run(args.mmseqs_binary, ["createdb", str(query_fasta), str(query_db)], step="createdb")

    # --- unpaired: every selected database, merged and deduplicated -------
    unpaired_texts: list[str] = []
    for db_name in args.unpaired_databases:
        target = f"{args.db_root}/{_DB_FILENAMES[db_name]}"
        text = _search_one_database(
            binary=args.mmseqs_binary,
            query_db=query_db,
            target_db=target,
            workdir=scratch / f"unpaired_{db_name}",
            sensitivity=args.sensitivity,
            e_value=args.e_value,
            max_seqs=args.max_sequences,
            coverage=args.coverage,
            coverage_mode=args.coverage_mode,
            min_seq_id=args.min_seq_id,
            num_iterations=args.num_iterations,
            threads=args.threads,
            use_gpu=args.use_gpu,
        )
        unpaired_texts.append(text)

    if unpaired_texts:
        merged: list[tuple[str, str]] = []
        seen_sequences: set[str] = set()
        first_records = _parse_a3m(unpaired_texts[0])
        merged.append(first_records[0])
        seen_sequences.add(first_records[0][1])
        for text in unpaired_texts:
            for header, seq in _parse_a3m(text)[1:]:
                if seq not in seen_sequences:
                    seen_sequences.add(seq)
                    merged.append((header, seq))
        unpaired_a3m_text = _write_a3m(merged)
        unpaired_hit_count = len(merged) - 1
    else:
        unpaired_a3m_text = _query_only_a3m(sequence)
        unpaired_hit_count = 0

    (output_dir / "unpaired.a3m").write_text(unpaired_a3m_text)

    # --- paired: UniProt, not deduplicated ---------------------------------
    if args.pair:
        paired_text = _search_one_database(
            binary=args.mmseqs_binary,
            query_db=query_db,
            target_db=f"{args.db_root}/{_PAIR_DB_FILENAME}",
            workdir=scratch / "paired",
            sensitivity=args.sensitivity,
            e_value=args.e_value,
            max_seqs=args.max_sequences,
            coverage=args.coverage,
            coverage_mode=args.coverage_mode,
            min_seq_id=args.min_seq_id,
            num_iterations=args.num_iterations,
            threads=args.threads,
            use_gpu=args.use_gpu,
        )
        paired_hit_count = len(_parse_a3m(paired_text)) - 1
    else:
        paired_text = _query_only_a3m(sequence)
        paired_hit_count = 0

    (output_dir / "paired.a3m").write_text(paired_text)

    # --- templates: pdb_seqres, raw mmseqs headers --------------------------
    if args.search_templates:
        template_text = _search_one_database(
            binary=args.mmseqs_binary,
            query_db=query_db,
            target_db=f"{args.db_root}/{_TEMPLATE_DB_FILENAME}",
            workdir=scratch / "templates",
            sensitivity=args.sensitivity,
            e_value=args.template_e_value,
            max_seqs=args.max_template_hits,
            coverage=args.coverage,
            coverage_mode=args.coverage_mode,
            min_seq_id=args.min_seq_id,
            num_iterations=1,
            threads=args.threads,
            use_gpu=args.use_gpu,
        )
        template_hit_count = len(_parse_a3m(template_text)) - 1
    else:
        template_text = _query_only_a3m(sequence)
        template_hit_count = 0

    (output_dir / "templates.a3m").write_text(template_text)

    summary = {
        "query_length": len(sequence),
        "unpaired_hit_count": unpaired_hit_count,
        "paired_hit_count": paired_hit_count,
        "template_hit_count": template_hit_count,
        "unpaired_databases_searched": list(args.unpaired_databases),
        "pair_searched": bool(args.pair),
        "templates_searched": bool(args.search_templates),
        "used_gpu": bool(args.use_gpu),
        "elapsed_seconds": time.time() - start,
    }
    (output_dir / "search_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"elapsed_seconds: {summary['elapsed_seconds']:.2f}")


if __name__ == "__main__":
    try:
        main()
    except MmseqsStepError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
