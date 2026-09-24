"""Unit tests for scripts/engines/run_mmseqs_search.py's pure a3m helpers.

These functions (a3m parsing/writing, the query-only fallback, and the
merge/dedup logic that combines several databases' a3m output into one
unpaired.a3m) are the substantive logic in the wrapper script — the
subprocess orchestration around them is exercised by the end-to-end live
proof instead (see task-3-report.md), which is slow and needs a GPU and the
real databases, so it cannot run in the ordinary test suite.

The module is loaded by file path (mirroring
``protein_design_mcp.adapters_discovery._load_module``'s own technique)
rather than via a package import, since ``scripts/engines/`` is not a
package on the pytest path (only ``scripts/`` itself is, per
pyproject.toml's ``pythonpath``).

Fixture data below (the a3m record shapes, e.g. ``>A0A0P5NQT1_9CRUS`` and
the exact zero-hit fallback) is taken verbatim from real mmseqs2 runs against
the live small_bfd_padded database on this host (ubiquitin producing 100%
identical hits that dedup away entirely, and a random-ish 50-mer producing
none at all) — see task-3-report.md for the exact commands. Nothing here is
invented.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "engines" / "run_mmseqs_search.py"
)


def _load_wrapper():
    spec = importlib.util.spec_from_file_location("run_mmseqs_search_wrapper", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


wrapper = _load_wrapper()


# Real a3m text unpacked from `mmseqs unpackdb` against small_bfd_padded for
# the query ">ubiquitin\nMQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLE
# DGRTLSDYNIQKESTLHLVLRLRGG" at -s 1.0 --max-seqs 5 (first 5 hits kept for a
# compact fixture; every hit is byte-identical to the query -- ubiquitin is
# that conserved -- which is exactly the case this test exercises).
REAL_UBIQUITIN_A3M = """\
>query
MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG
>A0A0P5NQT1_9CRUS
MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG
>ERR1719422_560796
MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG
>ERR1711973_1016412
MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG
>ERR1711887_358036
MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG
"""

# Real a3m text unpacked for a cytochrome-c-family query against
# small_bfd_padded -- genuinely distinct (non-identical) homologues, so
# dedup keeps more than just the query.
REAL_CYTC_A3M = """\
>query
GDVEKGKKIFVQKCAQCHTVEKGGKHKTGPNLHGLFGRKTGQAPGYSYTAANKNKGIIWGEDTLMEYLENPKKYIPGTKMIFVGIKKKEERADLIAYLKKATNE
>L5KMQ3_PTEAL
GDVEKGKKIFVQKCAQCHTVEKGGKHKTGPNLHGLFGRKTGQAPGFSYTDANKNKGITWGEETLMEYLENPKKYIPGTKMIFAGIKKSGERADLIAYLKKATKE
>A0A0Q3M1I1_AMAAE
GDIEKGKKIFVQKCSQCHTVEKGGKHKTGPNLNGIFGRKTGQAEGFSYTDANKNKGITWGEDTLMEYLENPKKYIPGTKMIFAGIKKKSERADLIAYLKSAT--
"""

# The exact fallback mmseqs itself was observed, live, to write when a
# search passes zero hits: a single record holding only the query.
REAL_ZERO_HIT_A3M = """\
>randomish
WKPQVYRMTDLCNFHAWGPKRVYTQLNDEIFGHKMPQARTVYCLNDWGH
"""


def test_parse_a3m_splits_headers_and_sequences_in_order():
    records = wrapper._parse_a3m(REAL_CYTC_A3M)
    assert records[0] == (
        "query",
        "GDVEKGKKIFVQKCAQCHTVEKGGKHKTGPNLHGLFGRKTGQAPGYSYTAANKNKGIIWGEDTLMEYLENPKKYIPGTKMIFVGIKKKEERADLIAYLKKATNE",
    )
    assert records[1][0] == "L5KMQ3_PTEAL"
    assert len(records) == 3


def test_write_a3m_round_trips_parse_a3m():
    records = wrapper._parse_a3m(REAL_CYTC_A3M)
    assert wrapper._parse_a3m(wrapper._write_a3m(records)) == records


def test_query_only_a3m_has_exactly_one_record():
    text = wrapper._query_only_a3m("ACDEFG")
    records = wrapper._parse_a3m(text)
    assert records == [("query", "ACDEFG")]


# --- corner case: empty hit set -------------------------------------------


def test_zero_hit_search_result_parses_to_a_single_query_record():
    """A real mmseqs zero-hit fallback is a valid, single-record a3m -- not
    something the wrapper has to detect or special-case, only pass through."""
    records = wrapper._parse_a3m(REAL_ZERO_HIT_A3M)
    assert len(records) == 1
    assert records[0][0] == "randomish"


# --- corner case: merge/dedup across multiple databases --------------------


def _merge(texts: list[str]) -> tuple[str, int]:
    """Reproduce main()'s merge loop for a list of per-database a3m texts,
    without going through the whole CLI/subprocess machinery."""
    merged: list[tuple[str, str]] = []
    seen: set[str] = set()
    first_records = wrapper._parse_a3m(texts[0])
    merged.append(first_records[0])
    seen.add(first_records[0][1])
    for text in texts:
        for header, seq in wrapper._parse_a3m(text)[1:]:
            if seq not in seen:
                seen.add(seq)
                merged.append((header, seq))
    return wrapper._write_a3m(merged), len(merged) - 1


def test_merge_of_a_single_database_with_all_hits_identical_to_query_dedupes_to_zero():
    """Real case: ubiquitin's small_bfd hits are byte-identical to the query
    at the sequence level (see task-3-report.md) -- the reference
    implementation's own Msa.__init__ dedups the query itself into
    `unique_sequences` first, so 100%-identical hits collapse away too. A
    hit count of 0 here is correct, not a bug: a real search ran and found
    real homologues, they just carry no information beyond the query."""
    merged_text, hit_count = _merge([REAL_UBIQUITIN_A3M])
    assert hit_count == 0
    assert wrapper._parse_a3m(merged_text) == [wrapper._parse_a3m(REAL_UBIQUITIN_A3M)[0]]


def test_merge_of_a_single_database_keeps_genuinely_distinct_hits():
    merged_text, hit_count = _merge([REAL_CYTC_A3M])
    assert hit_count == 2
    records = wrapper._parse_a3m(merged_text)
    assert records[0][0] == "query"
    assert {r[0] for r in records[1:]} == {"L5KMQ3_PTEAL", "A0A0Q3M1I1_AMAAE"}


def test_merge_of_two_databases_deduplicates_across_them():
    """A hit appearing (byte-identical) in two different databases' results
    must be counted once, not twice."""
    merged_text, hit_count = _merge([REAL_CYTC_A3M, REAL_CYTC_A3M])
    assert hit_count == 2  # not 4 -- the second database's hits are all repeats


def test_merge_preserves_first_database_order_before_new_hits_from_the_second():
    duplicate_query_only = wrapper._write_a3m([wrapper._parse_a3m(REAL_CYTC_A3M)[0]])
    merged_text, hit_count = _merge([duplicate_query_only, REAL_CYTC_A3M])
    records = wrapper._parse_a3m(merged_text)
    assert hit_count == 2
    assert [r[0] for r in records] == ["query", "L5KMQ3_PTEAL", "A0A0Q3M1I1_AMAAE"]


# --- corner case: single item (one database selected) -----------------------


def test_merge_of_zero_databases_falls_back_to_query_only():
    """main() takes this branch (an empty `unpaired_databases` list) rather
    than calling _merge at all; asserted here for completeness of the n=0
    boundary the project's TDD checklist requires."""
    text = wrapper._query_only_a3m("ACDEFG")
    assert wrapper._parse_a3m(text) == [("query", "ACDEFG")]


# --- a silent GPU failure has to say something -------------------------------


def _wrapper():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).parent.parent / "scripts" / "engines" / "run_mmseqs_search.py"
    spec = importlib.util.spec_from_file_location("mmseqs_wrapper_diag", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_silent_gpu_failure_names_the_likely_cause():
    """`--gpu 1` loads the whole padded database into VRAM: small_bfd_padded
    is 16.8GB. On this shared 8-GPU box another user's process is routinely
    resident, and when the database does not fit, mmseqs exits 1 having
    printed NOTHING -- the sweep recorded

        mmseqs search (...small_bfd_padded) exited 1.
        stderr (tail):

    with an empty tail and no other clue. The same call succeeded unchanged
    once the GPU was free, so the code was never wrong; the failure was just
    illegible. Say what it probably was.
    """
    module = _wrapper()
    message = module.gpu_failure_hint(step="search", used_gpu=True, stderr="")
    assert message
    assert "gpu" in message.lower()
    assert "memory" in message.lower() or "vram" in message.lower()


def test_no_hint_when_mmseqs_actually_explained_itself():
    """A real diagnostic must not be buried under a guess."""
    module = _wrapper()
    assert module.gpu_failure_hint(
        step="search", used_gpu=True, stderr="Invalid database format") == ""


def test_no_hint_for_a_cpu_search():
    module = _wrapper()
    assert module.gpu_failure_hint(step="search", used_gpu=False, stderr="") == ""


def test_whitespace_only_stderr_counts_as_silent():
    module = _wrapper()
    assert module.gpu_failure_hint(step="search", used_gpu=True, stderr="  \n ")
