"""Adapter tests for run_mmseqs_search.

REAL_SUMMARY_JSON below is byte-for-byte the search_summary.json a real run
wrote (see task-3-report.md for the exact commands): a cytochrome-c-family
query against small_bfd_padded (unpaired), uniprot_padded (paired) and
pdb_seqres_padded (templates), run through the actual GPU dispatch path
(ServerApp.call_tool -> EnvDispatcher.run -> the real mmseqs binary), not
invented. Its counts (99/100/448 hits) were independently cross-checked by
counting '>' records in the collected .a3m files themselves.
"""

from pathlib import Path

import pytest

from protein_design_mcp.adapters.mmseqs_search import (
    DB_ROOT,
    MMSEQS_BINARY,
    build_args,
    parse_output,
)
from protein_design_mcp.app import manifest_dir
from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.loader import load_manifests
from protein_design_mcp.validation import validate_and_fill

MANIFEST_DIR = manifest_dir()

REAL_SUMMARY_JSON = """\
{
  "query_length": 104,
  "unpaired_hit_count": 99,
  "paired_hit_count": 100,
  "template_hit_count": 448,
  "unpaired_databases_searched": [
    "small_bfd"
  ],
  "pair_searched": true,
  "templates_searched": true,
  "used_gpu": true,
  "elapsed_seconds": 87.22870302200317
}
"""

# Real search_summary.json from a run with pair=False and search_templates=False
# (a query too short/random to hit anything in small_bfd either): both corner
# cases the brief requires -- "single-chain" (pair skipped) and "empty hit set"
# -- captured from one real, if unremarkable, run.
REAL_SUMMARY_JSON_SKIPPED_PAIR_AND_TEMPLATES_ZERO_HITS = """\
{
  "query_length": 76,
  "unpaired_hit_count": 0,
  "paired_hit_count": 0,
  "template_hit_count": 0,
  "unpaired_databases_searched": [
    "small_bfd"
  ],
  "pair_searched": false,
  "templates_searched": false,
  "used_gpu": true,
  "elapsed_seconds": 16.7703640460968
}
"""


def _manifest():
    return next(m for m in load_manifests(MANIFEST_DIR) if m.name == "run_mmseqs_search")


def _full_params(**overrides):
    params = validate_and_fill(_manifest(), {"sequence": "ACDEFGHIKLMNPQRSTVWY"})
    params.update(overrides)
    return params


def test_build_args_includes_the_sequence_and_output_dir():
    args = build_args(_manifest(), _full_params())
    assert args[0] == "ACDEFGHIKLMNPQRSTVWY"
    assert "--output-dir" in args
    assert args[args.index("--output-dir") + 1] == "."


def test_build_args_hardcodes_the_deployment_binary_and_db_root_not_a_param():
    """MMSEQS_BINARY / DB_ROOT are deployment facts, not schema parameters --
    a caller cannot request a different binary or database root."""
    args = build_args(_manifest(), _full_params())
    assert args[args.index("--mmseqs-binary") + 1] == MMSEQS_BINARY
    assert args[args.index("--db-root") + 1] == DB_ROOT
    assert "mmseqs_binary" not in _manifest().schema
    assert "db_root" not in _manifest().schema


def test_build_args_maps_every_search_knob():
    params = _full_params(
        sensitivity=4.2,
        e_value=1e-6,
        max_sequences=42,
        coverage=0.5,
        coverage_mode=2,
        min_seq_id=0.3,
        num_iterations=3,
        threads=16,
        template_e_value=50.0,
        max_template_hits=200,
    )
    args = build_args(_manifest(), params)
    assert args[args.index("--sensitivity") + 1] == "4.2"
    assert args[args.index("--e-value") + 1] == "1e-06"
    assert args[args.index("--max-sequences") + 1] == "42"
    assert args[args.index("--coverage") + 1] == "0.5"
    assert args[args.index("--coverage-mode") + 1] == "2"
    assert args[args.index("--min-seq-id") + 1] == "0.3"
    assert args[args.index("--num-iterations") + 1] == "3"
    assert args[args.index("--threads") + 1] == "16"
    assert args[args.index("--template-e-value") + 1] == "50.0"
    assert args[args.index("--max-template-hits") + 1] == "200"


def test_build_args_repeats_unpaired_database_flag_for_each_entry():
    params = _full_params(unpaired_databases=["uniref90", "small_bfd"])
    args = build_args(_manifest(), params)
    indices = [i for i, a in enumerate(args) if a == "--unpaired-database"]
    values = [args[i + 1] for i in indices]
    assert values == ["uniref90", "small_bfd"]


def test_build_args_maps_pair_true_to_the_flag():
    args = build_args(_manifest(), _full_params(pair=True))
    assert "--pair" in args
    assert "--no-pair" not in args


def test_build_args_maps_pair_false_to_the_negative_flag():
    """Corner case: single-chain / monomer query -- caller opts out of the
    UniProt pairing search."""
    args = build_args(_manifest(), _full_params(pair=False))
    assert "--no-pair" in args
    assert "--pair" not in args


def test_build_args_maps_search_templates_false():
    args = build_args(_manifest(), _full_params(search_templates=False))
    assert "--no-search-templates" in args


def test_build_args_maps_use_gpu_false():
    args = build_args(_manifest(), _full_params(use_gpu=False))
    assert "--no-use-gpu" in args


def test_parse_output_reads_the_real_summary(tmp_path):
    summary_path = tmp_path / "search_summary.json"
    summary_path.write_text(REAL_SUMMARY_JSON)
    run = CompletedRun(
        returncode=0,
        stdout="",
        stderr="",
        workdir=tmp_path,
        outputs={
            "unpaired_a3m": str(tmp_path / "unpaired.a3m"),
            "paired_a3m": str(tmp_path / "paired.a3m"),
            "templates_a3m": str(tmp_path / "templates.a3m"),
            "search_summary": str(summary_path),
        },
    )
    result = parse_output(_manifest(), run)
    assert result["query_length"] == 104
    assert result["unpaired_hit_count"] == 99
    assert result["paired_hit_count"] == 100
    assert result["template_hit_count"] == 448
    assert result["unpaired_databases_searched"] == ["small_bfd"]
    assert result["pair_searched"] is True
    assert result["templates_searched"] is True
    assert result["used_gpu"] is True
    assert result["elapsed_seconds"] == pytest.approx(87.2287, rel=1e-4)
    # parse_output must never itself set 'outputs' -- app.py adds the
    # collected file paths afterward and treats a pre-existing key as a bug.
    assert "outputs" not in result


# --- corner cases: single-chain (pair skipped) and empty hit set, from a
# real run (see the module docstring). --------------------------------------


def test_parse_output_reports_zero_hits_as_a_normal_result_not_an_error(tmp_path):
    summary_path = tmp_path / "search_summary.json"
    summary_path.write_text(REAL_SUMMARY_JSON_SKIPPED_PAIR_AND_TEMPLATES_ZERO_HITS)
    run = CompletedRun(
        returncode=0,
        stdout="",
        stderr="",
        workdir=tmp_path,
        outputs={
            "unpaired_a3m": str(tmp_path / "unpaired.a3m"),
            "paired_a3m": str(tmp_path / "paired.a3m"),
            "templates_a3m": str(tmp_path / "templates.a3m"),
            "search_summary": str(summary_path),
        },
    )
    result = parse_output(_manifest(), run)
    assert result["unpaired_hit_count"] == 0
    assert result["paired_hit_count"] == 0
    assert result["template_hit_count"] == 0
    assert result["pair_searched"] is False
    assert result["templates_searched"] is False


def test_parse_output_raises_a_clear_error_when_summary_output_is_missing():
    run = CompletedRun(returncode=0, stdout="", stderr="", workdir=Path("/tmp"), outputs={})
    with pytest.raises(ValueError, match="search_summary"):
        parse_output(_manifest(), run)


def test_parse_output_raises_when_summary_json_is_missing_a_key(tmp_path):
    summary_path = tmp_path / "search_summary.json"
    summary_path.write_text('{"query_length": 10}')
    run = CompletedRun(
        returncode=0,
        stdout="",
        stderr="",
        workdir=tmp_path,
        outputs={"search_summary": str(summary_path)},
    )
    with pytest.raises(ValueError, match="missing key"):
        parse_output(_manifest(), run)
