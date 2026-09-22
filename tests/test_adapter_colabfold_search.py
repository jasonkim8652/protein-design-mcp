from pathlib import Path

import pytest

from protein_design_mcp.adapters.colabfold_search import build_args, parse_output
from protein_design_mcp.app import manifest_dir
from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.loader import load_manifests
from protein_design_mcp.validation import ToolInputError, validate_and_fill

MANIFEST_DIR = manifest_dir()

SAMPLE_A3M = ">query\nMQIFVKTL\n>hit1\nMQIFVKT-\n"


def _manifest():
    return next(m for m in load_manifests(MANIFEST_DIR) if m.name == "run_colabfold_search")


def test_manifest_loads_and_is_not_composite():
    m = _manifest()
    assert m.category == "msa"
    assert m.composite is False
    assert m.requires.gpu is False


def test_manifest_documents_local_only_enforcement():
    doc = _manifest().doc
    assert "## Local search only" in doc
    assert "## Verification status" in doc


def test_build_args_includes_fixed_db_and_binary_constants():
    from protein_design_mcp.adapters.colabfold_search import DB_ROOT, MMSEQS_BINARY

    params = {
        "sequence": "MKT",
        "db1": "uniref30_2302_db",
        "db3": "colabfold_envdb_202108_db",
        "use_env": True,
        "prefilter_mode": 0,
        "sensitivity": None,
        "filter": 1,
        "expand_eval": float("inf"),
        "align_eval": 10,
        "diff": 3000,
        "qsc": -20.0,
        "max_accept": 1000000,
        "db_load_mode": 0,
        "threads": 64,
        "use_gpu": False,
        "gpu_server": False,
    }
    args = build_args(_manifest(), params)
    assert "MKT" in args
    assert MMSEQS_BINARY in args
    assert DB_ROOT in args
    assert "-s" not in args  # sensitivity=None must be omitted, not "None"


def test_build_args_includes_sensitivity_when_set():
    params = {
        "sequence": "MKT",
        "db1": "uniref30_2302_db",
        "db3": "colabfold_envdb_202108_db",
        "use_env": True,
        "prefilter_mode": 0,
        "sensitivity": 5.5,
        "filter": 1,
        "expand_eval": float("inf"),
        "align_eval": 10,
        "diff": 3000,
        "qsc": -20.0,
        "max_accept": 1000000,
        "db_load_mode": 0,
        "threads": 64,
        "use_gpu": False,
        "gpu_server": False,
    }
    args = build_args(_manifest(), params)
    assert "--sensitivity" in args
    assert "5.5" in args


def test_build_args_use_env_false_passes_no_use_env():
    params = {
        "sequence": "MKT",
        "db1": "uniref30_2302_db",
        "db3": "colabfold_envdb_202108_db",
        "use_env": False,
        "prefilter_mode": 0,
        "sensitivity": None,
        "filter": 1,
        "expand_eval": float("inf"),
        "align_eval": 10,
        "diff": 3000,
        "qsc": -20.0,
        "max_accept": 1000000,
        "db_load_mode": 0,
        "threads": 64,
        "use_gpu": False,
        "gpu_server": False,
    }
    args = build_args(_manifest(), params)
    assert "--no-use-env" in args


def test_parse_output_reads_query_length_from_real_file(tmp_path):
    a3m = tmp_path / "query.a3m"
    a3m.write_text(SAMPLE_A3M)
    result = parse_output(
        _manifest(),
        CompletedRun(
            returncode=0,
            stdout="",
            stderr="",
            workdir=tmp_path,
            outputs={"a3m": str(a3m)},
        ),
    )
    assert result["query_length"] == 8


def test_parse_output_raises_when_a3m_output_missing():
    with pytest.raises(ValueError, match="a3m"):
        parse_output(
            _manifest(),
            CompletedRun(returncode=0, stdout="", stderr="", workdir=Path("/tmp"), outputs={}),
        )


# --- Corner cases required by CLAUDE.md's TDD workflow ---------------------


def test_parse_output_single_residue_query(tmp_path):
    """Aggregation with n=1: a one-residue query's a3m still has a well-
    defined query_length."""
    a3m = tmp_path / "query.a3m"
    a3m.write_text(">query\nM\n")
    result = parse_output(
        _manifest(),
        CompletedRun(
            returncode=0, stdout="", stderr="", workdir=tmp_path, outputs={"a3m": str(a3m)}
        ),
    )
    assert result["query_length"] == 1


def test_parse_output_empty_a3m_file_gives_zero_length(tmp_path):
    """An empty/degenerate a3m (boundary: no sequence line at all) must
    report 0, not raise or return None."""
    a3m = tmp_path / "query.a3m"
    a3m.write_text(">query\n")
    result = parse_output(
        _manifest(),
        CompletedRun(
            returncode=0, stdout="", stderr="", workdir=tmp_path, outputs={"a3m": str(a3m)}
        ),
    )
    assert result["query_length"] == 0


def test_validation_rejects_lowercase_sequence():
    with pytest.raises(ToolInputError, match="sequence"):
        validate_and_fill(_manifest(), {"sequence": "mkt"})


def test_validation_accepts_ambiguity_codes():
    """Unlike run_esm_score, this tool matches run_mmseqs_search's broader
    character policy (X/B/Z/J/U/O allowed)."""
    params = validate_and_fill(_manifest(), {"sequence": "MKTX"})
    assert params["sequence"] == "MKTX"


def test_validation_fills_defaults():
    params = validate_and_fill(_manifest(), {"sequence": "MKT"})
    assert params["db1"] == "uniref30_2302_db"
    assert params["db3"] == "colabfold_envdb_202108_db"
    assert params["use_env"] is True
    assert params["sensitivity"] is None
    assert params["align_eval"] == 10
    assert params["expand_eval"] == float("inf")


def test_validation_rejects_zero_diff():
    with pytest.raises(ToolInputError, match="diff"):
        validate_and_fill(_manifest(), {"sequence": "MKT", "diff": 0})


def test_validation_accepts_diff_at_the_boundary():
    params = validate_and_fill(_manifest(), {"sequence": "MKT", "diff": 1})
    assert params["diff"] == 1


def test_validation_rejects_sensitivity_out_of_range():
    with pytest.raises(ToolInputError, match="sensitivity"):
        validate_and_fill(_manifest(), {"sequence": "MKT", "sensitivity": 0.5})
