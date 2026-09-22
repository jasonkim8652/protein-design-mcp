from pathlib import Path

import pytest

from protein_design_mcp.adapters.esm_score import build_args, parse_output
from protein_design_mcp.app import manifest_dir
from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.loader import load_manifests
from protein_design_mcp.validation import ToolInputError, validate_and_fill

MANIFEST_DIR = manifest_dir()


def _manifest():
    return next(m for m in load_manifests(MANIFEST_DIR) if m.name == "run_esm_score")


def test_manifest_loads_and_is_not_composite():
    m = _manifest()
    assert m.category == "scoring"
    assert m.composite is False
    assert m.requires.gpu is True


def test_manifest_documents_what_it_is_not():
    doc = _manifest().doc
    assert "## What this is NOT" in doc
    assert "NOT a binding predictor" in _manifest().summary


def test_build_args_serialises_sequence_and_batch_size():
    import json

    args = build_args(_manifest(), {"sequence": "MKT", "batch_size": 16})
    assert len(args) == 1
    job = json.loads(args[0])
    assert job == {"sequence": "MKT", "batch_size": 16}


def test_parse_output_extracts_pll_and_per_residue_scores():
    import json

    stdout = json.dumps(
        {
            "pseudo_log_likelihood": -1.5,
            "per_residue_log_likelihood": [-0.1, -2.0, -2.4],
            "sequence_length": 3,
            "device": "cuda",
        }
    )
    result = parse_output(
        _manifest(),
        CompletedRun(returncode=0, stdout=stdout, stderr="", workdir=Path("/tmp")),
    )
    assert result["pseudo_log_likelihood"] == pytest.approx(-1.5)
    assert result["per_residue_log_likelihood"] == [-0.1, -2.0, -2.4]
    assert result["sequence_length"] == 3
    assert result["device"] == "cuda"


def test_parse_output_carries_the_developability_caveat():
    import json

    stdout = json.dumps(
        {"pseudo_log_likelihood": -1.0, "per_residue_log_likelihood": [-1.0], "sequence_length": 1}
    )
    result = parse_output(
        _manifest(),
        CompletedRun(returncode=0, stdout=stdout, stderr="", workdir=Path("/tmp")),
    )
    assert "developability proxy" in result["caveat"]
    assert "bind its target" in result["caveat"]


def test_parse_output_reads_the_last_stdout_line_not_the_first():
    """torch/esm import warnings can land on stdout before the JSON result --
    the adapter must not be confused by noise preceding the real line."""
    import json

    result_line = json.dumps(
        {"pseudo_log_likelihood": -0.5, "per_residue_log_likelihood": [-0.5], "sequence_length": 1}
    )
    stdout = f"UserWarning: some noisy warning\n{result_line}\n"
    result = parse_output(
        _manifest(),
        CompletedRun(returncode=0, stdout=stdout, stderr="", workdir=Path("/tmp")),
    )
    assert result["pseudo_log_likelihood"] == pytest.approx(-0.5)


def test_parse_output_raises_when_stdout_is_empty():
    with pytest.raises(ValueError, match="no output"):
        parse_output(
            _manifest(),
            CompletedRun(returncode=0, stdout="", stderr="some crash", workdir=Path("/tmp")),
        )


def test_parse_output_raises_when_stdout_is_not_json():
    with pytest.raises(ValueError, match="not valid JSON"):
        parse_output(
            _manifest(),
            CompletedRun(returncode=0, stdout="not json at all", stderr="", workdir=Path("/tmp")),
        )


def test_parse_output_raises_when_a_required_field_is_missing():
    import json

    stdout = json.dumps(
        {"pseudo_log_likelihood": -1.0, "per_residue_log_likelihood": [-1.0]}
    )
    with pytest.raises(ValueError, match="sequence_length"):
        parse_output(
            _manifest(),
            CompletedRun(returncode=0, stdout=stdout, stderr="", workdir=Path("/tmp")),
        )


# --- Corner cases required by CLAUDE.md's TDD workflow ---------------------


def test_parse_output_handles_zero_valued_pll():
    """A pseudo_log_likelihood of exactly 0 must survive -- 0 is a valid
    (if unusual) float score and must not be treated as falsy/absent."""
    import json

    stdout = json.dumps(
        {"pseudo_log_likelihood": 0.0, "per_residue_log_likelihood": [0.0], "sequence_length": 1}
    )
    result = parse_output(
        _manifest(),
        CompletedRun(returncode=0, stdout=stdout, stderr="", workdir=Path("/tmp")),
    )
    assert result["pseudo_log_likelihood"] == 0.0


def test_parse_output_single_residue_sequence():
    """Aggregation with n=1: a single-residue sequence's PLL is just that
    one residue's own score, not an average over a longer list."""
    import json

    stdout = json.dumps(
        {"pseudo_log_likelihood": -3.2, "per_residue_log_likelihood": [-3.2], "sequence_length": 1}
    )
    result = parse_output(
        _manifest(),
        CompletedRun(returncode=0, stdout=stdout, stderr="", workdir=Path("/tmp")),
    )
    assert result["sequence_length"] == 1
    assert result["per_residue_log_likelihood"] == [-3.2]


def test_validation_rejects_empty_sequence():
    with pytest.raises(ToolInputError, match="sequence"):
        validate_and_fill(_manifest(), {"sequence": ""})


def test_validation_rejects_ambiguity_codes():
    with pytest.raises(ToolInputError, match="sequence"):
        validate_and_fill(_manifest(), {"sequence": "MKTX"})


def test_validation_rejects_lowercase():
    with pytest.raises(ToolInputError, match="sequence"):
        validate_and_fill(_manifest(), {"sequence": "mkt"})


def test_validation_rejects_sequence_over_2000_residues():
    with pytest.raises(ToolInputError, match="sequence"):
        validate_and_fill(_manifest(), {"sequence": "A" * 2001})


def test_validation_accepts_sequence_at_the_2000_boundary():
    params = validate_and_fill(_manifest(), {"sequence": "A" * 2000})
    assert len(params["sequence"]) == 2000


def test_validation_accepts_single_residue_boundary():
    params = validate_and_fill(_manifest(), {"sequence": "A"})
    assert params["sequence"] == "A"


def test_validation_fills_default_batch_size():
    params = validate_and_fill(_manifest(), {"sequence": "MKT"})
    assert params["batch_size"] == 32


def test_validation_rejects_batch_size_zero():
    with pytest.raises(ToolInputError, match="batch_size"):
        validate_and_fill(_manifest(), {"sequence": "MKT", "batch_size": 0})
