from pathlib import Path

import pytest

from protein_design_mcp.adapters.ipsae import build_args, parse_output
from protein_design_mcp.app import manifest_dir
from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.loader import load_manifests
from protein_design_mcp.validation import ToolInputError, validate_and_fill

SAMPLE_STDOUT = """\
Chn1 Chn2   ipSAE   ipTM_af  pDockQ   LIS
A    B      0.7213  0.6900   0.5412   0.3311
"""


def _manifest():
    return next(m for m in load_manifests(manifest_dir()) if m.name == "run_ipsae")


def test_manifest_loads_and_is_not_composite():
    m = _manifest()
    assert m.category == "scoring"
    assert m.composite is False
    assert m.requires.gpu is False


def test_manifest_documents_the_choice_against_siblings():
    assert "## When to use this instead of the alternatives" in _manifest().doc


def test_build_args_passes_both_files_and_the_cutoffs():
    args = build_args(
        _manifest(),
        {"pae_json": "/tmp/pae.json", "structure": "/tmp/m.cif",
         "pae_cutoff": 10.0, "dist_cutoff": 10.0},
    )
    assert "/tmp/pae.json" in args
    assert "/tmp/m.cif" in args
    assert "10.0" in args


def test_parse_output_extracts_ipsae_for_the_chain_pair():
    result = parse_output(
        _manifest(),
        CompletedRun(returncode=0, stdout=SAMPLE_STDOUT, stderr="",
                     workdir=Path("/tmp")),
    )
    assert result["ipsae"] == pytest.approx(0.7213)
    assert result["iptm_af"] == pytest.approx(0.6900)
    assert result["pdockq"] == pytest.approx(0.5412)
    assert result["chain_pair"] == "A_B"


def test_parse_output_raises_when_no_row_is_present():
    with pytest.raises(ValueError, match="ipSAE"):
        parse_output(
            _manifest(),
            CompletedRun(returncode=0, stdout="no table here", stderr="",
                         workdir=Path("/tmp")),
        )


def test_validation_rejects_a_non_json_pae_file():
    with pytest.raises(ToolInputError, match="pae_json"):
        validate_and_fill(_manifest(), {"pae_json": "notes.txt",
                                        "structure": "m.cif"})


def test_validation_fills_the_default_cutoffs():
    params = validate_and_fill(
        _manifest(), {"pae_json": "p.json", "structure": "m.cif"}
    )
    assert params["pae_cutoff"] == 10.0
    assert params["dist_cutoff"] == 10.0
