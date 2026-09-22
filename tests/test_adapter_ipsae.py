from pathlib import Path

import pytest

from protein_design_mcp.adapters.ipsae import build_args, parse_output
from protein_design_mcp.app import manifest_dir
from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.loader import load_manifests
from protein_design_mcp.validation import ToolInputError, validate_and_fill

# Matches the real DunbrackLab/IPSAE column set (Chn1 Chn2 PAE Dist Type
# ipSAE ipSAE_d0chn ipSAE_d0dom ipTM_af ipTM_d0chn pDockQ pDockQ2 LIS n0res
# n0chn n0dom d0res d0chn d0dom nres1 nres2 dist1 dist2 Model), not the
# fabricated short-form table the original fixture used.
SAMPLE_STDOUT = """\
Chn1 Chn2  PAE Dist  Type   ipSAE    ipSAE_d0chn ipSAE_d0dom  ipTM_af  ipTM_d0chn     pDockQ     pDockQ2    LIS       n0res  n0chn  n0dom   d0res   d0chn   d0dom  nres1   nres2   dist1   dist2  Model
A    B     15   8   asym  0.721300  0.715000    0.702000     0.690000  0.685000       0.541200   0.530000   0.331100  50     45     40      12.3    11.2    10.5   120     115     25.3    24.1    model.cif
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


def test_manifest_discloses_only_the_first_chain_pair_is_returned():
    assert "first chain-pair row" in _manifest().doc


def test_build_args_passes_both_files_and_the_cutoffs_in_order():
    args = build_args(
        _manifest(),
        {"pae_json": "/tmp/pae.json", "structure": "/tmp/m.cif",
         "pae_cutoff": 12.5, "dist_cutoff": 8.0},
    )
    assert args == ["/tmp/pae.json", "/tmp/m.cif", "12.5", "8.0"]


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


def test_parse_output_handles_negative_and_scientific_notation():
    stdout = (
        "Chn1 Chn2  PAE Dist  Type   ipSAE      ipTM_af   pDockQ     LIS   Model\n"
        "A    B     15  8    asym  -0.500000  1.2e-05    0.541200   0.33  model.cif\n"
    )
    result = parse_output(
        _manifest(),
        CompletedRun(returncode=0, stdout=stdout, stderr="", workdir=Path("/tmp")),
    )
    assert result["ipsae"] == pytest.approx(-0.5)
    assert result["iptm_af"] == pytest.approx(1.2e-05)


def test_parse_output_tolerates_non_numeric_columns_outside_the_required_set():
    """PAE, Dist and Type are not read by name and must not be assumed
    numeric — a placeholder like 'NA' in PAE/Dist must not break parsing."""
    stdout = (
        "Chn1 Chn2  PAE Dist  Type   ipSAE      ipTM_af   pDockQ     LIS   Model\n"
        "A    B     NA  NA   asym  0.721300   0.690000  0.541200   0.33  model.cif\n"
    )
    result = parse_output(
        _manifest(),
        CompletedRun(returncode=0, stdout=stdout, stderr="", workdir=Path("/tmp")),
    )
    assert result["ipsae"] == pytest.approx(0.7213)
    assert result["chain_pair"] == "A_B"


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
