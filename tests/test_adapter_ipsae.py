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
SAMPLE_TABLE = """\
Chn1 Chn2  PAE Dist  Type   ipSAE    ipSAE_d0chn ipSAE_d0dom  ipTM_af  ipTM_d0chn     pDockQ     pDockQ2    LIS       n0res  n0chn  n0dom   d0res   d0chn   d0dom  nres1   nres2   dist1   dist2  Model
A    B     15   8   asym  0.721300  0.715000    0.702000     0.690000  0.685000       0.541200   0.530000   0.331100  50     45     40      12.3    11.2    10.5   120     115     25.3    24.1    model.cif
"""

# ipSAE's real by-residue detail file, written alongside the results table
# (see ipsae/core.py's _write_byres_output): a real column set that shares
# "ipSAE" with the required set but is missing Chn1/Chn2/ipTM_af/pDockQ, so
# _find_header must reject it and keep looking rather than mis-parse it.
SAMPLE_BYRES = """\
i   AlignChn ScoredChain  AlignResNum  AlignResType  AlignRespLDDT      n0chn  n0dom  n0res    d0chn     d0dom     d0res   ipTM_pae  ipSAE_d0chn ipSAE_d0dom    ipSAE
1    A          B          1            ALA            90.00              50     40     35    11.20     10.50     9.80   0.6500    0.7150      0.7020         0.7213
"""


def _manifest():
    return next(m for m in load_manifests(manifest_dir()) if m.name == "run_ipsae")


def _completed_run_with_results(
    tmp_path: Path, table_text: str, byres_text: str | None = SAMPLE_BYRES
) -> CompletedRun:
    """Build a CompletedRun the way the real dispatcher populates one for
    run_ipsae after Task 7 fix round 1: `structure` is staged into
    `workdir/structure/`, ipSAE writes its results there, and
    `run.outputs["results_txt"]` is a LIST (multiple: true) of every
    collected `.txt` file — the results table AND the by-residue file."""
    results_dir = tmp_path / "structure"
    results_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    table_path = results_dir / "two_chain_complex_10_10.txt"
    table_path.write_text(table_text)
    paths.append(str(table_path))
    if byres_text is not None:
        byres_path = results_dir / "two_chain_complex_10_10_byres.txt"
        byres_path.write_text(byres_text)
        paths.append(str(byres_path))
    return CompletedRun(
        returncode=0, stdout="Calculation completed successfully!\n", stderr="",
        workdir=tmp_path, outputs={"results_txt": paths},
    )


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
        {"pae_file": "/tmp/pae.json", "structure": "/tmp/m.cif",
         "pae_cutoff": 12.5, "dist_cutoff": 8.0},
    )
    assert args == ["/tmp/pae.json", "/tmp/m.cif", "12.5", "8.0"]


def test_parse_output_extracts_ipsae_for_the_chain_pair(tmp_path):
    result = parse_output(_manifest(), _completed_run_with_results(tmp_path, SAMPLE_TABLE))
    assert result["ipsae"] == pytest.approx(0.7213)
    assert result["iptm_af"] == pytest.approx(0.6900)
    assert result["pdockq"] == pytest.approx(0.5412)
    assert result["chain_pair"] == "A_B"


def test_parse_output_skips_the_byres_file_and_reads_the_results_table(tmp_path):
    """multiple: true collects BOTH ipSAE output files; the by-residue file
    (written first here, to prove ordering doesn't matter) must never be
    mistaken for the results table even though it also contains 'ipSAE'."""
    run = _completed_run_with_results(tmp_path, SAMPLE_TABLE)
    reversed_run = CompletedRun(
        returncode=0, stdout=run.stdout, stderr="", workdir=run.workdir,
        outputs={"results_txt": list(reversed(run.outputs["results_txt"]))},
    )
    result = parse_output(_manifest(), reversed_run)
    assert result["chain_pair"] == "A_B"


def test_parse_output_raises_when_results_txt_output_is_missing(tmp_path):
    run = CompletedRun(
        returncode=0, stdout="", stderr="", workdir=tmp_path, outputs={},
    )
    with pytest.raises(ValueError, match="results_txt"):
        parse_output(_manifest(), run)


def test_parse_output_raises_when_no_row_is_present(tmp_path):
    with pytest.raises(ValueError, match="ipSAE"):
        parse_output(
            _manifest(),
            _completed_run_with_results(tmp_path, "no table here", byres_text=None),
        )


def test_parse_output_handles_negative_and_scientific_notation(tmp_path):
    table = (
        "Chn1 Chn2  PAE Dist  Type   ipSAE      ipTM_af   pDockQ     LIS   Model\n"
        "A    B     15  8    asym  -0.500000  1.2e-05    0.541200   0.33  model.cif\n"
    )
    result = parse_output(
        _manifest(), _completed_run_with_results(tmp_path, table, byres_text=None)
    )
    assert result["ipsae"] == pytest.approx(-0.5)
    assert result["iptm_af"] == pytest.approx(1.2e-05)


def test_parse_output_tolerates_non_numeric_columns_outside_the_required_set(tmp_path):
    """PAE, Dist and Type are not read by name and must not be assumed
    numeric — a placeholder like 'NA' in PAE/Dist must not break parsing."""
    table = (
        "Chn1 Chn2  PAE Dist  Type   ipSAE      ipTM_af   pDockQ     LIS   Model\n"
        "A    B     NA  NA   asym  0.721300   0.690000  0.541200   0.33  model.cif\n"
    )
    result = parse_output(
        _manifest(), _completed_run_with_results(tmp_path, table, byres_text=None)
    )
    assert result["ipsae"] == pytest.approx(0.7213)
    assert result["chain_pair"] == "A_B"


def test_validation_rejects_a_non_json_pae_file():
    with pytest.raises(ToolInputError, match="pae_file"):
        validate_and_fill(_manifest(), {"pae_file": "notes.txt",
                                        "structure": "m.cif"})


def test_validation_fills_the_default_cutoffs():
    params = validate_and_fill(
        _manifest(), {"pae_file": "p.json", "structure": "m.cif"}
    )
    assert params["pae_cutoff"] == 10.0
    assert params["dist_cutoff"] == 10.0
