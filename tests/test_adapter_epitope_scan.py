import json
from pathlib import Path

import pytest

from protein_design_mcp.adapters.epitope_scan import build_args, parse_output
from protein_design_mcp.app import manifest_dir
from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.loader import load_manifests
from protein_design_mcp.validation import ToolInputError, validate_and_fill

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "test_pdbs"
MINI_PROTEIN_PDB = FIXTURES_DIR / "mini_protein.pdb"

SAMPLE_RESULT = {
    "chain": "A",
    "exposure_threshold_a2": 30.0,
    "conserved_threshold": 0.8,
    "msa_provided": False,
    "num_aligned_sequences": 0,
    "residues": [
        {
            "chain": "A",
            "residue_number": 1,
            "residue_name": "MET",
            "sasa_a2": 169.068,
            "is_exposed": True,
            "conservation_score": None,
            "is_conserved": False,
            "hotspot_tag": "A1",
        }
    ],
    "hotspot_tags": ["A1"],
    "n_exposed_residues": 1,
    "n_conserved_residues": 0,
}


def _manifest():
    return next(m for m in load_manifests(manifest_dir()) if m.name == "run_epitope_scan")


def _run_with_results(tmp_path: Path, result: dict) -> CompletedRun:
    results_path = tmp_path / "results.json"
    results_path.write_text(json.dumps(result))
    return CompletedRun(
        returncode=0,
        stdout="n_exposed_residues: 1\n",
        stderr="",
        workdir=tmp_path,
        outputs={"results_json": str(results_path)},
    )


class TestManifest:
    def test_manifest_loads_and_is_not_composite(self):
        m = _manifest()
        assert m.category == "target_analysis"
        assert m.composite is False
        assert m.requires.gpu is False

    def test_manifest_documents_what_it_needs(self):
        doc = _manifest().doc
        assert "## What this is" in doc
        assert "## What you must supply" in doc
        assert "## When to use this instead of the alternatives" in doc

    def test_manifest_documents_no_network_access(self):
        doc = _manifest().doc
        assert "No network access" in doc

    def test_manifest_documents_why_it_replaces_suggest_hotspots(self):
        doc = _manifest().doc
        assert "suggest_hotspots" in doc

    def test_manifest_shows_worked_examples_for_all_four_consumers(self):
        doc = _manifest().doc
        for tool in (
            "run_rfdiffusion_binder",
            "run_genie3_binder",
            "run_protpardelle",
            "run_rfdiffusion3_binder",
        ):
            assert tool in doc


class TestBuildArgs:
    def test_maps_required_parameters_without_msa(self):
        args = build_args(
            _manifest(),
            {
                "target_pdb": "/tmp/t.pdb",
                "chain": "A",
                "msa": None,
                "exposure_threshold_a2": 30.0,
                "conserved_threshold": 0.8,
                "top_n_hotspots": 10,
            },
        )
        assert "--target-pdb" in args and "/tmp/t.pdb" in args
        assert "--chain" in args and "A" in args
        assert "--msa" not in args

    def test_maps_msa_when_given(self):
        args = build_args(
            _manifest(),
            {
                "target_pdb": "/tmp/t.pdb",
                "chain": "A",
                "msa": "/tmp/query.a3m",
                "exposure_threshold_a2": 30.0,
                "conserved_threshold": 0.8,
                "top_n_hotspots": 10,
            },
        )
        assert "--msa" in args and "/tmp/query.a3m" in args


class TestParseOutput:
    def test_returns_the_results_json_contents(self, tmp_path):
        run = _run_with_results(tmp_path, SAMPLE_RESULT)
        result = parse_output(_manifest(), run)
        assert result == SAMPLE_RESULT

    def test_missing_results_json_raises(self, tmp_path):
        run = CompletedRun(
            returncode=0, stdout="", stderr="", workdir=tmp_path, outputs={}
        )
        with pytest.raises(ValueError, match="results_json"):
            parse_output(_manifest(), run)


class TestValidation:
    def test_msa_is_required_but_nullable(self):
        with pytest.raises(ToolInputError):
            validate_and_fill(
                _manifest(),
                {"target_pdb": "t.pdb", "chain": "A"},
            )

    def test_msa_null_is_accepted(self):
        params = validate_and_fill(
            _manifest(),
            {"target_pdb": "t.pdb", "chain": "A", "msa": None},
        )
        assert params["msa"] is None

    def test_msa_path_must_end_in_a3m(self):
        with pytest.raises(ToolInputError):
            validate_and_fill(
                _manifest(),
                {"target_pdb": "t.pdb", "chain": "A", "msa": "alignment.fasta"},
            )

    def test_exposure_threshold_boundary_zero_accepted(self):
        params = validate_and_fill(
            _manifest(),
            {
                "target_pdb": "t.pdb",
                "chain": "A",
                "msa": None,
                "exposure_threshold_a2": 0.0,
            },
        )
        assert params["exposure_threshold_a2"] == 0.0

    def test_top_n_hotspots_defaults_to_ten(self):
        params = validate_and_fill(
            _manifest(), {"target_pdb": "t.pdb", "chain": "A", "msa": None}
        )
        assert params["top_n_hotspots"] == 10


class TestWrapperScriptOnRealFixture:
    """Exercises scripts/engines/epitope_scan.py directly against the
    already-tested mini_protein.pdb fixture (tests/test_pdb_utils.py /
    tests/test_sasa.py both already use it). Corner cases per CLAUDE.md:
    empty collection (no msa), single item, value=0 boundary (a fully
    buried synthetic residue), explicit None vs a supplied-but-empty
    alignment.
    """

    @pytest.fixture(autouse=True)
    def _import_wrapper(self):
        import importlib.util
        import sys

        wrapper_path = Path(__file__).parent.parent / "scripts" / "engines" / "epitope_scan.py"
        spec = importlib.util.spec_from_file_location("epitope_scan_wrapper", wrapper_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["epitope_scan_wrapper"] = module
        spec.loader.exec_module(module)
        self.wrapper = module

    def _run(self, tmp_path, monkeypatch, extra_argv):
        import sys

        monkeypatch.chdir(tmp_path)
        argv = ["epitope_scan.py"] + extra_argv
        monkeypatch.setattr(sys, "argv", argv)
        self.wrapper.main()
        return json.loads((tmp_path / "results.json").read_text())

    def test_no_msa_gives_null_conservation_for_every_residue(self, tmp_path, monkeypatch):
        result = self._run(
            tmp_path,
            monkeypatch,
            [
                "--target-pdb",
                str(MINI_PROTEIN_PDB),
                "--chain",
                "A",
                "--exposure-threshold-a2",
                "30.0",
                "--conserved-threshold",
                "0.8",
                "--top-n-hotspots",
                "10",
            ],
        )
        assert result["msa_provided"] is False
        assert result["num_aligned_sequences"] == 0
        assert all(r["conservation_score"] is None for r in result["residues"])
        assert all(r["is_conserved"] is False for r in result["residues"])

    def test_query_only_a3m_is_a_real_zero_homolog_result_not_an_error(
        self, tmp_path, monkeypatch
    ):
        """Mirrors run_mmseqs_search's own documented behavior: a zero-hit
        search still writes a query-only a3m, and that is a normal, valid
        input -- not an error -- but there is no diversity to score.
        """
        a3m = tmp_path / "query_only.a3m"
        a3m.write_text(">query\nMKVGA\n")
        result = self._run(
            tmp_path,
            monkeypatch,
            [
                "--target-pdb",
                str(MINI_PROTEIN_PDB),
                "--chain",
                "A",
                "--msa",
                str(a3m),
                "--exposure-threshold-a2",
                "30.0",
                "--conserved-threshold",
                "0.8",
                "--top-n-hotspots",
                "10",
            ],
        )
        assert result["msa_provided"] is True
        assert result["num_aligned_sequences"] == 0
        assert all(r["conservation_score"] is None for r in result["residues"])

    def test_alignment_with_homologs_scores_conservation_per_column(
        self, tmp_path, monkeypatch
    ):
        a3m = tmp_path / "aln.a3m"
        a3m.write_text(
            ">query\nMKVGA\n>hit1\nMKVGA\n>hit2\nMKlVAA\n>hit3\nM-VGS\n"
        )
        result = self._run(
            tmp_path,
            monkeypatch,
            [
                "--target-pdb",
                str(MINI_PROTEIN_PDB),
                "--chain",
                "A",
                "--msa",
                str(a3m),
                "--exposure-threshold-a2",
                "30.0",
                "--conserved-threshold",
                "0.6",
                "--top-n-hotspots",
                "10",
            ],
        )
        assert result["num_aligned_sequences"] == 3
        by_pos = {r["residue_number"]: r for r in result["residues"]}
        # position 1 (M): every row agrees -> fully conserved
        assert by_pos[1]["conservation_score"] == pytest.approx(1.0)
        # position 4 (G): G,G,A,G among query+hits -> 3/4 = 0.75
        assert by_pos[4]["conservation_score"] == pytest.approx(0.75)

    def test_query_mismatch_raises_clear_error(self, tmp_path, monkeypatch):
        a3m = tmp_path / "bad.a3m"
        a3m.write_text(">query\nZZZZZ\n>hit\nZZZZZ\n")
        with pytest.raises(ValueError, match="does not match chain"):
            self._run(
                tmp_path,
                monkeypatch,
                [
                    "--target-pdb",
                    str(MINI_PROTEIN_PDB),
                    "--chain",
                    "A",
                    "--msa",
                    str(a3m),
                    "--exposure-threshold-a2",
                    "30.0",
                    "--conserved-threshold",
                    "0.8",
                    "--top-n-hotspots",
                    "10",
                ],
            )

    def test_unknown_chain_raises_clear_error(self, tmp_path, monkeypatch):
        from protein_design_mcp.exceptions import InvalidPDBError

        with pytest.raises(InvalidPDBError, match="chain"):
            self._run(
                tmp_path,
                monkeypatch,
                [
                    "--target-pdb",
                    str(MINI_PROTEIN_PDB),
                    "--chain",
                    "Z",
                    "--exposure-threshold-a2",
                    "30.0",
                    "--conserved-threshold",
                    "0.8",
                    "--top-n-hotspots",
                    "10",
                ],
            )

    def test_top_n_hotspots_caps_the_convenience_list_but_not_residues(
        self, tmp_path, monkeypatch
    ):
        result = self._run(
            tmp_path,
            monkeypatch,
            [
                "--target-pdb",
                str(MINI_PROTEIN_PDB),
                "--chain",
                "A",
                "--exposure-threshold-a2",
                "0.0",
                "--conserved-threshold",
                "0.8",
                "--top-n-hotspots",
                "1",
            ],
        )
        assert len(result["hotspot_tags"]) == 1
        assert len(result["residues"]) == 5  # mini_protein.pdb chain A: 5 residues

    def test_high_exposure_threshold_gives_empty_hotspot_list_not_an_error(
        self, tmp_path, monkeypatch
    ):
        """Empty collection corner case: nothing clears an absurdly high
        threshold, but that is a normal result, not a crash."""
        result = self._run(
            tmp_path,
            monkeypatch,
            [
                "--target-pdb",
                str(MINI_PROTEIN_PDB),
                "--chain",
                "A",
                "--exposure-threshold-a2",
                "1000.0",
                "--conserved-threshold",
                "0.8",
                "--top-n-hotspots",
                "10",
            ],
        )
        assert result["hotspot_tags"] == []
        assert result["n_exposed_residues"] == 0
        assert result["residues"]  # full table still returned
