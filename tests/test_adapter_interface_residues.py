import json
from pathlib import Path

import pytest

from protein_design_mcp.adapters.interface_residues import build_args, parse_output
from protein_design_mcp.app import manifest_dir
from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.loader import load_manifests
from protein_design_mcp.validation import ToolInputError, validate_and_fill

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "test_pdbs"
TWO_CHAIN_PDB = FIXTURES_DIR / "two_chain_complex.pdb"

SAMPLE_RESULT = {
    "target_chain": "A",
    "binder_chains": ["B"],
    "contact_cutoff_angstrom": 8.0,
    "residues": [
        {
            "chain": "A",
            "residue_number": 1,
            "residue_name": "ALA",
            "is_contact": False,
            "buried_sasa_a2": 0.0,
            "sasa_unbound_a2": 178.729,
            "sasa_bound_a2": 178.729,
            "hotspot_tag": "A1",
        },
        {
            "chain": "A",
            "residue_number": 3,
            "residue_name": "SER",
            "is_contact": True,
            "buried_sasa_a2": 45.89,
            "sasa_unbound_a2": 183.559,
            "sasa_bound_a2": 137.67,
            "hotspot_tag": "A3",
        },
    ],
    "partner_residues": [],
    "hotspot_tags": ["A3"],
    "n_interface_residues": 1,
    "total_buried_sasa_a2": 45.89,
}


def _manifest():
    return next(m for m in load_manifests(manifest_dir()) if m.name == "run_interface_residues")


def _run_with_results(tmp_path: Path, result: dict) -> CompletedRun:
    results_path = tmp_path / "results.json"
    results_path.write_text(json.dumps(result))
    return CompletedRun(
        returncode=0,
        stdout="n_interface_residues: 1\n",
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
    def test_maps_required_parameters(self):
        args = build_args(
            _manifest(),
            {
                "complex_pdb": "/tmp/c.pdb",
                "target_chain": "A",
                "binder_chains": ["B"],
                "contact_cutoff": 8.0,
            },
        )
        assert "--complex-pdb" in args and "/tmp/c.pdb" in args
        assert "--target-chain" in args and "A" in args
        assert "--binder-chains" in args
        assert json.loads(args[args.index("--binder-chains") + 1]) == ["B"]
        assert "--contact-cutoff" in args and "8.0" in args

    def test_multiple_binder_chains_serialize_as_json_list(self):
        args = build_args(
            _manifest(),
            {
                "complex_pdb": "/tmp/c.pdb",
                "target_chain": "A",
                "binder_chains": ["H", "L"],
                "contact_cutoff": 8.0,
            },
        )
        idx = args.index("--binder-chains")
        assert json.loads(args[idx + 1]) == ["H", "L"]


class TestParseOutput:
    def test_returns_the_results_json_contents(self, tmp_path):
        run = _run_with_results(tmp_path, SAMPLE_RESULT)
        result = parse_output(_manifest(), run)
        assert result == SAMPLE_RESULT

    def test_hotspot_tags_are_chain_plus_residue_number(self, tmp_path):
        run = _run_with_results(tmp_path, SAMPLE_RESULT)
        result = parse_output(_manifest(), run)
        assert result["hotspot_tags"] == ["A3"]

    def test_missing_results_json_raises(self, tmp_path):
        run = CompletedRun(
            returncode=0, stdout="", stderr="", workdir=tmp_path, outputs={}
        )
        with pytest.raises(ValueError, match="results_json"):
            parse_output(_manifest(), run)


class TestValidation:
    def test_binder_chains_requires_at_least_one(self):
        with pytest.raises(ToolInputError):
            validate_and_fill(
                _manifest(),
                {
                    "complex_pdb": "complex.pdb",
                    "target_chain": "A",
                    "binder_chains": [],
                },
            )

    def test_contact_cutoff_defaults_to_eight(self):
        params = validate_and_fill(
            _manifest(),
            {
                "complex_pdb": "complex.pdb",
                "target_chain": "A",
                "binder_chains": ["B"],
            },
        )
        assert params["contact_cutoff"] == 8.0

    def test_contact_cutoff_boundary_values_accepted(self):
        for cutoff in (3.0, 20.0):
            params = validate_and_fill(
                _manifest(),
                {
                    "complex_pdb": "complex.pdb",
                    "target_chain": "A",
                    "binder_chains": ["B"],
                    "contact_cutoff": cutoff,
                },
            )
            assert params["contact_cutoff"] == cutoff

    def test_contact_cutoff_out_of_range_rejected(self):
        with pytest.raises(ToolInputError):
            validate_and_fill(
                _manifest(),
                {
                    "complex_pdb": "complex.pdb",
                    "target_chain": "A",
                    "binder_chains": ["B"],
                    "contact_cutoff": 2.0,
                },
            )

    def test_multi_character_chain_id_rejected(self):
        with pytest.raises(ToolInputError):
            validate_and_fill(
                _manifest(),
                {
                    "complex_pdb": "complex.pdb",
                    "target_chain": "AB",
                    "binder_chains": ["B"],
                },
            )


class TestWrapperScriptOnRealFixture:
    """Exercises scripts/engines/interface_residues.py's own module-level
    functions directly (no subprocess) against a real, already-tested
    fixture (tests/fixtures/test_pdbs/two_chain_complex.pdb -- see
    tests/test_pdb_utils.py, which already exercises get_interface_residues
    against this exact file). This is the corner-case coverage CLAUDE.md
    requires (single item, value = 0 boundary, empty collection).
    """

    @pytest.fixture(autouse=True)
    def _import_wrapper(self):
        import importlib.util
        import sys

        wrapper_path = (
            Path(__file__).parent.parent / "scripts" / "engines" / "interface_residues.py"
        )
        spec = importlib.util.spec_from_file_location("interface_residues_wrapper", wrapper_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["interface_residues_wrapper"] = module
        spec.loader.exec_module(module)
        self.wrapper = module

    def test_real_two_chain_fixture_finds_a_real_contact(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        import sys

        argv = [
            "interface_residues.py",
            "--complex-pdb",
            str(TWO_CHAIN_PDB),
            "--target-chain",
            "A",
            "--binder-chains",
            '["B"]',
            "--contact-cutoff",
            "8.0",
        ]
        monkeypatch.setattr(sys, "argv", argv)
        self.wrapper.main()

        result = json.loads((tmp_path / "results.json").read_text())
        assert result["hotspot_tags"] == ["A3"]
        assert result["n_interface_residues"] == 1
        # residue A1 is present with explicit zero buried SASA -- value=0
        # must not be dropped or treated as "missing"
        res_a1 = next(r for r in result["residues"] if r["residue_number"] == 1)
        assert res_a1["buried_sasa_a2"] == 0.0
        assert res_a1["is_contact"] is False
        # partner side is evidence too, not empty
        assert len(result["partner_residues"]) == 2

    def test_unknown_target_chain_raises_clear_error(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        import sys
        from protein_design_mcp.exceptions import InvalidPDBError

        argv = [
            "interface_residues.py",
            "--complex-pdb",
            str(TWO_CHAIN_PDB),
            "--target-chain",
            "Z",
            "--binder-chains",
            '["B"]',
            "--contact-cutoff",
            "8.0",
        ]
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(InvalidPDBError, match="target_chain"):
            self.wrapper.main()

    def test_unknown_binder_chain_raises_clear_error(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        import sys
        from protein_design_mcp.exceptions import InvalidPDBError

        argv = [
            "interface_residues.py",
            "--complex-pdb",
            str(TWO_CHAIN_PDB),
            "--target-chain",
            "A",
            "--binder-chains",
            '["Z"]',
            "--contact-cutoff",
            "8.0",
        ]
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(InvalidPDBError, match="binder_chains"):
            self.wrapper.main()

    def test_single_binder_chain_list_boundary(self, tmp_path, monkeypatch):
        """minItems: 1 boundary -- exactly one binder chain must work."""
        monkeypatch.chdir(tmp_path)
        import sys

        argv = [
            "interface_residues.py",
            "--complex-pdb",
            str(TWO_CHAIN_PDB),
            "--target-chain",
            "A",
            "--binder-chains",
            '["B"]',
            "--contact-cutoff",
            "3.0",
        ]
        monkeypatch.setattr(sys, "argv", argv)
        self.wrapper.main()
        result = json.loads((tmp_path / "results.json").read_text())
        # a very tight cutoff is a valid, real result: possibly zero contacts
        assert result["hotspot_tags"] == []
        assert result["n_interface_residues"] == 0
        assert result["residues"]  # full evidence table is never empty
