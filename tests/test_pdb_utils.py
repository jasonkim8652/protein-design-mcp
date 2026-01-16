"""Tests for PDB utilities - TDD RED phase first."""

from pathlib import Path

import pytest

from protein_design_mcp.utils.pdb import (
    Chain,
    Residue,
    Structure,
    extract_sequence,
    get_interface_residues,
    parse_pdb,
    validate_pdb,
    write_pdb,
)


# Test fixtures path
FIXTURES_DIR = Path(__file__).parent / "fixtures" / "test_pdbs"
MINI_PROTEIN_PDB = FIXTURES_DIR / "mini_protein.pdb"
TWO_CHAIN_PDB = FIXTURES_DIR / "two_chain_complex.pdb"


class TestValidatePdb:
    """Tests for validate_pdb function."""

    def test_validate_existing_pdb_file(self):
        """Valid PDB file should pass validation."""
        is_valid, issues = validate_pdb(MINI_PROTEIN_PDB)
        assert is_valid is True
        assert len(issues) == 0

    def test_validate_nonexistent_file(self):
        """Non-existent file should fail validation."""
        is_valid, issues = validate_pdb("/nonexistent/path.pdb")
        assert is_valid is False
        assert "File not found" in issues[0]

    def test_validate_unusual_extension(self):
        """Unusual file extension should produce warning."""
        # Create a temp file with wrong extension
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 50.00           N\n")
            temp_path = f.name

        is_valid, issues = validate_pdb(temp_path)
        # Should have warning about extension
        assert any("extension" in issue.lower() for issue in issues)

        # Cleanup
        Path(temp_path).unlink()

    def test_validate_pdb_with_atom_records(self):
        """PDB with valid ATOM records should pass."""
        is_valid, issues = validate_pdb(MINI_PROTEIN_PDB)
        assert is_valid is True

    def test_validate_empty_pdb(self):
        """PDB without ATOM records should fail."""
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
            f.write(b"HEADER    EMPTY\nEND\n")
            temp_path = f.name

        is_valid, issues = validate_pdb(temp_path)
        assert is_valid is False
        assert any("no atom" in issue.lower() or "atom" in issue.lower() for issue in issues)

        Path(temp_path).unlink()


class TestParsePdb:
    """Tests for parse_pdb function."""

    def test_parse_single_chain(self):
        """Parse PDB with single chain."""
        structure = parse_pdb(MINI_PROTEIN_PDB)

        assert isinstance(structure, Structure)
        assert len(structure.chains) == 1
        assert structure.chains[0].chain_id == "A"

    def test_parse_two_chains(self):
        """Parse PDB with two chains."""
        structure = parse_pdb(TWO_CHAIN_PDB)

        assert isinstance(structure, Structure)
        assert len(structure.chains) == 2

        chain_ids = [c.chain_id for c in structure.chains]
        assert "A" in chain_ids
        assert "B" in chain_ids

    def test_parse_extracts_residues(self):
        """Parse correctly extracts residue information."""
        structure = parse_pdb(MINI_PROTEIN_PDB)

        chain_a = structure.chains[0]
        assert len(chain_a.residues) == 5  # MET, LYS, VAL, GLY, ALA

        # Check first residue
        first_res = chain_a.residues[0]
        assert first_res.residue_name == "MET"
        assert first_res.residue_number == 1
        assert first_res.chain_id == "A"

    def test_parse_extracts_sequence(self):
        """Parse extracts correct sequence."""
        structure = parse_pdb(MINI_PROTEIN_PDB)

        assert structure.chains[0].sequence == "MKVGA"

    def test_parse_extracts_atoms(self):
        """Parse correctly extracts atom information."""
        structure = parse_pdb(MINI_PROTEIN_PDB)

        first_res = structure.chains[0].residues[0]
        # MET has 5 atoms in our test file (N, CA, C, O, CB)
        assert len(first_res.atoms) == 5

        # Check atom properties
        n_atom = next(a for a in first_res.atoms if a["name"] == "N")
        assert n_atom["element"] == "N"
        assert n_atom["x"] == pytest.approx(0.0)
        assert n_atom["y"] == pytest.approx(0.0)
        assert n_atom["z"] == pytest.approx(0.0)

    def test_parse_nonexistent_file_raises(self):
        """Parsing non-existent file raises exception."""
        from protein_design_mcp.exceptions import InvalidPDBError

        with pytest.raises((InvalidPDBError, FileNotFoundError)):
            parse_pdb("/nonexistent/path.pdb")

    def test_parse_structure_name(self):
        """Structure name is derived from filename."""
        structure = parse_pdb(MINI_PROTEIN_PDB)
        assert "mini_protein" in structure.name.lower()


class TestExtractSequence:
    """Tests for extract_sequence function."""

    def test_extract_all_chains(self):
        """Extract sequence from all chains."""
        sequence = extract_sequence(MINI_PROTEIN_PDB)
        assert sequence == "MKVGA"

    def test_extract_specific_chain(self):
        """Extract sequence from specific chain."""
        sequence = extract_sequence(TWO_CHAIN_PDB, chain_id="A")
        assert sequence == "AGS"

    def test_extract_chain_b(self):
        """Extract sequence from chain B."""
        sequence = extract_sequence(TWO_CHAIN_PDB, chain_id="B")
        assert sequence == "FY"

    def test_extract_nonexistent_chain(self):
        """Extracting from non-existent chain raises error."""
        from protein_design_mcp.exceptions import InvalidPDBError

        with pytest.raises(InvalidPDBError):
            extract_sequence(TWO_CHAIN_PDB, chain_id="X")


class TestWritePdb:
    """Tests for write_pdb function."""

    def test_write_and_read_roundtrip(self, tmp_path):
        """Write PDB and read it back should preserve structure."""
        # First parse
        original = parse_pdb(MINI_PROTEIN_PDB)

        # Write to temp file
        output_path = tmp_path / "output.pdb"
        write_pdb(original, output_path)

        # Read back
        reread = parse_pdb(output_path)

        # Compare
        assert len(reread.chains) == len(original.chains)
        assert reread.chains[0].sequence == original.chains[0].sequence

    def test_write_creates_file(self, tmp_path):
        """Write creates the output file."""
        structure = parse_pdb(MINI_PROTEIN_PDB)
        output_path = tmp_path / "new_output.pdb"

        write_pdb(structure, output_path)

        assert output_path.exists()

    def test_write_preserves_atom_coordinates(self, tmp_path):
        """Written coordinates should match original."""
        original = parse_pdb(MINI_PROTEIN_PDB)
        output_path = tmp_path / "output.pdb"
        write_pdb(original, output_path)

        reread = parse_pdb(output_path)

        orig_atom = original.chains[0].residues[0].atoms[0]
        new_atom = reread.chains[0].residues[0].atoms[0]

        assert new_atom["x"] == pytest.approx(orig_atom["x"], abs=0.001)
        assert new_atom["y"] == pytest.approx(orig_atom["y"], abs=0.001)
        assert new_atom["z"] == pytest.approx(orig_atom["z"], abs=0.001)


class TestGetInterfaceResidues:
    """Tests for get_interface_residues function."""

    def test_find_interface_residues(self):
        """Find residues at interface between two chains."""
        chain_a_res, chain_b_res = get_interface_residues(
            TWO_CHAIN_PDB, chain_a="A", chain_b="B", distance_cutoff=10.0
        )

        # Should find some interface residues
        assert len(chain_a_res) > 0
        assert len(chain_b_res) > 0

    def test_interface_with_large_cutoff(self):
        """Large distance cutoff should find more residues."""
        res_a_small, res_b_small = get_interface_residues(
            TWO_CHAIN_PDB, chain_a="A", chain_b="B", distance_cutoff=5.0
        )
        res_a_large, res_b_large = get_interface_residues(
            TWO_CHAIN_PDB, chain_a="A", chain_b="B", distance_cutoff=15.0
        )

        assert len(res_a_large) >= len(res_a_small)
        assert len(res_b_large) >= len(res_b_small)

    def test_interface_invalid_chain(self):
        """Invalid chain ID should raise error."""
        from protein_design_mcp.exceptions import InvalidPDBError

        with pytest.raises(InvalidPDBError):
            get_interface_residues(TWO_CHAIN_PDB, chain_a="A", chain_b="X")


# Corner cases from CLAUDE.md
class TestEdgeCases:
    """Test edge cases and corner cases."""

    def test_empty_chain(self, tmp_path):
        """Handle structure with no residues gracefully."""
        # This tests the empty collection corner case
        empty_pdb = tmp_path / "empty.pdb"
        empty_pdb.write_text("HEADER EMPTY\nEND\n")

        is_valid, issues = validate_pdb(empty_pdb)
        assert is_valid is False

    def test_single_residue(self):
        """Handle protein with single residue."""
        import tempfile

        single_res = """ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 50.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00 50.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00 50.00           C
ATOM      4  O   ALA A   1       1.245   2.389   0.000  1.00 50.00           O
TER
END
"""
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w") as f:
            f.write(single_res)
            temp_path = f.name

        structure = parse_pdb(temp_path)
        assert len(structure.chains) == 1
        assert len(structure.chains[0].residues) == 1
        assert structure.chains[0].sequence == "A"

        Path(temp_path).unlink()

    def test_residue_number_zero(self):
        """Handle residue number 0 correctly."""
        import tempfile

        zero_res = """ATOM      1  N   ALA A   0       0.000   0.000   0.000  1.00 50.00           N
ATOM      2  CA  ALA A   0       1.458   0.000   0.000  1.00 50.00           C
TER
END
"""
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w") as f:
            f.write(zero_res)
            temp_path = f.name

        structure = parse_pdb(temp_path)
        assert structure.chains[0].residues[0].residue_number == 0

        Path(temp_path).unlink()

    def test_negative_coordinates(self):
        """Handle negative coordinates correctly."""
        import tempfile

        neg_coords = """ATOM      1  N   ALA A   1     -10.500  -5.250  -2.125  1.00 50.00           N
ATOM      2  CA  ALA A   1      -9.042  -5.250  -2.125  1.00 50.00           C
TER
END
"""
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w") as f:
            f.write(neg_coords)
            temp_path = f.name

        structure = parse_pdb(temp_path)
        first_atom = structure.chains[0].residues[0].atoms[0]
        assert first_atom["x"] == pytest.approx(-10.5)
        assert first_atom["y"] == pytest.approx(-5.25)
        assert first_atom["z"] == pytest.approx(-2.125)

        Path(temp_path).unlink()
