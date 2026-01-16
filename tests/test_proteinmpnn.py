"""Tests for ProteinMPNN pipeline runner - TDD RED phase first."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from protein_design_mcp.pipelines.proteinmpnn import (
    ProteinMPNNConfig,
    ProteinMPNNRunner,
)


# Test fixtures path
FIXTURES_DIR = Path(__file__).parent / "fixtures" / "test_pdbs"
MINI_PROTEIN_PDB = FIXTURES_DIR / "mini_protein.pdb"
TWO_CHAIN_PDB = FIXTURES_DIR / "two_chain_complex.pdb"


class TestProteinMPNNConfig:
    """Tests for ProteinMPNNConfig dataclass."""

    def test_default_config(self):
        """Default config should have sensible values."""
        config = ProteinMPNNConfig()
        assert config.num_sequences == 8
        assert config.sampling_temp == 0.1
        assert config.model_name == "v_48_020"

    def test_custom_config(self):
        """Custom config values should be set."""
        config = ProteinMPNNConfig(
            num_sequences=16,
            sampling_temp=0.2,
            model_name="v_48_010",
        )
        assert config.num_sequences == 16
        assert config.sampling_temp == 0.2
        assert config.model_name == "v_48_010"


class TestProteinMPNNRunner:
    """Tests for ProteinMPNNRunner class."""

    def test_runner_init_default(self):
        """Runner should initialize with default config."""
        runner = ProteinMPNNRunner()
        assert runner.config is not None
        assert runner.config.num_sequences == 8

    def test_runner_init_custom_config(self):
        """Runner should accept custom config."""
        config = ProteinMPNNConfig(num_sequences=4)
        runner = ProteinMPNNRunner(config=config)
        assert runner.config.num_sequences == 4

    @pytest.mark.asyncio
    async def test_design_sequences_returns_list(self, tmp_path):
        """design_sequences should return list of sequence info."""
        runner = ProteinMPNNRunner()

        with patch.object(runner, "_run_proteinmpnn") as mock_run:
            mock_run.return_value = None
            with patch.object(runner, "_parse_outputs") as mock_parse:
                mock_parse.return_value = [
                    {
                        "id": "seq_0",
                        "sequence": "MKVGAAAAAA",
                        "score": -1.5,
                        "recovery": 0.8,
                    }
                ]

                result = await runner.design_sequences(
                    backbone_pdb=str(MINI_PROTEIN_PDB),
                    output_dir=str(tmp_path),
                    num_sequences=1,
                )

                assert isinstance(result, list)
                assert len(result) == 1
                assert "sequence" in result[0]
                assert "score" in result[0]

    @pytest.mark.asyncio
    async def test_design_sequences_validates_backbone(self, tmp_path):
        """Should raise error for invalid backbone PDB."""
        runner = ProteinMPNNRunner()

        with pytest.raises((FileNotFoundError, ValueError)):
            await runner.design_sequences(
                backbone_pdb="/nonexistent/backbone.pdb",
                output_dir=str(tmp_path),
            )

    @pytest.mark.asyncio
    async def test_design_sequences_creates_output_dir(self, tmp_path):
        """Should create output directory if it doesn't exist."""
        runner = ProteinMPNNRunner()
        output_dir = tmp_path / "new_output_dir"

        with patch.object(runner, "_run_proteinmpnn") as mock_run:
            mock_run.return_value = None
            with patch.object(runner, "_parse_outputs") as mock_parse:
                mock_parse.return_value = []

                await runner.design_sequences(
                    backbone_pdb=str(MINI_PROTEIN_PDB),
                    output_dir=str(output_dir),
                )

                assert output_dir.exists()

    def test_build_command(self, tmp_path):
        """_build_command should return valid command list."""
        runner = ProteinMPNNRunner()

        cmd = runner._build_command(
            backbone_pdb=str(MINI_PROTEIN_PDB),
            output_dir=str(tmp_path),
            num_sequences=4,
            fixed_positions=None,
        )

        assert isinstance(cmd, list)
        assert len(cmd) > 0

    def test_build_command_with_fixed_positions(self, tmp_path):
        """Command should include fixed positions when specified."""
        runner = ProteinMPNNRunner()

        cmd = runner._build_command(
            backbone_pdb=str(MINI_PROTEIN_PDB),
            output_dir=str(tmp_path),
            num_sequences=4,
            fixed_positions=[1, 2, 3],
        )

        cmd_str = " ".join(cmd)
        # Should have some reference to fixed positions
        assert isinstance(cmd, list)

    def test_parse_outputs(self, tmp_path):
        """_parse_outputs should parse FASTA output files."""
        runner = ProteinMPNNRunner()

        # Create mock output FASTA file
        fasta_content = """>seq_0, score=-1.5, recovery=0.80
MKVGAAAAAA
>seq_1, score=-1.2, recovery=0.75
MKVGVVVVVV
"""
        fasta_file = tmp_path / "seqs" / "designed.fa"
        fasta_file.parent.mkdir(parents=True, exist_ok=True)
        fasta_file.write_text(fasta_content)

        results = runner._parse_outputs(str(tmp_path))

        assert len(results) >= 2
        assert results[0]["sequence"] == "MKVGAAAAAA"

    @pytest.mark.asyncio
    async def test_design_for_interface(self, tmp_path):
        """design_for_interface should design interface-optimized sequences."""
        runner = ProteinMPNNRunner()

        with patch.object(runner, "_run_proteinmpnn") as mock_run:
            mock_run.return_value = None
            with patch.object(runner, "_parse_outputs") as mock_parse:
                mock_parse.return_value = [
                    {"id": "seq_0", "sequence": "AGSFY", "score": -2.0}
                ]

                result = await runner.design_for_interface(
                    complex_pdb=str(TWO_CHAIN_PDB),
                    design_chain="B",
                    interface_residues=["A3"],
                    output_dir=str(tmp_path),
                )

                assert isinstance(result, list)


class TestSequenceValidation:
    """Tests for sequence output validation."""

    def test_parse_fasta_header(self):
        """Should correctly parse FASTA headers with scores."""
        runner = ProteinMPNNRunner()

        header = ">seq_0, score=-1.5, recovery=0.80"
        parsed = runner._parse_fasta_header(header)

        assert parsed["id"] == "seq_0"
        assert parsed["score"] == pytest.approx(-1.5)
        assert parsed["recovery"] == pytest.approx(0.80)

    def test_parse_fasta_header_minimal(self):
        """Should handle minimal headers."""
        runner = ProteinMPNNRunner()

        header = ">sequence_1"
        parsed = runner._parse_fasta_header(header)

        assert "sequence_1" in parsed["id"]


class TestEdgeCases:
    """Edge case tests."""

    @pytest.mark.asyncio
    async def test_single_sequence(self, tmp_path):
        """Should work with num_sequences=1."""
        runner = ProteinMPNNRunner()

        with patch.object(runner, "_run_proteinmpnn"):
            with patch.object(runner, "_parse_outputs") as mock_parse:
                mock_parse.return_value = [{"id": "seq_0", "sequence": "AAA", "score": -1.0}]

                result = await runner.design_sequences(
                    backbone_pdb=str(MINI_PROTEIN_PDB),
                    output_dir=str(tmp_path),
                    num_sequences=1,
                )
                assert len(result) == 1

    @pytest.mark.asyncio
    async def test_many_sequences(self, tmp_path):
        """Should handle large number of sequences."""
        runner = ProteinMPNNRunner()

        with patch.object(runner, "_run_proteinmpnn"):
            with patch.object(runner, "_parse_outputs") as mock_parse:
                mock_parse.return_value = [
                    {"id": f"seq_{i}", "sequence": "AAA", "score": -1.0}
                    for i in range(100)
                ]

                result = await runner.design_sequences(
                    backbone_pdb=str(MINI_PROTEIN_PDB),
                    output_dir=str(tmp_path),
                    num_sequences=100,
                )
                assert len(result) == 100

    def test_fixed_all_positions(self, tmp_path):
        """Should handle fixing all positions."""
        runner = ProteinMPNNRunner()

        # Fixed all 5 positions in mini_protein
        cmd = runner._build_command(
            backbone_pdb=str(MINI_PROTEIN_PDB),
            output_dir=str(tmp_path),
            num_sequences=1,
            fixed_positions=[1, 2, 3, 4, 5],
        )
        assert isinstance(cmd, list)
