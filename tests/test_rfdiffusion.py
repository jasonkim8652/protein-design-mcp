"""Tests for RFdiffusion pipeline runner - TDD RED phase first."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import subprocess

import pytest

from protein_design_mcp.pipelines.rfdiffusion import (
    RFdiffusionConfig,
    RFdiffusionRunner,
)


# Test fixtures path
FIXTURES_DIR = Path(__file__).parent / "fixtures" / "test_pdbs"
MINI_PROTEIN_PDB = FIXTURES_DIR / "mini_protein.pdb"


class TestRFdiffusionConfig:
    """Tests for RFdiffusionConfig dataclass."""

    def test_default_config(self):
        """Default config should have sensible values."""
        config = RFdiffusionConfig()
        assert config.num_designs == 10
        assert config.binder_length == 80
        assert config.diffusion_steps == 50

    def test_custom_config(self):
        """Custom config values should be set."""
        config = RFdiffusionConfig(
            num_designs=5,
            binder_length=100,
            noise_scale=0.5,
        )
        assert config.num_designs == 5
        assert config.binder_length == 100
        assert config.noise_scale == 0.5


class TestRFdiffusionRunner:
    """Tests for RFdiffusionRunner class."""

    def test_runner_init_default(self):
        """Runner should initialize with default config."""
        runner = RFdiffusionRunner()
        assert runner.config is not None
        assert runner.config.num_designs == 10

    def test_runner_init_custom_config(self):
        """Runner should accept custom config."""
        config = RFdiffusionConfig(num_designs=5)
        runner = RFdiffusionRunner(config=config)
        assert runner.config.num_designs == 5

    @pytest.mark.asyncio
    async def test_generate_backbones_returns_list(self, tmp_path):
        """generate_backbones should return list of backbone info."""
        runner = RFdiffusionRunner()

        with patch.object(runner, "_run_rfdiffusion") as mock_run:
            mock_run.return_value = None
            with patch.object(runner, "_parse_outputs") as mock_parse:
                mock_parse.return_value = [
                    {
                        "id": "design_0000",
                        "pdb_path": str(tmp_path / "design_0000.pdb"),
                        "score": 0.85,
                    }
                ]

                result = await runner.generate_backbones(
                    target_pdb=str(MINI_PROTEIN_PDB),
                    hotspot_residues=["A1", "A2", "A3"],
                    output_dir=str(tmp_path),
                    num_designs=1,
                )

                assert isinstance(result, list)
                assert len(result) == 1
                assert "id" in result[0]
                assert "pdb_path" in result[0]

    @pytest.mark.asyncio
    async def test_generate_backbones_validates_target(self, tmp_path):
        """Should raise error for invalid target PDB."""
        runner = RFdiffusionRunner()

        with pytest.raises((FileNotFoundError, ValueError)):
            await runner.generate_backbones(
                target_pdb="/nonexistent/target.pdb",
                hotspot_residues=["A1"],
                output_dir=str(tmp_path),
            )

    @pytest.mark.asyncio
    async def test_generate_backbones_validates_hotspots(self, tmp_path):
        """Should raise error for empty hotspot list."""
        runner = RFdiffusionRunner()

        with pytest.raises(ValueError, match="[Hh]otspot"):
            await runner.generate_backbones(
                target_pdb=str(MINI_PROTEIN_PDB),
                hotspot_residues=[],
                output_dir=str(tmp_path),
            )

    @pytest.mark.asyncio
    async def test_generate_backbones_creates_output_dir(self, tmp_path):
        """Should create output directory if it doesn't exist."""
        runner = RFdiffusionRunner()
        output_dir = tmp_path / "new_output_dir"

        with patch.object(runner, "_run_rfdiffusion") as mock_run:
            mock_run.return_value = None
            with patch.object(runner, "_parse_outputs") as mock_parse:
                mock_parse.return_value = []

                await runner.generate_backbones(
                    target_pdb=str(MINI_PROTEIN_PDB),
                    hotspot_residues=["A1"],
                    output_dir=str(output_dir),
                )

                assert output_dir.exists()

    def test_build_command(self, tmp_path):
        """_build_command should return valid command list."""
        runner = RFdiffusionRunner()

        cmd = runner._build_command(
            target_pdb=str(MINI_PROTEIN_PDB),
            hotspot_residues=["A1", "A2"],
            output_dir=str(tmp_path),
            num_designs=5,
            binder_length=80,
        )

        assert isinstance(cmd, list)
        assert len(cmd) > 0
        # Should contain key arguments
        cmd_str = " ".join(cmd)
        assert "inference" in cmd_str.lower() or "run" in cmd_str.lower()

    def test_parse_hotspot_residues(self):
        """Should correctly parse hotspot residue strings."""
        runner = RFdiffusionRunner()

        # Format: "ChainResNum" like "A45", "B123"
        hotspots = ["A45", "A46", "A49"]
        parsed = runner._parse_hotspot_residues(hotspots)

        assert isinstance(parsed, str)
        # Should be formatted for RFdiffusion contig spec

    def test_parse_outputs(self, tmp_path):
        """_parse_outputs should find generated PDB files."""
        runner = RFdiffusionRunner()

        # Create mock output files
        for i in range(3):
            pdb_file = tmp_path / f"design_{i:04d}.pdb"
            pdb_file.write_text(f"ATOM      1  CA  ALA A   1       0.0   0.0   0.0  1.00  0.00\nEND\n")

        results = runner._parse_outputs(str(tmp_path))

        assert len(results) == 3
        for r in results:
            assert "id" in r
            assert "pdb_path" in r


class TestHotspotParsing:
    """Tests for hotspot residue parsing."""

    def test_single_hotspot(self):
        """Single hotspot should be parsed correctly."""
        runner = RFdiffusionRunner()
        result = runner._parse_hotspot_residues(["A45"])
        assert "45" in result or "A45" in result

    def test_multiple_hotspots_same_chain(self):
        """Multiple hotspots on same chain."""
        runner = RFdiffusionRunner()
        result = runner._parse_hotspot_residues(["A45", "A46", "A49"])
        assert isinstance(result, str)

    def test_hotspots_different_chains(self):
        """Hotspots on different chains."""
        runner = RFdiffusionRunner()
        result = runner._parse_hotspot_residues(["A45", "B10"])
        assert isinstance(result, str)


class TestEdgeCases:
    """Edge case tests."""

    def test_single_design(self, tmp_path):
        """Should work with num_designs=1."""
        runner = RFdiffusionRunner()

        with patch.object(runner, "_run_rfdiffusion"):
            with patch.object(runner, "_parse_outputs") as mock_parse:
                mock_parse.return_value = [{"id": "design_0000", "pdb_path": "test.pdb"}]

                # Should not raise
                cmd = runner._build_command(
                    target_pdb=str(MINI_PROTEIN_PDB),
                    hotspot_residues=["A1"],
                    output_dir=str(tmp_path),
                    num_designs=1,
                    binder_length=50,
                )
                assert isinstance(cmd, list)

    def test_short_binder(self, tmp_path):
        """Should handle short binder lengths."""
        runner = RFdiffusionRunner()

        cmd = runner._build_command(
            target_pdb=str(MINI_PROTEIN_PDB),
            hotspot_residues=["A1"],
            output_dir=str(tmp_path),
            num_designs=1,
            binder_length=20,  # Very short
        )
        assert isinstance(cmd, list)

    def test_long_binder(self, tmp_path):
        """Should handle long binder lengths."""
        runner = RFdiffusionRunner()

        cmd = runner._build_command(
            target_pdb=str(MINI_PROTEIN_PDB),
            hotspot_residues=["A1"],
            output_dir=str(tmp_path),
            num_designs=1,
            binder_length=200,  # Long binder
        )
        assert isinstance(cmd, list)

    @pytest.mark.asyncio
    async def test_many_designs(self, tmp_path):
        """Should handle large number of designs."""
        runner = RFdiffusionRunner()

        with patch.object(runner, "_run_rfdiffusion"):
            with patch.object(runner, "_parse_outputs") as mock_parse:
                mock_parse.return_value = [
                    {"id": f"design_{i:04d}", "pdb_path": f"design_{i}.pdb"}
                    for i in range(100)
                ]

                result = await runner.generate_backbones(
                    target_pdb=str(MINI_PROTEIN_PDB),
                    hotspot_residues=["A1"],
                    output_dir=str(tmp_path),
                    num_designs=100,
                )
                assert len(result) == 100
