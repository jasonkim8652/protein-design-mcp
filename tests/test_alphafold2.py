"""Tests for AlphaFold2 pipeline runner using ColabFold - TDD RED phase first."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, mock_open
import asyncio
import json
import tempfile

import numpy as np
import pytest

from protein_design_mcp.pipelines.alphafold2 import (
    AlphaFold2Config,
    AlphaFold2Runner,
)
from protein_design_mcp.pipelines.esmfold import PredictionResult
from protein_design_mcp.exceptions import AlphaFold2Error


class TestAlphaFold2Config:
    """Tests for AlphaFold2Config dataclass."""

    def test_default_config(self):
        """Default config should have sensible values."""
        config = AlphaFold2Config()
        assert config.backend == "api"  # Default to API mode (no large databases)
        assert config.msa_mode == "mmseqs2"
        assert config.num_models == 1
        assert config.num_recycles == 3
        assert config.use_amber is False
        assert config.use_templates is False
        assert config.api_timeout == 600
        assert config.api_poll_interval == 5

    def test_local_backend_config(self):
        """Local backend config should be configurable."""
        config = AlphaFold2Config(
            backend="local",
            colabfold_path="/custom/path/colabfold_batch",
            msa_mode="mmseqs2",
        )
        assert config.backend == "local"
        assert config.colabfold_path == "/custom/path/colabfold_batch"

    def test_api_backend_config(self):
        """API backend config should be configurable."""
        config = AlphaFold2Config(
            backend="api",
            api_timeout=300,
            api_poll_interval=10,
        )
        assert config.backend == "api"
        assert config.api_timeout == 300
        assert config.api_poll_interval == 10

    def test_custom_config(self):
        """Custom config values should be set."""
        config = AlphaFold2Config(
            backend="local",
            colabfold_path="/custom/path/colabfold_batch",
            msa_mode="single_sequence",
            num_models=3,
            num_recycles=5,
            use_amber=True,
            use_templates=True,
        )
        assert config.colabfold_path == "/custom/path/colabfold_batch"
        assert config.msa_mode == "single_sequence"
        assert config.num_models == 3
        assert config.num_recycles == 5
        assert config.use_amber is True
        assert config.use_templates is True


class TestAlphaFold2Runner:
    """Tests for AlphaFold2Runner class."""

    def test_runner_init_default(self):
        """Runner should initialize with default config."""
        runner = AlphaFold2Runner()
        assert runner.config is not None

    def test_runner_init_custom_config(self):
        """Runner should accept custom config."""
        config = AlphaFold2Config(num_models=5)
        runner = AlphaFold2Runner(config=config)
        assert runner.config.num_models == 5

    @pytest.mark.asyncio
    async def test_predict_structure_returns_result(self, tmp_path):
        """predict_structure should return PredictionResult."""
        runner = AlphaFold2Runner()

        # Mock ColabFold execution
        with patch.object(runner, "_run_colabfold") as mock_run:
            # Simulate ColabFold output files
            mock_pdb = """ATOM      1  N   MET A   1       0.000   0.000   0.000  1.00 85.00           N
ATOM      2  CA  MET A   1       1.458   0.000   0.000  1.00 85.00           C
END
"""
            mock_scores = {
                "plddt": [85.0],
                "ptm": 0.82,
                "pae": [[0.5]],
            }

            async def mock_run_impl(fasta_path, output_dir, model_type):
                # Create mock output files
                pdb_file = Path(output_dir) / "test_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb"
                scores_file = Path(output_dir) / "test_scores_rank_001_alphafold2_ptm_model_1_seed_000.json"
                pdb_file.write_text(mock_pdb)
                scores_file.write_text(json.dumps(mock_scores))

            mock_run.side_effect = mock_run_impl

            result = await runner.predict_structure("M")

            assert isinstance(result, PredictionResult)
            assert result.sequence == "M"
            assert result.plddt > 0
            assert result.ptm > 0

    @pytest.mark.asyncio
    async def test_predict_structure_validates_sequence(self):
        """Invalid sequences should raise ValueError."""
        runner = AlphaFold2Runner()

        # Invalid character in sequence
        with pytest.raises(ValueError, match="[Ii]nvalid"):
            await runner.predict_structure("MKVGA123")

    @pytest.mark.asyncio
    async def test_predict_structure_empty_sequence(self):
        """Empty sequence should raise ValueError."""
        runner = AlphaFold2Runner()

        with pytest.raises(ValueError):
            await runner.predict_structure("")

    @pytest.mark.asyncio
    async def test_predict_structure_saves_pdb(self, tmp_path):
        """predict_structure should save PDB when output_pdb is specified."""
        runner = AlphaFold2Runner()

        pdb_content = """ATOM      1  N   MET A   1       0.000   0.000   0.000  1.00 85.00           N
ATOM      2  CA  MET A   1       1.458   0.000   0.000  1.00 85.00           C
END
"""
        with patch.object(runner, "_run_colabfold") as mock_run:
            async def mock_run_impl(fasta_path, output_dir, model_type):
                pdb_file = Path(output_dir) / "test_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb"
                scores_file = Path(output_dir) / "test_scores_rank_001_alphafold2_ptm_model_1_seed_000.json"
                pdb_file.write_text(pdb_content)
                scores_file.write_text(json.dumps({"plddt": [85.0], "ptm": 0.82, "pae": [[0.5]]}))

            mock_run.side_effect = mock_run_impl

            output_pdb = tmp_path / "prediction.pdb"
            result = await runner.predict_structure("M", output_pdb=str(output_pdb))

            assert output_pdb.exists()
            assert "ATOM" in output_pdb.read_text()

    @pytest.mark.asyncio
    async def test_predict_batch_returns_list(self, tmp_path):
        """predict_batch should return list of PredictionResults."""
        runner = AlphaFold2Runner()

        with patch.object(runner, "predict_structure") as mock_predict:
            mock_predict.return_value = PredictionResult(
                sequence="MKVGA",
                pdb_string="ATOM...",
                plddt=85.0,
                ptm=0.78,
                plddt_per_residue=np.array([80, 85, 90, 85, 80]),
                pae_matrix=None,
            )

            sequences = ["MKVGA", "AAAA", "GGGG"]
            results = await runner.predict_batch(sequences, str(tmp_path))

            assert len(results) == 3
            assert all(isinstance(r, PredictionResult) for r in results)


class TestAlphaFold2Multimer:
    """Tests for AlphaFold2 Multimer support."""

    @pytest.mark.asyncio
    async def test_predict_complex_returns_result(self, tmp_path):
        """predict_complex should return PredictionResult for multi-chain."""
        runner = AlphaFold2Runner()

        mock_pdb = """ATOM      1  CA  MET A   1       0.000   0.000   0.000  1.00 85.00           C
ATOM      2  CA  ALA B   1       5.000   0.000   0.000  1.00 82.00           C
END
"""
        mock_scores = {
            "plddt": [85.0, 82.0],
            "ptm": 0.75,
            "iptm": 0.70,
            "pae": [[0.5, 3.0], [3.0, 0.5]],
        }

        with patch.object(runner, "_run_colabfold") as mock_run:
            async def mock_run_impl(fasta_path, output_dir, model_type):
                assert model_type == "alphafold2_multimer_v3"
                pdb_file = Path(output_dir) / "test_unrelaxed_rank_001_alphafold2_multimer_v3_model_1_seed_000.pdb"
                scores_file = Path(output_dir) / "test_scores_rank_001_alphafold2_multimer_v3_model_1_seed_000.json"
                pdb_file.write_text(mock_pdb)
                scores_file.write_text(json.dumps(mock_scores))

            mock_run.side_effect = mock_run_impl

            result = await runner.predict_complex(["M", "A"])

            assert isinstance(result, PredictionResult)
            assert result.ptm > 0
            assert result.pae_matrix is not None

    @pytest.mark.asyncio
    async def test_predict_complex_creates_multimer_fasta(self, tmp_path):
        """predict_complex should create proper multi-sequence FASTA."""
        runner = AlphaFold2Runner()

        with patch.object(runner, "_run_colabfold") as mock_run:
            fasta_contents = []

            async def capture_fasta(fasta_path, output_dir, model_type):
                fasta_contents.append(Path(fasta_path).read_text())
                # Create minimal output
                pdb_file = Path(output_dir) / "test_unrelaxed_rank_001_alphafold2_multimer_v3_model_1_seed_000.pdb"
                scores_file = Path(output_dir) / "test_scores_rank_001_alphafold2_multimer_v3_model_1_seed_000.json"
                pdb_file.write_text("ATOM...")
                scores_file.write_text(json.dumps({"plddt": [85.0], "ptm": 0.8, "pae": [[0.5]]}))

            mock_run.side_effect = capture_fasta

            await runner.predict_complex(["MKVGA", "AAAA"])

            assert len(fasta_contents) == 1
            fasta = fasta_contents[0]
            # Multimer format uses colon separator
            assert ":" in fasta or (">chain_A" in fasta and ">chain_B" in fasta)

    @pytest.mark.asyncio
    async def test_predict_complex_requires_multiple_sequences(self):
        """predict_complex should require at least 2 sequences."""
        runner = AlphaFold2Runner()

        with pytest.raises(ValueError, match="at least 2"):
            await runner.predict_complex(["MKVGA"])


class TestColabFoldExecution:
    """Tests for ColabFold subprocess execution."""

    @pytest.mark.asyncio
    async def test_colabfold_command_construction(self):
        """_run_colabfold should construct correct command."""
        config = AlphaFold2Config(
            colabfold_path="/opt/colabfold/colabfold_batch",
            num_models=2,
            num_recycles=5,
            msa_mode="mmseqs2",
        )
        runner = AlphaFold2Runner(config=config)

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_process = MagicMock()
            mock_process.communicate = AsyncMock(return_value=(b"", b""))
            mock_process.returncode = 0
            mock_exec.return_value = mock_process

            await runner._run_colabfold("/tmp/input.fasta", "/tmp/output", "alphafold2_ptm")

            # Verify command construction
            call_args = mock_exec.call_args[0]
            assert "/opt/colabfold/colabfold_batch" in call_args
            assert "/tmp/input.fasta" in call_args
            assert "/tmp/output" in call_args
            assert "--num-models" in call_args
            assert "2" in call_args
            assert "--num-recycle" in call_args
            assert "5" in call_args

    @pytest.mark.asyncio
    async def test_colabfold_failure_raises_error(self):
        """ColabFold failure should raise AlphaFold2Error."""
        runner = AlphaFold2Runner()

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_process = MagicMock()
            mock_process.communicate = AsyncMock(return_value=(b"", b"Error: out of memory"))
            mock_process.returncode = 1
            mock_exec.return_value = mock_process

            with pytest.raises(AlphaFold2Error, match="ColabFold"):
                await runner._run_colabfold("/tmp/input.fasta", "/tmp/output", "alphafold2_ptm")


class TestSequenceValidation:
    """Tests for sequence validation in AlphaFold2Runner."""

    def test_valid_sequences(self):
        """Valid amino acid sequences should pass."""
        runner = AlphaFold2Runner()

        valid_sequences = [
            "MKVGA",
            "ACDEFGHIKLMNPQRSTVWY",
            "m",  # Single residue, lowercase
            "A" * 100,  # Long sequence
        ]

        for seq in valid_sequences:
            assert runner._validate_sequence(seq) is True

    def test_invalid_sequences(self):
        """Invalid sequences should be rejected."""
        runner = AlphaFold2Runner()

        invalid_sequences = [
            "",  # Empty
            "MKVGA1",  # Numbers
            "MKV GA",  # Space
            "MKVGA!",  # Special chars
        ]

        for seq in invalid_sequences:
            assert runner._validate_sequence(seq) is False


class TestOutputParsing:
    """Tests for ColabFold output parsing."""

    def test_parse_colabfold_scores(self, tmp_path):
        """Should correctly parse ColabFold JSON scores."""
        runner = AlphaFold2Runner()

        scores_data = {
            "plddt": [80.5, 85.2, 90.1, 75.3, 88.0],
            "ptm": 0.82,
            "pae": [
                [0.5, 1.2, 2.0, 3.5, 1.8],
                [1.2, 0.5, 1.0, 2.5, 1.5],
                [2.0, 1.0, 0.5, 2.0, 1.2],
                [3.5, 2.5, 2.0, 0.5, 2.2],
                [1.8, 1.5, 1.2, 2.2, 0.5],
            ],
        }

        scores_file = tmp_path / "scores.json"
        scores_file.write_text(json.dumps(scores_data))

        result = runner._parse_scores(str(scores_file))

        assert result["plddt"] == pytest.approx([80.5, 85.2, 90.1, 75.3, 88.0])
        assert result["ptm"] == pytest.approx(0.82)
        assert result["pae"] is not None
        assert len(result["pae"]) == 5

    def test_parse_colabfold_scores_with_iptm(self, tmp_path):
        """Should parse iptm score for multimer predictions."""
        runner = AlphaFold2Runner()

        scores_data = {
            "plddt": [80.5, 85.2],
            "ptm": 0.75,
            "iptm": 0.72,
            "pae": [[0.5, 3.0], [3.0, 0.5]],
        }

        scores_file = tmp_path / "scores.json"
        scores_file.write_text(json.dumps(scores_data))

        result = runner._parse_scores(str(scores_file))

        assert result["iptm"] == pytest.approx(0.72)


class TestEdgeCases:
    """Test edge cases for AlphaFold2Runner."""

    @pytest.mark.asyncio
    async def test_single_residue_prediction(self, tmp_path):
        """Single residue should still produce valid prediction."""
        runner = AlphaFold2Runner()

        with patch.object(runner, "_run_colabfold") as mock_run:
            async def mock_run_impl(fasta_path, output_dir, model_type):
                pdb_file = Path(output_dir) / "test_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb"
                scores_file = Path(output_dir) / "test_scores_rank_001_alphafold2_ptm_model_1_seed_000.json"
                pdb_file.write_text("ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 75.00           C\nEND")
                scores_file.write_text(json.dumps({"plddt": [75.0], "ptm": 0.5, "pae": [[0.5]]}))

            mock_run.side_effect = mock_run_impl

            result = await runner.predict_structure("A")
            assert len(result.plddt_per_residue) == 1

    @pytest.mark.asyncio
    async def test_plddt_range(self, tmp_path):
        """pLDDT scores should be in valid range [0, 100]."""
        runner = AlphaFold2Runner()

        with patch.object(runner, "_run_colabfold") as mock_run:
            async def mock_run_impl(fasta_path, output_dir, model_type):
                pdb_file = Path(output_dir) / "test_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb"
                scores_file = Path(output_dir) / "test_scores_rank_001_alphafold2_ptm_model_1_seed_000.json"
                pdb_file.write_text("ATOM...\nEND")
                scores_file.write_text(json.dumps({"plddt": [80, 85, 90, 85, 80], "ptm": 0.78, "pae": [[0.5]*5]*5}))

            mock_run.side_effect = mock_run_impl

            result = await runner.predict_structure("MKVGA")
            assert 0 <= result.plddt <= 100
            assert all(0 <= x <= 100 for x in result.plddt_per_residue)

    @pytest.mark.asyncio
    async def test_ptm_range(self, tmp_path):
        """pTM scores should be in valid range [0, 1]."""
        runner = AlphaFold2Runner()

        with patch.object(runner, "_run_colabfold") as mock_run:
            async def mock_run_impl(fasta_path, output_dir, model_type):
                pdb_file = Path(output_dir) / "test_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb"
                scores_file = Path(output_dir) / "test_scores_rank_001_alphafold2_ptm_model_1_seed_000.json"
                pdb_file.write_text("ATOM...\nEND")
                scores_file.write_text(json.dumps({"plddt": [80, 85, 90, 85, 80], "ptm": 0.78, "pae": [[0.5]*5]*5}))

            mock_run.side_effect = mock_run_impl

            result = await runner.predict_structure("MKVGA")
            assert 0 <= result.ptm <= 1


class TestRMSDCalculation:
    """Tests for RMSD calculation (reuses ESMFold implementation)."""

    def test_rmsd_identical_structures(self, tmp_path):
        """RMSD of identical structures should be 0."""
        runner = AlphaFold2Runner()

        pdb_content = """ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 50.00           C
ATOM      2  CA  GLY A   2       3.800   0.000   0.000  1.00 50.00           C
ATOM      3  CA  SER A   3       7.600   0.000   0.000  1.00 50.00           C
END
"""
        pdb1 = tmp_path / "struct1.pdb"
        pdb2 = tmp_path / "struct2.pdb"
        pdb1.write_text(pdb_content)
        pdb2.write_text(pdb_content)

        rmsd = runner.calculate_rmsd(str(pdb1), str(pdb2))
        assert rmsd == pytest.approx(0.0, abs=0.01)

    def test_rmsd_different_structures(self, tmp_path):
        """RMSD of different structures should be > 0."""
        runner = AlphaFold2Runner()

        pdb1_content = """ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 50.00           C
ATOM      2  CA  GLY A   2       3.800   0.000   0.000  1.00 50.00           C
END
"""
        pdb2_content = """ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 50.00           C
ATOM      2  CA  GLY A   2       3.800   5.000   0.000  1.00 50.00           C
END
"""
        pdb1 = tmp_path / "struct1.pdb"
        pdb2 = tmp_path / "struct2.pdb"
        pdb1.write_text(pdb1_content)
        pdb2.write_text(pdb2_content)

        rmsd = runner.calculate_rmsd(str(pdb1), str(pdb2))
        assert rmsd > 0
