"""
AlphaFold2 wrapper using ColabFold/LocalColabFold backend.

ColabFold provides fast AlphaFold2 predictions with MMseqs2 for MSA generation.
Supports both monomer (alphafold2_ptm) and multimer (alphafold2_multimer_v3) models.

Two backend modes are supported:
- "local": Uses local colabfold_batch with local MMseqs2 databases (fastest, requires setup)
- "api": Uses ColabFold server API for MSA generation (no local database needed)
"""

import asyncio
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urljoin

import numpy as np

from protein_design_mcp.exceptions import AlphaFold2Error
from protein_design_mcp.pipelines.esmfold import PredictionResult


logger = logging.getLogger(__name__)

# Valid amino acid characters
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

# ColabFold API endpoints
COLABFOLD_MSA_API = "https://api.colabfold.com"
COLABFOLD_TICKET_ENDPOINT = "/ticket/msa"
COLABFOLD_RESULT_ENDPOINT = "/result/msa"


@dataclass
class AlphaFold2Config:
    """Configuration for AlphaFold2/ColabFold predictions.

    Args:
        backend: Backend mode - "local" for local installation, "api" for ColabFold server API.
            "api" mode uses the ColabFold MMseqs2 server for MSA generation, avoiding the need
            for large local databases (~2TB). Requires internet connection.
        colabfold_path: Path to colabfold_batch executable (for local backend).
        msa_mode: MSA generation mode - "mmseqs2" (local), "single_sequence" (no MSA),
            or "server" (ColabFold API, same as backend="api").
        num_models: Number of AF2 models to use (1-5). More models = better quality but slower.
        num_recycles: Number of recycle iterations (1-48). More = potentially better quality.
        use_amber: Whether to use AMBER relaxation (slower but more accurate geometry).
        use_templates: Whether to search for structural templates.
        device: GPU device to use (e.g., "0", "cuda:0"). None for auto-detection.
        api_timeout: Timeout in seconds for API requests (for api backend).
        api_poll_interval: Polling interval in seconds when waiting for API results.
    """

    backend: Literal["local", "api"] = os.environ.get("COLABFOLD_BACKEND", "api")
    colabfold_path: str = os.environ.get(
        "COLABFOLD_PATH",
        "/opt/localcolabfold/colabfold-conda/bin/colabfold_batch"
    )
    msa_mode: str = "mmseqs2"  # "mmseqs2", "single_sequence", or "server"
    num_models: int = 1  # Number of AF2 models to use (1-5)
    num_recycles: int = 3  # Recycle iterations
    use_amber: bool = False  # Amber relaxation (slower but more accurate)
    use_templates: bool = False  # Template search
    device: str | None = None  # GPU device (e.g., "0" or "cuda:0")
    api_timeout: int = 600  # API timeout in seconds (10 minutes)
    api_poll_interval: int = 5  # Polling interval in seconds


class AlphaFold2Runner:
    """Wrapper for running AlphaFold2 predictions via ColabFold.

    Supports two backend modes:
    - "local": Full local ColabFold installation with colabfold_batch
    - "api": Uses ColabFold server API for MSA, then local model inference

    Example usage:
        # Using API backend (default, no local database needed)
        runner = AlphaFold2Runner()
        result = await runner.predict_structure("MKTVRQERLKSIVRILERSKEPVSGAQ")

        # Using local backend (requires full ColabFold setup)
        config = AlphaFold2Config(backend="local")
        runner = AlphaFold2Runner(config)
        result = await runner.predict_structure("MKTVRQERLKSIVRILERSKEPVSGAQ")
    """

    def __init__(self, config: AlphaFold2Config | None = None):
        """Initialize AlphaFold2 runner."""
        self.config = config or AlphaFold2Config()
        self._http_client = None

    async def _get_http_client(self):
        """Get or create HTTP client for API requests."""
        if self._http_client is None:
            try:
                import aiohttp
                self._http_client = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=self.config.api_timeout)
                )
            except ImportError:
                raise AlphaFold2Error(
                    "aiohttp is required for API backend. Install with: pip install aiohttp"
                )
        return self._http_client

    async def _close_http_client(self):
        """Close HTTP client if open."""
        if self._http_client is not None:
            await self._http_client.close()
            self._http_client = None

    def _validate_sequence(self, sequence: str) -> bool:
        """
        Validate amino acid sequence.

        Args:
            sequence: Amino acid sequence

        Returns:
            True if valid, False otherwise
        """
        if not sequence:
            return False

        # Convert to uppercase for validation
        seq_upper = sequence.upper()

        # Check all characters are valid amino acids
        return all(aa in VALID_AA for aa in seq_upper)

    async def _submit_msa_job(self, sequence: str) -> str:
        """
        Submit MSA job to ColabFold API server.

        Args:
            sequence: Amino acid sequence

        Returns:
            Job ticket ID

        Raises:
            AlphaFold2Error: If submission fails
        """
        client = await self._get_http_client()

        # ColabFold API expects the sequence in a specific format
        data = {
            "q": sequence,
            "mode": "all",  # Search all databases
        }

        url = urljoin(COLABFOLD_MSA_API, COLABFOLD_TICKET_ENDPOINT)

        try:
            async with client.post(url, data=data) as response:
                if response.status != 200:
                    text = await response.text()
                    raise AlphaFold2Error(
                        f"ColabFold API submission failed ({response.status}): {text}"
                    )
                result = await response.json()
                return result.get("id")
        except Exception as e:
            if isinstance(e, AlphaFold2Error):
                raise
            raise AlphaFold2Error(f"Failed to submit MSA job: {e}") from e

    async def _poll_msa_result(self, ticket_id: str) -> dict[str, Any]:
        """
        Poll for MSA job completion.

        Args:
            ticket_id: Job ticket ID from submission

        Returns:
            MSA result data

        Raises:
            AlphaFold2Error: If polling fails or times out
        """
        client = await self._get_http_client()
        url = urljoin(COLABFOLD_MSA_API, f"{COLABFOLD_RESULT_ENDPOINT}/{ticket_id}")

        start_time = time.time()
        while time.time() - start_time < self.config.api_timeout:
            try:
                async with client.get(url) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 202:
                        # Job still processing
                        logger.debug(f"MSA job {ticket_id} still processing...")
                        await asyncio.sleep(self.config.api_poll_interval)
                    else:
                        text = await response.text()
                        raise AlphaFold2Error(
                            f"ColabFold API error ({response.status}): {text}"
                        )
            except Exception as e:
                if isinstance(e, AlphaFold2Error):
                    raise
                raise AlphaFold2Error(f"Failed to poll MSA result: {e}") from e

        raise AlphaFold2Error(
            f"MSA job {ticket_id} timed out after {self.config.api_timeout}s"
        )

    async def _run_colabfold(
        self,
        fasta_path: str,
        output_dir: str,
        model_type: str = "alphafold2_ptm",
    ) -> None:
        """
        Run ColabFold subprocess.

        Args:
            fasta_path: Path to input FASTA file
            output_dir: Directory for output files
            model_type: Model type ("alphafold2_ptm" or "alphafold2_multimer_v3")

        Raises:
            AlphaFold2Error: If ColabFold execution fails
        """
        # Determine MSA mode based on backend
        if self.config.backend == "api":
            msa_mode = "server"  # Use ColabFold server for MSA
        else:
            msa_mode = self.config.msa_mode

        cmd = [
            self.config.colabfold_path,
            fasta_path,
            output_dir,
            "--num-models", str(self.config.num_models),
            "--num-recycle", str(self.config.num_recycles),
            "--msa-mode", msa_mode,
            "--model-type", model_type,
        ]

        if self.config.use_amber:
            cmd.append("--amber")

        if self.config.use_templates:
            cmd.append("--templates")

        if self.config.device:
            cmd.extend(["--gpu-device", self.config.device])

        logger.info(f"Running ColabFold: {' '.join(cmd)}")

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            raise AlphaFold2Error(
                f"ColabFold failed with exit code {process.returncode}: "
                f"{stderr.decode()}"
            )

    def _parse_scores(self, scores_path: str) -> dict[str, Any]:
        """
        Parse ColabFold scores JSON file.

        Args:
            scores_path: Path to scores JSON file

        Returns:
            Dictionary with plddt, ptm, pae, and optionally iptm
        """
        with open(scores_path, "r") as f:
            data = json.load(f)

        result = {
            "plddt": data.get("plddt", []),
            "ptm": data.get("ptm", 0.0),
            "pae": data.get("pae"),
        }

        # Add iptm for multimer predictions
        if "iptm" in data:
            result["iptm"] = data["iptm"]

        return result

    def _find_output_files(
        self,
        output_dir: str,
        model_type: str,
    ) -> tuple[str, str]:
        """
        Find ColabFold output PDB and scores files.

        Args:
            output_dir: Output directory
            model_type: Model type used

        Returns:
            Tuple of (pdb_path, scores_path)

        Raises:
            AlphaFold2Error: If output files not found
        """
        output_path = Path(output_dir)

        # ColabFold naming: *_unrelaxed_rank_001_*.pdb or *_relaxed_rank_001_*.pdb
        pdb_pattern = "*_rank_001_*.pdb"
        pdb_files = list(output_path.glob(pdb_pattern))

        # Prefer relaxed if available
        relaxed_files = [f for f in pdb_files if "relaxed" in f.name and "unrelaxed" not in f.name]
        unrelaxed_files = [f for f in pdb_files if "unrelaxed" in f.name]

        if relaxed_files:
            pdb_file = relaxed_files[0]
        elif unrelaxed_files:
            pdb_file = unrelaxed_files[0]
        elif pdb_files:
            pdb_file = pdb_files[0]
        else:
            raise AlphaFold2Error(f"No PDB output found in {output_dir}")

        # Find corresponding scores file
        scores_pattern = "*_scores_rank_001_*.json"
        scores_files = list(output_path.glob(scores_pattern))

        if not scores_files:
            raise AlphaFold2Error(f"No scores file found in {output_dir}")

        scores_file = scores_files[0]

        return str(pdb_file), str(scores_file)

    async def predict_structure(
        self,
        sequence: str,
        output_pdb: str | None = None,
    ) -> PredictionResult:
        """
        Predict structure for a protein sequence.

        Uses ColabFold with either local or API backend depending on configuration.

        Args:
            sequence: Amino acid sequence
            output_pdb: Optional path to save PDB file

        Returns:
            PredictionResult with structure and metrics

        Raises:
            ValueError: If sequence is invalid
            AlphaFold2Error: If prediction fails
        """
        # Validate sequence
        if not self._validate_sequence(sequence):
            raise ValueError(
                f"Invalid sequence. Must contain only valid amino acids: {VALID_AA}"
            )

        sequence = sequence.upper()

        # Create temporary directory for ColabFold
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write FASTA file
            fasta_path = Path(tmpdir) / "input.fasta"
            fasta_path.write_text(f">query\n{sequence}\n")

            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()

            # Run ColabFold
            await self._run_colabfold(
                str(fasta_path),
                str(output_dir),
                model_type="alphafold2_ptm",
            )

            # Find and parse output files
            pdb_path, scores_path = self._find_output_files(
                str(output_dir),
                "alphafold2_ptm",
            )

            pdb_string = Path(pdb_path).read_text()
            scores = self._parse_scores(scores_path)

            # Build PredictionResult
            plddt_per_residue = np.array(scores["plddt"])
            mean_plddt = float(np.mean(plddt_per_residue))

            pae_matrix = None
            if scores["pae"]:
                pae_matrix = np.array(scores["pae"])

            result = PredictionResult(
                sequence=sequence,
                pdb_string=pdb_string,
                plddt=mean_plddt,
                ptm=scores["ptm"],
                plddt_per_residue=plddt_per_residue,
                pae_matrix=pae_matrix,
            )

            # Save PDB if requested
            if output_pdb:
                output_path = Path(output_pdb)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(pdb_string)

            return result

    async def predict_batch(
        self,
        sequences: list[str],
        output_dir: str,
    ) -> list[PredictionResult]:
        """
        Predict structures for multiple sequences.

        Args:
            sequences: List of amino acid sequences
            output_dir: Directory to save PDB files

        Returns:
            List of PredictionResults
        """
        results = []
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for i, seq in enumerate(sequences):
            pdb_path = output_path / f"prediction_{i:04d}.pdb"
            result = await self.predict_structure(seq, output_pdb=str(pdb_path))
            results.append(result)

        return results

    async def predict_complex(
        self,
        sequences: list[str],
        output_pdb: str | None = None,
        chain_names: list[str] | None = None,
    ) -> PredictionResult:
        """
        Predict structure for a protein complex using AlphaFold2-Multimer.

        Args:
            sequences: List of amino acid sequences (one per chain)
            output_pdb: Optional path to save PDB file
            chain_names: Optional chain identifiers (default: A, B, C, ...)

        Returns:
            PredictionResult with complex structure and metrics

        Raises:
            ValueError: If fewer than 2 sequences provided or sequences invalid
            AlphaFold2Error: If prediction fails
        """
        if len(sequences) < 2:
            raise ValueError("predict_complex requires at least 2 sequences")

        # Validate all sequences
        for i, seq in enumerate(sequences):
            if not self._validate_sequence(seq):
                raise ValueError(
                    f"Invalid sequence at index {i}. "
                    f"Must contain only valid amino acids: {VALID_AA}"
                )

        sequences = [seq.upper() for seq in sequences]

        # Generate chain names if not provided
        if chain_names is None:
            chain_names = [chr(ord('A') + i) for i in range(len(sequences))]

        # Create temporary directory for ColabFold
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write multimer FASTA file (colon-separated sequences)
            fasta_path = Path(tmpdir) / "input.fasta"

            # ColabFold multimer format: sequences separated by colon
            # Or multi-entry FASTA where each entry is a chain
            fasta_content = f">complex\n{':'.join(sequences)}\n"
            fasta_path.write_text(fasta_content)

            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()

            # Run ColabFold with multimer model
            await self._run_colabfold(
                str(fasta_path),
                str(output_dir),
                model_type="alphafold2_multimer_v3",
            )

            # Find and parse output files
            pdb_path, scores_path = self._find_output_files(
                str(output_dir),
                "alphafold2_multimer_v3",
            )

            pdb_string = Path(pdb_path).read_text()
            scores = self._parse_scores(scores_path)

            # Build PredictionResult
            plddt_per_residue = np.array(scores["plddt"])
            mean_plddt = float(np.mean(plddt_per_residue))

            pae_matrix = None
            if scores["pae"]:
                pae_matrix = np.array(scores["pae"])

            # For multimer, use iptm if available, otherwise ptm
            ptm_score = scores.get("iptm", scores["ptm"])

            # Combined sequence for result
            combined_sequence = ":".join(sequences)

            result = PredictionResult(
                sequence=combined_sequence,
                pdb_string=pdb_string,
                plddt=mean_plddt,
                ptm=ptm_score,
                plddt_per_residue=plddt_per_residue,
                pae_matrix=pae_matrix,
            )

            # Save PDB if requested
            if output_pdb:
                output_path = Path(output_pdb)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(pdb_string)

            return result

    def calculate_rmsd(
        self,
        predicted_pdb: str,
        reference_pdb: str,
        align: bool = True,
    ) -> float:
        """
        Calculate RMSD between predicted and reference structures.

        Args:
            predicted_pdb: Path to predicted structure
            reference_pdb: Path to reference structure
            align: Whether to align structures before RMSD calculation

        Returns:
            RMSD value in Angstroms
        """
        # Extract CA coordinates from both structures
        pred_coords = self._get_ca_coordinates(predicted_pdb)
        ref_coords = self._get_ca_coordinates(reference_pdb)

        if len(pred_coords) != len(ref_coords):
            raise ValueError(
                f"Structure length mismatch: {len(pred_coords)} vs {len(ref_coords)}"
            )

        pred_coords = np.array(pred_coords)
        ref_coords = np.array(ref_coords)

        if align:
            # Kabsch alignment
            pred_coords = self._kabsch_align(pred_coords, ref_coords)

        # Calculate RMSD
        diff = pred_coords - ref_coords
        rmsd = np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))

        return float(rmsd)

    def _get_ca_coordinates(self, pdb_path: str) -> list[list[float]]:
        """Extract CA atom coordinates from PDB file."""
        coords = []
        with open(pdb_path, "r") as f:
            for line in f:
                if line.startswith("ATOM") and " CA " in line:
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    coords.append([x, y, z])
        return coords

    def _kabsch_align(
        self,
        mobile: np.ndarray,
        target: np.ndarray,
    ) -> np.ndarray:
        """
        Align mobile coordinates to target using Kabsch algorithm.

        Args:
            mobile: Coordinates to align (N x 3)
            target: Reference coordinates (N x 3)

        Returns:
            Aligned mobile coordinates
        """
        # Center both coordinate sets
        mobile_center = np.mean(mobile, axis=0)
        target_center = np.mean(target, axis=0)

        mobile_centered = mobile - mobile_center
        target_centered = target - target_center

        # Compute covariance matrix
        H = mobile_centered.T @ target_centered

        # SVD
        U, S, Vt = np.linalg.svd(H)

        # Compute rotation matrix
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        R = Vt.T @ np.diag([1, 1, d]) @ U.T

        # Apply rotation and translation
        aligned = (mobile_centered @ R.T) + target_center

        return aligned
