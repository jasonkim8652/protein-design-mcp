"""
Boltz wrapper for structure prediction and affinity scoring.

Boltz is a fast structure prediction model (alternative to AlphaFold2/ESMFold).
This runner invokes the Boltz CLI via subprocess, supporting cross-env
execution through ``conda run -n boltz``.

Two operations:
- **predict_structure**: Predict the 3D structure of a protein from sequence.
- **predict_affinity**: Predict binding affinity for a protein complex.
"""

import asyncio
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from protein_design_mcp.exceptions import BoltzError

logger = logging.getLogger(__name__)


@dataclass
class BoltzConfig:
    """Configuration for Boltz predictions."""

    conda_env: str | None = "boltz"
    model: str = "boltz2"
    num_samples: int = 1
    timeout: int = 600  # 10 minutes
    output_format: str | None = None  # e.g. "pdb"; None = Boltz default (mmCIF)
    use_msa_server: bool = True  # auto-fetch MSAs from ColabFold API
    no_kernels: bool = False  # disable cuequivariance kernels (fallback to PyTorch)


class BoltzRunner:
    """Wrapper for running Boltz structure prediction via CLI."""

    def __init__(self, config: BoltzConfig | None = None):
        self.config = config or BoltzConfig()

    async def predict_structure(
        self,
        sequence: str,
        model: str | None = None,
        num_samples: int | None = None,
    ) -> dict[str, Any]:
        """Predict the 3D structure of a protein sequence using Boltz.

        Args:
            sequence: Amino acid sequence.
            model: Model name (default: boltz2).
            num_samples: Number of structure samples.

        Returns:
            Dict with pdb_path, plddt, ptm, and per-residue confidence.
        """
        model = model or self.config.model
        n_samples = num_samples or self.config.num_samples

        # Validate sequence
        valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
        seq_upper = sequence.upper().strip()
        invalid = set(seq_upper) - valid_aa
        if invalid:
            raise BoltzError(f"Invalid amino acid characters: {invalid}")

        # Create input YAML for Boltz
        work_dir = tempfile.mkdtemp(prefix="boltz_")
        input_yaml = Path(work_dir) / "input.yaml"
        input_yaml.write_text(
            f"sequences:\n"
            f"  - protein:\n"
            f"      id: A\n"
            f"      sequence: {seq_upper}\n"
        )

        output_dir = Path(work_dir) / "output"
        output_dir.mkdir()

        # Build command
        cmd = self._build_predict_command(
            input_path=str(input_yaml),
            output_dir=str(output_dir),
            model=model,
            num_samples=n_samples,
        )

        # Run Boltz
        result = await self._run_boltz(cmd, work_dir)

        # Parse outputs
        return self._parse_structure_output(output_dir, seq_upper, result)

    async def predict_affinity(
        self,
        sequences: list[str],
        model: str | None = None,
    ) -> dict[str, Any]:
        """Predict binding affinity for a protein complex using Boltz.

        Args:
            sequences: List of amino acid sequences (one per chain).
            model: Model name (default: boltz2).

        Returns:
            Dict with affinity_score, pdb_path, confidence metrics.
        """
        model = model or self.config.model

        if len(sequences) < 2:
            raise BoltzError("At least 2 sequences required for affinity prediction")

        # Validate sequences
        valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
        clean_seqs = []
        for i, seq in enumerate(sequences):
            seq_upper = seq.upper().strip()
            invalid = set(seq_upper) - valid_aa
            if invalid:
                raise BoltzError(f"Invalid characters in sequence {i}: {invalid}")
            clean_seqs.append(seq_upper)

        # Create input YAML for multimer
        work_dir = tempfile.mkdtemp(prefix="boltz_affinity_")
        input_yaml = Path(work_dir) / "input.yaml"

        chain_ids = [chr(65 + i) for i in range(len(clean_seqs))]  # A, B, C, ...
        yaml_content = "sequences:\n"
        for chain_id, seq in zip(chain_ids, clean_seqs):
            yaml_content += (
                f"  - protein:\n"
                f"      id: {chain_id}\n"
                f"      sequence: {seq}\n"
            )

        input_yaml.write_text(yaml_content)

        output_dir = Path(work_dir) / "output"
        output_dir.mkdir()

        # Build command with affinity flag
        cmd = self._build_predict_command(
            input_path=str(input_yaml),
            output_dir=str(output_dir),
            model=model,
            num_samples=1,
            affinity=True,
        )

        # Run Boltz
        result = await self._run_boltz(cmd, work_dir)

        # Parse outputs
        return self._parse_affinity_output(output_dir, clean_seqs, result)

    def _build_predict_command(
        self,
        input_path: str,
        output_dir: str,
        model: str,
        num_samples: int,
        affinity: bool = False,
    ) -> list[str]:
        """Build the Boltz CLI command.

        When ``conda_env`` is set, prefixes with ``conda run -n <env>``.
        When ``conda_env`` is None or empty (e.g. inside Docker), invokes
        ``boltz predict`` directly.
        """
        if self.config.conda_env:
            cmd = [
                "conda", "run", "-n", self.config.conda_env, "--no-banner",
                "boltz", "predict",
                input_path,
                "--out_dir", output_dir,
                "--diffusion_samples", str(num_samples),
            ]
        else:
            cmd = [
                "boltz", "predict",
                input_path,
                "--out_dir", output_dir,
                "--diffusion_samples", str(num_samples),
            ]

        if model and model not in ("boltz2", "boltz"):
            cmd.extend(["--model", model])

        # Boltz 2 computes affinity automatically for multimer inputs.
        # No --affinity flag needed; just pass --diffusion_samples_affinity if desired.

        if self.config.output_format:
            cmd.extend(["--output_format", self.config.output_format])

        if self.config.use_msa_server:
            cmd.append("--use_msa_server")

        if self.config.no_kernels:
            cmd.append("--no_kernels")

        return cmd

    async def _run_boltz(
        self,
        cmd: list[str],
        work_dir: str,
    ) -> dict[str, Any]:
        """Execute a Boltz command via subprocess.

        Returns:
            Dict with stdout, stderr, and return code.
        """
        logger.info("Running Boltz: %s", " ".join(cmd))

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=work_dir,
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=self.config.timeout,
            )

            stdout_str = stdout.decode("utf-8", errors="replace")
            stderr_str = stderr.decode("utf-8", errors="replace")

            if proc.returncode != 0:
                raise BoltzError(
                    f"Boltz exited with code {proc.returncode}. "
                    f"stderr: {stderr_str[:500]}"
                )

            return {
                "stdout": stdout_str,
                "stderr": stderr_str,
                "returncode": proc.returncode,
            }
        except asyncio.TimeoutError:
            raise BoltzError(
                f"Boltz timed out after {self.config.timeout}s"
            )
        except FileNotFoundError:
            raise BoltzError(
                "Boltz CLI not found. Install with: "
                "conda create -n boltz python=3.11 && "
                "conda activate boltz && pip install boltz"
            )

    def _parse_structure_output(
        self,
        output_dir: Path,
        sequence: str,
        run_result: dict[str, Any],
    ) -> dict[str, Any]:
        """Parse Boltz structure prediction output files.

        Boltz outputs mmCIF/PDB files and confidence scores in the
        output directory.
        """
        # Look for output structure files
        pdb_files = list(output_dir.rglob("*.pdb"))
        cif_files = list(output_dir.rglob("*.cif"))

        structure_path = None
        if pdb_files:
            structure_path = str(pdb_files[0])
        elif cif_files:
            structure_path = str(cif_files[0])

        # Look for confidence JSON
        confidence = {}
        json_files = list(output_dir.rglob("*confidence*.json"))
        if not json_files:
            json_files = list(output_dir.rglob("*.json"))

        for jf in json_files:
            try:
                data = json.loads(jf.read_text())
                if isinstance(data, dict):
                    confidence.update(data)
            except (json.JSONDecodeError, OSError):
                continue

        plddt = confidence.get("plddt", confidence.get("mean_plddt"))
        ptm = confidence.get("ptm", confidence.get("pTM"))

        # Parse pLDDT from PDB B-factor if not in JSON
        if plddt is None and structure_path and structure_path.endswith(".pdb"):
            plddt = self._extract_plddt_from_pdb(structure_path)

        result: dict[str, Any] = {
            "sequence": sequence,
            "num_residues": len(sequence),
            "model": self.config.model,
        }

        if structure_path:
            result["predicted_structure_pdb"] = structure_path
        if plddt is not None:
            result["plddt"] = round(float(plddt), 2)
        if ptm is not None:
            result["ptm"] = round(float(ptm), 4)
        if confidence.get("plddt_per_residue"):
            result["plddt_per_residue"] = confidence["plddt_per_residue"]

        return result

    def _parse_affinity_output(
        self,
        output_dir: Path,
        sequences: list[str],
        run_result: dict[str, Any],
    ) -> dict[str, Any]:
        """Parse Boltz affinity prediction output files."""
        # Look for output files
        pdb_files = list(output_dir.rglob("*.pdb"))
        cif_files = list(output_dir.rglob("*.cif"))

        structure_path = None
        if pdb_files:
            structure_path = str(pdb_files[0])
        elif cif_files:
            structure_path = str(cif_files[0])

        # Look for affinity/confidence JSON
        confidence = {}
        json_files = list(output_dir.rglob("*.json"))
        for jf in json_files:
            try:
                data = json.loads(jf.read_text())
                if isinstance(data, dict):
                    confidence.update(data)
            except (json.JSONDecodeError, OSError):
                continue

        result: dict[str, Any] = {
            "sequences": sequences,
            "num_chains": len(sequences),
            "model": self.config.model,
        }

        if structure_path:
            result["predicted_structure_pdb"] = structure_path

        affinity = confidence.get("affinity", confidence.get("binding_affinity"))
        if affinity is not None:
            result["affinity_score"] = round(float(affinity), 4)

        plddt = confidence.get("plddt", confidence.get("mean_plddt"))
        if plddt is not None:
            result["plddt"] = round(float(plddt), 2)

        iptm = confidence.get("iptm", confidence.get("ipTM"))
        if iptm is not None:
            result["iptm"] = round(float(iptm), 4)

        ptm = confidence.get("ptm", confidence.get("pTM"))
        if ptm is not None:
            result["ptm"] = round(float(ptm), 4)

        return result

    @staticmethod
    def _extract_plddt_from_pdb(pdb_path: str) -> float | None:
        """Extract mean pLDDT from PDB B-factor column."""
        b_factors = []
        try:
            with open(pdb_path) as f:
                for line in f:
                    if line.startswith(("ATOM  ", "HETATM")):
                        # B-factor is columns 61-66
                        try:
                            bf = float(line[60:66].strip())
                            b_factors.append(bf)
                        except (ValueError, IndexError):
                            continue
        except OSError:
            return None

        if not b_factors:
            return None

        return sum(b_factors) / len(b_factors)
