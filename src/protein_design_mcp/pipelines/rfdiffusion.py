"""
RFdiffusion wrapper for protein backbone generation.

RFdiffusion is a diffusion model for generating novel protein backbones
that can bind to target proteins.
"""

import asyncio
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from protein_design_mcp.exceptions import RFdiffusionError


@dataclass
class RFdiffusionConfig:
    """Configuration for RFdiffusion runs."""

    rfdiffusion_path: Path = Path(os.environ.get("RFDIFFUSION_PATH", "/opt/RFdiffusion"))
    python_path: str = os.environ.get("BIO_PYTHON_PATH", "python")
    num_designs: int = 10
    binder_length: int = 80
    noise_scale: float = 1.0
    diffusion_steps: int = 50


class RFdiffusionRunner:
    """Wrapper for running RFdiffusion."""

    def __init__(self, config: RFdiffusionConfig | None = None):
        """Initialize RFdiffusion runner."""
        self.config = config or RFdiffusionConfig()

    def _parse_hotspot_residues(self, hotspots: list[str]) -> str:
        """
        Parse hotspot residue strings into RFdiffusion contig format.

        Args:
            hotspots: List of residue strings like ["A45", "A46", "B10"]

        Returns:
            Formatted string for RFdiffusion hotspot specification
        """
        if not hotspots:
            return ""

        # Group by chain
        chain_residues: dict[str, list[int]] = {}
        for hs in hotspots:
            # Parse format: "A45" -> chain="A", resnum=45
            match = re.match(r"([A-Za-z])(\d+)", hs)
            if match:
                chain = match.group(1).upper()
                resnum = int(match.group(2))
                if chain not in chain_residues:
                    chain_residues[chain] = []
                chain_residues[chain].append(resnum)

        # Format for RFdiffusion: [A45,A46,A49]
        parts = []
        for chain, residues in sorted(chain_residues.items()):
            for res in sorted(residues):
                parts.append(f"{chain}{res}")

        return "[" + ",".join(parts) + "]"

    def _build_command(
        self,
        target_pdb: str,
        hotspot_residues: list[str],
        output_dir: str,
        num_designs: int,
        binder_length: int,
    ) -> list[str]:
        """
        Build RFdiffusion command line.

        Args:
            target_pdb: Path to target protein PDB
            hotspot_residues: Hotspot residues for binding
            output_dir: Output directory for designs
            num_designs: Number of designs to generate
            binder_length: Length of binder protein

        Returns:
            Command line as list of strings
        """
        script_path = self.config.rfdiffusion_path / "scripts" / "run_inference.py"

        # Build contig specification
        hotspot_str = self._parse_hotspot_residues(hotspot_residues)

        # Get target chain from first hotspot
        target_chain = hotspot_residues[0][0].upper() if hotspot_residues else "A"

        # Build gap-aware contig from actual PDB residues
        target_contig = self._build_chain_contig(target_pdb, target_chain)

        # Build hydra overrides
        overrides = [
            f"inference.output_prefix={output_dir}/design",
            f"inference.input_pdb={target_pdb}",
            f"inference.num_designs={num_designs}",
            f"contigmap.contigs=[{target_contig}/0 {binder_length}-{binder_length}]",
            f"ppi.hotspot_res={hotspot_str}",
            f"denoiser.noise_scale_ca={self.config.noise_scale}",
            f"denoiser.noise_scale_frame={self.config.noise_scale}",
            f"diffuser.T={self.config.diffusion_steps}",
        ]

        # Use a wrapper that disables JIT GPU fusion before running inference.
        # This avoids nvrtc compilation errors on newer GPUs (sm_89+) with
        # older CUDA toolkits (e.g. CUDA 11.1 + L40S).
        wrapper_code = (
            "import torch;"
            "torch._C._jit_override_can_fuse_on_gpu(False);"
            "torch._C._jit_set_profiling_executor(False);"
            "torch._C._jit_set_profiling_mode(False);"
            "import sys,runpy;"
            f"sys.argv=['{script_path}']+"
            f"{overrides!r};"
            f"runpy.run_path('{script_path}',run_name='__main__')"
        )

        cmd = [self.config.python_path, "-c", wrapper_code]

        return cmd

    @staticmethod
    def _build_chain_contig(pdb_path: str, chain: str) -> str:
        """Build gap-aware contig string from PDB residue numbering.

        E.g. if chain A has residues 19-29 and 43-127, returns 'A19-29/A43-127'.
        """
        resnums: set[int] = set()
        with open(pdb_path) as f:
            for line in f:
                if line.startswith("ATOM") and line[21] == chain:
                    resnums.add(int(line[22:26].strip()))
        if not resnums:
            raise ValueError(f"Chain {chain} not found in {pdb_path}")

        sorted_res = sorted(resnums)
        segments: list[str] = []
        seg_start = sorted_res[0]
        prev = sorted_res[0]
        for r in sorted_res[1:]:
            if r != prev + 1:
                segments.append(f"{chain}{seg_start}-{prev}")
                seg_start = r
            prev = r
        segments.append(f"{chain}{seg_start}-{prev}")
        return "/".join(segments)

    def _parse_outputs(self, output_dir: str) -> list[dict[str, Any]]:
        """
        Parse RFdiffusion output files.

        Args:
            output_dir: Directory containing output PDB files

        Returns:
            List of dictionaries with design info
        """
        output_path = Path(output_dir)
        results = []

        # Find all generated PDB files
        pdb_files = sorted(output_path.glob("design_*.pdb"))

        for pdb_file in pdb_files:
            # Extract design ID from filename
            design_id = pdb_file.stem

            results.append({
                "id": design_id,
                "pdb_path": str(pdb_file),
                "score": None,  # RFdiffusion doesn't output scores in PDB
            })

        return results

    async def _run_rfdiffusion(self, cmd: list[str], output_dir: str) -> None:
        """
        Execute RFdiffusion command.

        Args:
            cmd: Command line arguments
            output_dir: Output directory

        Raises:
            RFdiffusionError: If execution fails
        """
        try:
            env = os.environ.copy()
            env["RFDIFFUSION_PATH"] = str(self.config.rfdiffusion_path)
            # Disable PyTorch JIT GPU fusion to avoid nvrtc compilation
            # errors on newer GPUs (sm_89+) with older CUDA toolkits
            env["PYTORCH_JIT_USE_NNC_NOT_NVFUSER"] = "1"

            # Run asynchronously
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.config.rfdiffusion_path),
                env=env,
            )

            # 30 min timeout per design run (can be slow for long binders)
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=1800
                )
            except asyncio.TimeoutError:
                process.kill()
                raise RFdiffusionError(
                    "RFdiffusion timed out after 30 minutes"
                )

            if process.returncode != 0:
                raise RFdiffusionError(
                    f"RFdiffusion failed with code {process.returncode}: "
                    f"{stderr.decode()}"
                )

        except FileNotFoundError as e:
            raise RFdiffusionError(
                f"RFdiffusion not found. Ensure RFDIFFUSION_PATH is set correctly. "
                f"Current path: {self.config.rfdiffusion_path}"
            ) from e

    async def generate_backbones(
        self,
        target_pdb: str,
        hotspot_residues: list[str],
        output_dir: str,
        num_designs: int | None = None,
        binder_length: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Generate binder backbones using RFdiffusion.

        Args:
            target_pdb: Path to target protein PDB
            hotspot_residues: Residues for binder interface (e.g., ["A45", "A46"])
            output_dir: Directory to save generated structures
            num_designs: Number of designs (uses config default if None)
            binder_length: Binder length (uses config default if None)

        Returns:
            List of dictionaries containing backbone info:
            - id: Design identifier
            - pdb_path: Path to generated PDB file
            - score: Optional confidence score

        Raises:
            FileNotFoundError: If target PDB doesn't exist
            ValueError: If hotspot_residues is empty
            RFdiffusionError: If generation fails
        """
        # Validate inputs
        target_path = Path(target_pdb)
        if not target_path.exists():
            raise FileNotFoundError(f"Target PDB not found: {target_pdb}")

        if not hotspot_residues:
            raise ValueError("Hotspot residues cannot be empty")

        # Use config defaults if not specified
        num_designs = num_designs or self.config.num_designs
        binder_length = binder_length or self.config.binder_length

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Build and run command
        cmd = self._build_command(
            target_pdb=str(target_path.absolute()),
            hotspot_residues=hotspot_residues,
            output_dir=str(output_path.absolute()),
            num_designs=num_designs,
            binder_length=binder_length,
        )

        await self._run_rfdiffusion(cmd, str(output_path))

        # Parse and return results
        return self._parse_outputs(str(output_path))
