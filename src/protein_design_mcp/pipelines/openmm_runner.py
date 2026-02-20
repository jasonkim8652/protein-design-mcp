"""
OpenMM energy minimization runner.

Performs all-atom energy minimization with AMBER14 force field and
optional implicit solvent (GBn2).  Useful for refining predicted or
designed protein structures.
"""

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OpenMMConfig:
    """Configuration for OpenMM minimization."""

    force_field: str = "amber14-all.xml"
    solvent: str = "implicit"  # "implicit" or "none"
    implicit_solvent_model: str = "implicit/gbn2.xml"
    max_iterations: int = 500
    tolerance_kj: float = 10.0  # kJ/mol/nm


class OpenMMRunner:
    """Energy minimization using OpenMM with AMBER14 force field."""

    def __init__(self, config: OpenMMConfig | None = None):
        self.config = config or OpenMMConfig()

    async def minimize(
        self,
        pdb_path: str,
        output_path: str | None = None,
        num_steps: int | None = None,
    ) -> dict[str, Any]:
        """Energy-minimize a PDB structure.

        Args:
            pdb_path: Path to input PDB file.
            output_path: Where to write minimized PDB.  Auto-generated if None.
            num_steps: Override max minimization iterations.

        Returns:
            Dict with ``minimized_pdb_path``, ``initial_energy``, ``final_energy``,
            ``energy_change``, ``rmsd_from_initial``, ``num_iterations``.
        """
        try:
            from openmm import app, unit, LangevinMiddleIntegrator
            from openmm.app import PDBFile, ForceField, Modeller, Simulation
        except ImportError:
            raise RuntimeError(
                "OpenMM is required for energy minimization. "
                "Install with: pip install openmm"
            )

        pdb_path = str(pdb_path)
        max_iter = num_steps if num_steps is not None else self.config.max_iterations

        # Load structure
        pdb = PDBFile(pdb_path)

        # Set up force field
        ff_xmls = [self.config.force_field]
        if self.config.solvent == "implicit":
            ff_xmls.append(self.config.implicit_solvent_model)
        forcefield = ForceField(*ff_xmls)

        # Add missing hydrogens / heavy atoms
        modeller = Modeller(pdb.topology, pdb.positions)
        modeller.addHydrogens(forcefield)

        # Create system
        system = forcefield.createSystem(
            modeller.topology,
            nonbondedCutoff=1.0 * unit.nanometers,
            constraints=app.HBonds,
        )

        # Integrator (only used for system setup, not MD)
        integrator = LangevinMiddleIntegrator(
            300 * unit.kelvin, 1.0 / unit.picoseconds, 0.002 * unit.picoseconds
        )

        simulation = Simulation(modeller.topology, system, integrator)
        simulation.context.setPositions(modeller.positions)

        # Get initial energy
        state_initial = simulation.context.getState(getEnergy=True, getPositions=True)
        initial_energy_kj = state_initial.getPotentialEnergy().value_in_unit(
            unit.kilojoules_per_mole
        )
        initial_positions = state_initial.getPositions(asNumpy=True).value_in_unit(
            unit.nanometers
        )

        # Minimize
        simulation.minimizeEnergy(
            maxIterations=max_iter,
            tolerance=self.config.tolerance_kj * unit.kilojoules_per_mole / unit.nanometers,
        )

        # Get final energy
        state_final = simulation.context.getState(getEnergy=True, getPositions=True)
        final_energy_kj = state_final.getPotentialEnergy().value_in_unit(
            unit.kilojoules_per_mole
        )
        final_positions = state_final.getPositions(asNumpy=True).value_in_unit(
            unit.nanometers
        )

        # RMSD (all heavy atoms with matching indices)
        rmsd = float(np.sqrt(np.mean(np.sum(
            (final_positions - initial_positions) ** 2, axis=1
        )))) * 10.0  # nm -> Angstrom

        # Write output PDB
        if output_path is None:
            fd = tempfile.NamedTemporaryFile(
                suffix=".pdb", prefix="minimized_", delete=False
            )
            output_path = fd.name
            fd.close()

        output_path = str(output_path)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            PDBFile.writeFile(
                simulation.topology,
                state_final.getPositions(),
                f,
            )

        return {
            "minimized_pdb_path": output_path,
            "initial_energy": round(initial_energy_kj, 2),
            "final_energy": round(final_energy_kj, 2),
            "energy_change": round(final_energy_kj - initial_energy_kj, 2),
            "rmsd_from_initial": round(rmsd, 3),
            "num_iterations": max_iter,
            "force_field": self.config.force_field,
            "solvent_model": self.config.solvent,
        }
