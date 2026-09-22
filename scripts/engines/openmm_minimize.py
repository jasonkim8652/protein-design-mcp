"""Energy-minimise a structure with OpenMM. Runs inside the `md` environment.

Prints `key: value` lines the adapter parses, and writes the minimised
structure to the path given as the second argument, relative to the working
directory the dispatcher created.
"""

from __future__ import annotations

import argparse

from openmm import LangevinMiddleIntegrator, unit
from openmm.app import PDBFile, ForceField, Modeller, Simulation, HBonds, NoCutoff

FORCEFIELDS = {
    "amber14": ("amber14-all.xml", "amber14/tip3pfb.xml"),
    "charmm36": ("charmm36.xml", "charmm36/water.xml"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_pdb")
    parser.add_argument("output_pdb")
    parser.add_argument("--max-iterations", type=int, default=500)
    parser.add_argument("--forcefield", default="amber14", choices=sorted(FORCEFIELDS))
    args = parser.parse_args()

    pdb = PDBFile(args.input_pdb)
    forcefield = ForceField(*FORCEFIELDS[args.forcefield])
    modeller = Modeller(pdb.topology, pdb.positions)
    modeller.addHydrogens(forcefield)

    system = forcefield.createSystem(
        modeller.topology, nonbondedMethod=NoCutoff, constraints=HBonds
    )
    integrator = LangevinMiddleIntegrator(
        300 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds
    )
    simulation = Simulation(modeller.topology, system, integrator)
    simulation.context.setPositions(modeller.positions)

    initial = simulation.context.getState(
        getEnergy=True
    ).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

    simulation.minimizeEnergy(maxIterations=args.max_iterations)

    state = simulation.context.getState(getPositions=True, getEnergy=True)
    final = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

    with open(args.output_pdb, "w") as handle:
        PDBFile.writeFile(simulation.topology, state.getPositions(), handle)

    print(f"initial_potential_energy_kj_mol: {initial:.4f}")
    print(f"final_potential_energy_kj_mol: {final:.4f}")
    print(f"iterations: {args.max_iterations}")
    print(f"output_pdb: {args.output_pdb}")


if __name__ == "__main__":
    main()
