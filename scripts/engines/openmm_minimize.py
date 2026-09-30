"""Fixed staged vacuum minimization, with explicit runtime and replay artifacts."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import openmm as mm
from openmm import unit
from openmm.app import PDBFile, ForceField, Modeller, Simulation, HBonds, NoCutoff
from pdbfixer import PDBFixer

FORCEFIELDS = {
    'amber14': ('amber14-all.xml', 'amber14/tip3pfb.xml'),
    'charmm36': ('charmm36.xml', 'charmm36/water.xml'),
}
PROTOCOL = 'soft-repulsion-flexible-hbonds-v1'
SEED = 20260930


class NumericalFailure(ValueError):
    """Coordinates or energy cannot support a physical measurement."""



def platform_settings(name, precision):
    platform = mm.Platform.getPlatformByName(name)
    properties = {'Precision': precision, 'DeterministicForces': 'true'} if name == 'CUDA' else {}
    return platform, properties


def final_system(forcefield, topology):
    return forcefield.createSystem(topology, nonbondedMethod=NoCutoff, constraints=HBonds)


def soft_system(forcefield, topology, positions):
    """Bound overlap forces while retaining flexible bonded force-field terms."""
    system = forcefield.createSystem(topology, nonbondedMethod=NoCutoff, constraints=None, rigidWater=False)
    index, nb = next((i, f) for i, f in enumerate(system.getForces()) if isinstance(f, mm.NonbondedForce))
    soft = mm.CustomNonbondedForce('0.5*k*max(0,0.5*(sigma1+sigma2)-r)^2')
    soft.addGlobalParameter('k', 10000)
    soft.addPerParticleParameter('sigma')
    soft.setNonbondedMethod(mm.CustomNonbondedForce.NoCutoff)
    sigmas = [nb.getParticleParameters(i)[1].value_in_unit(unit.nanometer) for i in range(nb.getNumParticles())]
    remove = [index]
    # CHARMM expresses its Lennard-Jones terms through tabulated type pairs,
    # with dummy sigma=1 in NonbondedForce, and a separate 1-4 CustomBondForce.
    for i, force in enumerate(system.getForces()):
        if isinstance(force, mm.CustomNonbondedForce):
            if [force.getTabulatedFunctionName(j) for j in range(force.getNumTabulatedFunctions())] != ['acoef', 'bcoef']:
                raise ValueError('Unsupported custom nonbonded force in soft preparation')
            width, _, avec = force.getTabulatedFunction(0).getFunctionParameters()
            _, _, bvec = force.getTabulatedFunction(1).getFunctionParameters()
            for j in range(force.getNumParticles()):
                kind = int(force.getParticleParameters(j)[0])
                a, b = avec[kind+width*kind], bvec[kind+width*kind]
                sigmas[j] = (a/b)**(1/6) if a > 0 and b > 0 else 0.0
            remove.append(i)
        if isinstance(force, mm.CustomBondForce) and force.getEnergyFunction() == '4*epsilon*((sigma/r)^12-(sigma/r)^6)':
            remove.append(i)
    for sigma in sigmas:
        soft.addParticle([sigma])
    for i in range(nb.getNumExceptions()):
        a, b, *_ = nb.getExceptionParameters(i)
        soft.addExclusion(a, b)
    for index in sorted(remove, reverse=True):
        system.removeForce(index)
    system.addForce(soft)
    restraint = mm.CustomExternalForce('0.5*krest*((x-x0)^2+(y-y0)^2+(z-z0)^2)')
    restraint.addGlobalParameter('krest', 1000)
    for key in ('x0', 'y0', 'z0'):
        restraint.addPerParticleParameter(key)
    xyz = positions.value_in_unit(unit.nanometer)
    for atom in topology.atoms():
        if atom.element is not None and atom.element.symbol not in ('H', 'D'):
            restraint.addParticle(atom.index, xyz[atom.index])
    system.addForce(restraint)
    return system


class Reporter(mm.MinimizationReporter):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.passes = 0
        self.last = None

    def report(self, iteration, x, grad, args):
        if self.last is None or iteration <= self.last['iteration']:
            self.passes += 1
        self.calls += 1
        self.last = {'iteration': int(iteration), **dict(args)}
        return False


def geometry(topology, positions):
    xyz = np.asarray(positions.value_in_unit(unit.angstrom))
    if not np.isfinite(xyz).all():
        return {'finite_coordinates': False, 'passed': False}
    atoms = list(topology.atoms())
    heavy = [a.index for a in atoms if a.element is not None and a.element.symbol not in ('H', 'D')]
    contacts, closest = 0, None
    for n, i in enumerate(heavy[:-1]):
        distances = np.linalg.norm(xyz[heavy[n+1:]]-xyz[i], axis=1)
        contacts += int(np.sum(distances < 1.0))
        closest = float(distances.min()) if closest is None else min(closest, float(distances.min()))
    distorted = []
    for residue in topology.residues():
        names = {a.name: a.index for a in residue.atoms()}
        for a, b, low, high in [('N', 'CA', 1.1, 2.0), ('CA', 'C', 1.1, 2.0), ('C', 'O', 1.0, 1.8)]:
            if a in names and b in names:
                distance = float(np.linalg.norm(xyz[names[a]]-xyz[names[b]]))
                if not low <= distance <= high:
                    distorted.append({'chain': residue.chain.id, 'residue': residue.id, 'atoms': [a,b], 'distance_A': distance})
    return {'finite_coordinates': True, 'heavy_contacts_lt_1A': contacts,
            'closest_heavy_A': closest, 'distorted_backbone_bonds': distorted,
            'passed': contacts == 0 and not distorted}


def save_positions(path, positions):
    np.save(path, np.asarray(positions.value_in_unit(unit.nanometer), dtype=np.float64))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_stage(name, topology, system, positions, platform, properties, cap, directory, base_positions):
    xml_path = directory / (name + '.system.xml')
    xml_path.write_text(mm.XmlSerializer.serialize(system))
    integrator = mm.VerletIntegrator(0.001 * unit.picoseconds)
    simulation = Simulation(topology, system, integrator, platform, properties)
    context = simulation.context
    context.setPositions(positions)
    before = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    reporter = Reporter()
    started = time.monotonic()
    simulation.minimizeEnergy(tolerance=10 * unit.kilojoule_per_mole / unit.nanometer,
                              maxIterations=cap, reporter=reporter)
    state = context.getState(getPositions=True, getEnergy=True, getForces=True)
    after_positions = state.getPositions()
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
    xyz = np.asarray(after_positions.value_in_unit(unit.nanometer))
    base = np.asarray(base_positions.value_in_unit(unit.nanometer))
    heavy = [a.index for a in topology.atoms() if a.element is not None and a.element.symbol not in ('H', 'D')]
    displacements = np.linalg.norm(xyz[heavy]-base[heavy], axis=1) * 10
    constraint_errors = []
    for i in range(system.getNumConstraints()):
        a, b, length = system.getConstraintParameters(i)
        expected = length.value_in_unit(unit.nanometer)
        constraint_errors.append(abs(np.linalg.norm(xyz[a]-xyz[b])-expected)/expected)
    row = {'name': name, 'max_iterations_per_pass': cap, 'reporter_calls': reporter.calls,
           'observed_passes': reporter.passes, 'last_report': reporter.last,
           'elapsed_seconds': time.monotonic()-started, 'platform': context.getPlatform().getName(),
           'platform_properties': {p: platform.getPropertyValue(context, p) for p in platform.getPropertyNames()},
           'initial_energy_kj_mol': float(before), 'final_energy_kj_mol': float(energy),
           'force_rms_kj_mol_nm': float(np.sqrt(np.mean(forces**2))),
           'max_constraint_relative_error': float(max(constraint_errors, default=0)),
           'geometry': geometry(topology, after_positions),
           'heavy_displacement_from_prepared_A': {'rms': float(np.sqrt(np.mean(displacements**2))), 'max': float(displacements.max(initial=0))},
           'system_file': str(xml_path), 'system_sha256': hashlib.sha256(xml_path.read_bytes()).hexdigest(),
           'positions_file': str(directory / (name+'.positions.npy')),
           'positions_sha256': save_positions(directory/(name+'.positions.npy'), after_positions)}
    row['finite_energy_and_forces'] = bool(np.isfinite(energy) and np.isfinite(forces).all())
    return after_positions, row


def execute(args, diagnostics):
    platform, properties = platform_settings(args.platform, args.precision)
    path = Path(args.input_pdb)
    diagnostics['input_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    compressed = path.suffix.lower() == '.gz'
    suffix = path.with_suffix('').suffix.lower() if compressed else path.suffix.lower()
    opener = gzip.open if compressed else open
    with opener(path, 'rt') as handle:
        pdb = PDBFixer(pdbxfile=handle, platform=mm.Platform.getPlatformByName('Reference')) if suffix in {'.cif', '.mmcif'} else PDBFixer(pdbfile=handle, platform=mm.Platform.getPlatformByName('Reference'))
    diagnostics['raw_geometry'] = geometry(pdb.topology, pdb.positions)
    pdb.findMissingResidues()
    pdb.missingResidues = {}
    pdb.findMissingAtoms()
    if pdb.missingAtoms:
        raise ValueError('Input is missing non-terminal heavy atoms; supply a complete structure')
    added_terminal_atoms = sum(len(atoms) for atoms in pdb.missingTerminals.values())
    if added_terminal_atoms:
        pdb.addMissingAtoms(seed=SEED)
    forcefield = ForceField(*FORCEFIELDS[args.forcefield])
    modeller = Modeller(pdb.topology, pdb.positions)
    random.seed(SEED)
    np.random.seed(SEED)
    variants = modeller.addHydrogens(platform=mm.Platform.getPlatformByName('Reference'))
    positions = modeller.positions
    directory = Path('openmm_states')
    directory.mkdir(exist_ok=True)
    with (directory/'prepared.pdb').open('w') as handle:
        PDBFile.writeFile(modeller.topology, positions, handle, keepIds=True)
    diagnostics.update(added_terminal_atoms=added_terminal_atoms,
        hydrogen_preparation={'method': 'OpenMM simplified geometry (forcefield=None)', 'platform': 'Reference', 'seed': SEED, 'variants': variants},
        prepared_geometry=geometry(modeller.topology, positions),
        prepared_positions_file=str(directory/'prepared.positions.npy'),
        prepared_topology_file=str(directory/'prepared.pdb'),
        prepared_system_file=str(directory/'prepared.system.xml'),
        prepared_positions_sha256=save_positions(directory/'prepared.positions.npy', positions),
        atom_order=[{'chain': a.residue.chain.id, 'residue': a.residue.id, 'residue_name': a.residue.name, 'atom': a.name} for a in modeller.topology.atoms()])
    final = final_system(forcefield, modeller.topology)
    final_xml = mm.XmlSerializer.serialize(final)
    (directory/'prepared.system.xml').write_text(final_xml)
    diagnostics['prepared_system_sha256'] = hashlib.sha256(final_xml.encode()).hexdigest()
    # Score the same prepared coordinates under the final physical system.
    probe = Simulation(modeller.topology, final, mm.VerletIntegrator(.001), platform, properties)
    probe.context.setPositions(positions)
    initial = probe.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    diagnostics['initial_potential_energy_kj_mol'] = float(initial)
    del probe
    if not np.isfinite(initial):
        raise NumericalFailure('Nonfinite initial potential energy')
    stages = [('soft', soft_system(forcefield, modeller.topology, positions), 200),
              ('flexible', forcefield.createSystem(modeller.topology, nonbondedMethod=NoCutoff, constraints=None, rigidWater=False), 200),
              ('final', final, args.max_iterations)]
    for name, system, cap in stages:
        diagnostics['current_stage'] = name
        positions, row = run_stage(name, modeller.topology, system, positions, platform, properties,
                                   cap, directory, modeller.positions)
        diagnostics['stages'].append(row)
        write_diagnostics(diagnostics)
        if not row['finite_energy_and_forces'] or not row['geometry']['finite_coordinates']:
            raise NumericalFailure(f'Nonfinite OpenMM result in {name} stage')
    last = diagnostics['stages'][-1]
    with open(args.output_pdb, 'w') as handle:
        PDBFile.writeFile(modeller.topology, positions, handle, keepIds=True)
    diagnostics.update(final_potential_energy_kj_mol=last['final_energy_kj_mol'],
                       geometry_passed=last['geometry']['passed'],
                       status='completed' if last['geometry']['passed'] else 'geometry_failed',
                       iterations=sum(s['reporter_calls'] for s in diagnostics['stages']))
    for key in ('initial_potential_energy_kj_mol', 'final_potential_energy_kj_mol', 'iterations', 'added_terminal_atoms'):
        print(f'{key}: {diagnostics[key]}')
    print(f'force_field: {FORCEFIELDS[args.forcefield][0]}')
    print('solvent_model: none\nenergy_units: kJ/mol')
    print(f'output_pdb: {args.output_pdb}')


def write_diagnostics(diagnostics):
    # Strict JSON: failed numerical values are reported as null, never NaN.
    def clean(value):
        if isinstance(value, dict): return {k: clean(v) for k, v in value.items()}
        if isinstance(value, list): return [clean(v) for v in value]
        if isinstance(value, float) and not np.isfinite(value): return None
        return value
    Path('openmm_diagnostics.json').write_text(json.dumps(clean(diagnostics), indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_pdb')
    parser.add_argument('output_pdb')
    parser.add_argument('--max-iterations', type=int, default=500)
    parser.add_argument('--forcefield', choices=sorted(FORCEFIELDS), default='amber14')
    parser.add_argument('--platform', choices=['CUDA', 'Reference', 'CPU'], default='CUDA')
    parser.add_argument('--precision', choices=['mixed', 'double'], default='double')
    args = parser.parse_args()
    if args.max_iterations < 1:
        parser.error('--max-iterations must be positive')
    diagnostics = {'schema_version': 1, 'protocol': PROTOCOL, 'openmm_version': mm.__version__,
        'engine_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'requested_platform': args.platform, 'requested_precision': args.precision,
        'force_field': FORCEFIELDS[args.forcefield][0], 'solvent_model': 'none', 'energy_units': 'kJ/mol',
        'stages': [], 'status': 'running', 'convergence_claimed': False,
        'soft_stage_parameters': {'repulsion_k_kj_mol_nm2': 10000, 'heavy_restraint_k_kj_mol_nm2': 1000, 'iteration_cap': 200},
        'flexible_stage_iteration_cap': 200, 'final_stage_iteration_cap': args.max_iterations}
    try:
        execute(args, diagnostics)
    except Exception as exc:
        message = str(exc).lower()
        numerical = isinstance(exc, NumericalFailure) or (isinstance(exc, mm.OpenMMException) and
            any(text in message for text in ('coordinate is nan', 'coordinates are nan', 'nonfinite', 'not finite')))
        diagnostics.update(status='numerical_failure' if numerical else 'error',
                           numerical_failure=numerical, geometry_passed=False,
                           error={'type': type(exc).__name__, 'message': str(exc)})
        if not numerical:
            raise
    finally:
        write_diagnostics(diagnostics)


if __name__ == '__main__':
    main()
