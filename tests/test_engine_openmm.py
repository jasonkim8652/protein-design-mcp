"""Numerical engine tests; run directly in the md environment (stdlib unittest)."""
import importlib.util
import tempfile
import os
import json
import hashlib
import unittest
from pathlib import Path

try:
    import numpy as np
    import openmm as mm
    from openmm import unit
    from openmm.app import ForceField, PDBFile
except ImportError:
    mm = None


@unittest.skipIf(mm is None, 'requires the md OpenMM environment')
class OpenMMProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / 'scripts/engines/openmm_minimize.py'
        spec = importlib.util.spec_from_file_location('engine', path)
        cls.engine = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.engine)

    def test_reporter_records_restarts_without_reporting_cap_as_iterations(self):
        reporter = self.engine.Reporter()
        for iteration in [0, 1, 2, 0, 1]:
            self.assertFalse(reporter.report(iteration, [], [], {'system energy': 1.0}))
        self.assertEqual(reporter.calls, 5)
        self.assertEqual(reporter.passes, 2)
        self.assertEqual(reporter.last['iteration'], 1)

    def test_nonfinite_geometry_is_explicitly_unusable(self):
        from openmm.app import Topology, element
        top = Topology(); chain = top.addChain(); res = top.addResidue('GLY', chain)
        top.addAtom('N', element.nitrogen, res)
        result = self.engine.geometry(top, np.array([[np.nan, 0, 0]]) * unit.nanometer)
        self.assertFalse(result['finite_coordinates'])
        self.assertFalse(result['passed'])

    def test_soft_stage_has_flexible_bonds_and_bounded_overlap_energy(self):
        # OpenMM's bundled alanine dipeptide gives a real supported topology.
        import openmm.app
        pdb = PDBFile(str(Path(openmm.app.__file__).parent / 'data/test.pdb'))
        ff = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
        soft = self.engine.soft_system(ff, pdb.topology, pdb.positions)
        self.assertEqual(soft.getNumConstraints(), 0)
        self.assertFalse(any(isinstance(f, mm.NonbondedForce) for f in soft.getForces()))
        repulsion = next(f for f in soft.getForces() if isinstance(f, mm.CustomNonbondedForce))
        self.assertEqual(repulsion.getNumParticles(), len(list(pdb.topology.atoms())))
        self.assertEqual(repulsion.getGlobalParameterDefaultValue(0), 10000)
        self.assertTrue(any(isinstance(f, mm.HarmonicBondForce) for f in soft.getForces()))

    def test_explicit_unavailable_platform_raises_without_fallback(self):
        with self.assertRaisesRegex(Exception, 'There is no registered Platform'):
            self.engine.platform_settings('UnavailablePlatform', 'double')

    def test_charmm_soft_stage_removes_all_singular_nonbonded_terms(self):
        import openmm.app
        pdb = PDBFile(str(Path(openmm.app.__file__).parent / 'data/test.pdb'))
        ff = ForceField('charmm36.xml', 'charmm36/water.xml')
        system = self.engine.soft_system(ff, pdb.topology, pdb.positions)
        custom = [f for f in system.getForces() if isinstance(f, mm.CustomNonbondedForce)]
        self.assertEqual(len(custom), 1)
        self.assertNotIn('/r^', custom[0].getEnergyFunction())
        self.assertFalse(any(isinstance(f, mm.CustomBondForce) for f in system.getForces()))
        self.assertLess(custom[0].getParticleParameters(0)[0], 0.5)

    @unittest.skipUnless(os.environ.get('OPENMM_REGRESSION_ROOT'), 'requires preserved six-case regression artifacts')
    def test_regression_replay_confirms_physical_energy_geometry_and_provenance(self):
        root = Path(os.environ['OPENMM_REGRESSION_ROOT'])
        cases = ['r001_a001_d001_binder', 'r001_a001_d002_binder', 'r003_a001_d001_binder',
                 'r002_a002_d001_complex', 'r002_a002_d001_binder', 'r002_a002_d001_target']
        for name in cases:
            with self.subTest(case=name):
                directory = root/name
                metadata = json.loads((directory/'openmm_diagnostics.json').read_text())
                self.assertEqual(metadata['status'], 'completed')
                self.assertEqual(metadata['protocol'], self.engine.PROTOCOL)
                self.assertEqual(metadata['engine_sha256'], hashlib.sha256(Path(self.engine.__file__).read_bytes()).hexdigest())
                self.assertTrue(metadata['geometry_passed'])
                self.assertEqual(metadata['iterations'], sum(s['reporter_calls'] for s in metadata['stages']))
                last = metadata['stages'][-1]
                self.assertEqual(last['platform'], 'CUDA')
                self.assertEqual(last['platform_properties']['Precision'], 'double')
                system_path = directory/last['system_file']
                positions_path = directory/last['positions_file']
                self.assertEqual(last['system_sha256'], hashlib.sha256(system_path.read_bytes()).hexdigest())
                self.assertEqual(last['positions_sha256'], hashlib.sha256(positions_path.read_bytes()).hexdigest())
                system = mm.XmlSerializer.deserialize(system_path.read_text())
                self.assertFalse(any(isinstance(f, (mm.CustomNonbondedForce, mm.CustomExternalForce)) for f in system.getForces()))
                pdb = PDBFile(str(directory/'openmm_states/prepared.pdb'))
                xyz = np.load(positions_path)*unit.nanometer
                self.assertTrue(self.engine.geometry(pdb.topology, xyz)['passed'])
                integrator = mm.VerletIntegrator(.001)
                context = mm.Context(system, integrator, mm.Platform.getPlatformByName('CUDA'), {'Precision': 'double', 'DeterministicForces': 'true'})
                context.setPositions(xyz)
                energy = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                self.assertAlmostEqual(energy, metadata['final_potential_energy_kj_mol'], delta=0.001)
                del context, integrator

    def test_numeric_failure_is_distinct_from_platform_failure(self):
        import os
        import json
        from unittest.mock import patch
        previous = os.getcwd()
        try:
            with tempfile.TemporaryDirectory() as directory:
                os.chdir(directory)
                with patch('sys.argv', ['engine', 'input.pdb', 'output.pdb']):
                    with patch.object(self.engine, 'execute', side_effect=mm.OpenMMException('Particle coordinate is NaN')):
                        self.engine.main()
                    metadata = json.loads(Path('openmm_diagnostics.json').read_text())
                    self.assertEqual(metadata['status'], 'numerical_failure')
                    self.assertFalse(metadata['geometry_passed'])
                    with patch.object(self.engine, 'execute', side_effect=mm.OpenMMException('CUDA_ERROR_UNSUPPORTED_PTX_VERSION')):
                        with self.assertRaises(mm.OpenMMException):
                            self.engine.main()
                    metadata = json.loads(Path('openmm_diagnostics.json').read_text())
                    self.assertEqual(metadata['status'], 'error')
        finally:
            os.chdir(previous)

    def test_final_system_contains_no_temporary_forces(self):
        import openmm.app
        pdb = PDBFile(str(Path(openmm.app.__file__).parent / 'data/test.pdb'))
        ff = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
        system = self.engine.final_system(ff, pdb.topology)
        self.assertGreater(system.getNumConstraints(), 0)
        self.assertTrue(any(isinstance(f, mm.NonbondedForce) for f in system.getForces()))
        self.assertFalse(any(isinstance(f, (mm.CustomNonbondedForce, mm.CustomExternalForce)) for f in system.getForces()))


if __name__ == '__main__':
    unittest.main()
