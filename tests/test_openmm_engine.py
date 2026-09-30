"""Real OpenMM regression, runnable with stdlib unittest inside the md env."""
import importlib.util
import gzip
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(importlib.util.find_spec("openmm"), "requires the md engine environment")
class OpenMMTerminiTest(unittest.TestCase):
    def test_native_cif_and_compressed_inputs_preserve_residues(self):
        from openmm.app import PDBFile, PDBxFile
        source = PDBFile(str(ROOT / "tests/fixtures/test_pdbs/af2_multimer_unrelaxed.pdb"))
        expected = [[residue.name for residue in chain.residues()] for chain in source.topology.chains()]
        for suffix in (".cif", ".mmcif", ".cif.gz", ".mmcif.gz", ".pdb.gz"):
            with self.subTest(suffix=suffix), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / ("input" + suffix)
                opener = gzip.open if suffix.endswith(".gz") else open
                writer = PDBFile if suffix == ".pdb.gz" else PDBxFile
                with opener(path, "wt") as handle:
                    writer.writeFile(source.topology, source.positions, handle, keepIds=True)
                original = path.read_bytes()
                output = Path(directory) / "minimized.pdb"
                result = subprocess.run([sys.executable, str(ROOT / "scripts/engines/openmm_minimize.py"),
                    str(path), str(output), "--max-iterations", "5"], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                minimized = PDBFile(str(output))
                self.assertEqual([[residue.name for residue in chain.residues()]
                                  for chain in minimized.topology.chains()], expected)
                energies = [float(line.split(": ")[1]) for line in result.stdout.splitlines()
                            if "potential_energy_kj_mol:" in line]
                self.assertEqual(len(energies), 2)
                self.assertTrue(all(math.isfinite(value) for value in energies))
                self.assertLess(energies[1], energies[0])
                self.assertIn("solvent_model: none", result.stdout)
                self.assertIn("force_field: amber14-all.xml", result.stdout)
                self.assertEqual(path.read_bytes(), original)

    def test_cif_missing_backbone_atom_is_rejected_without_reconstruction(self):
        from openmm.app import Modeller, PDBFile, PDBxFile
        source = PDBFile(str(ROOT / "tests/fixtures/test_pdbs/af2_multimer_unrelaxed.pdb"))
        modeller = Modeller(source.topology, source.positions)
        modeller.delete([next(atom for atom in modeller.topology.atoms() if atom.name == "CA")])
        with tempfile.TemporaryDirectory() as directory:
            path, output = Path(directory) / "incomplete.cif", Path(directory) / "minimized.pdb"
            with path.open("w") as handle:
                PDBxFile.writeFile(modeller.topology, modeller.positions, handle)
            result = subprocess.run([sys.executable, str(ROOT / "scripts/engines/openmm_minimize.py"),
                str(path), str(output), "--max-iterations", "5"], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("missing non-terminal heavy atoms", result.stderr)
            self.assertFalse(output.exists())

    def test_minimizes_af2_complex_without_terminal_oxygen(self):
        source = ROOT / "tests/fixtures/test_pdbs/af2_multimer_unrelaxed.pdb"
        original = source.read_bytes()
        self.assertNotIn(b" OXT ", original)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "minimized.pdb"
            result = subprocess.run([sys.executable, str(ROOT / "scripts/engines/openmm_minimize.py"),
                str(source), str(output), "--max-iterations", "5"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(output.read_text().count(" OXT "), 2)
            energies = [float(line.split(": ")[1]) for line in result.stdout.splitlines()
                        if "potential_energy_kj_mol:" in line]
            self.assertEqual(len(energies), 2)
            self.assertTrue(all(math.isfinite(value) for value in energies))
            self.assertLess(energies[1], energies[0])
            self.assertIn("solvent_model: none", result.stdout)
            self.assertIn("force_field: amber14-all.xml", result.stdout)
        self.assertEqual(source.read_bytes(), original)
