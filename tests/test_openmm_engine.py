"""Real OpenMM regression, runnable with stdlib unittest inside the md env."""
import importlib.util
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(importlib.util.find_spec("openmm"), "requires the md engine environment")
class OpenMMTerminiTest(unittest.TestCase):
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
