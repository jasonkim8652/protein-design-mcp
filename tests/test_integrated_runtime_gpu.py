"""Opt-in image test: imports cannot stand in for a real OpenMM CUDA kernel."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess

import pytest


@pytest.mark.skipif(not os.environ.get("PDMCP_RUNTIME_TEST_IMAGE"),
                    reason="requires a built integrated image and an available GPU")
def test_openmm_gpu_probe_computes_energy_and_forces_without_cpu_fallback():
    path = Path(__file__).parents[1] / "scripts/verify_integrated_runtime.py"
    spec = importlib.util.spec_from_file_location("runtime_verifier", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    request = {"imports": [{"module": "openmm", "external": False}], "binaries": [],
               "gpu_smoke": True, "jax_runtime": False}
    result = subprocess.run([
        "docker", "run", "--rm", "--network", "none",
        "--device=nvidia.com/gpu=" + os.environ.get("PDMCP_RUNTIME_TEST_GPU", "0"),
        "--entrypoint", "/opt/conda/envs/md/bin/python",
        os.environ["PDMCP_RUNTIME_TEST_IMAGE"], "-c", module.PROBE, json.dumps(request),
    ], capture_output=True, text=True, timeout=90)
    lines = [line[len(module.MARKER):] for line in result.stdout.splitlines()
             if line.startswith(module.MARKER)]
    assert lines, result.stderr
    report = json.loads(lines[-1])
    assert report.get("openmm", {}).get("checked") is True, report
    assert result.returncode == 0 and report["openmm"]["ok"], report
    assert report["openmm"]["platform"] == "CUDA"
    assert report["openmm"]["energy_kj_mol"] == pytest.approx(0.5)
    assert report["openmm"]["force0_x_kj_mol_nm"] == pytest.approx(10.0)
