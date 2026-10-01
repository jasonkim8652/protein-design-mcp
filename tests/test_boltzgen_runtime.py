"""Runtime regressions for the BoltzGen loader and live engine diagnostics."""

import importlib
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from protein_design_mcp.app import manifest_dir
from protein_design_mcp.manifest.loader import load_manifests
from protein_design_mcp.validation import ToolInputError, validate_and_fill


@pytest.mark.parametrize("tool, supplied", [
    ("design", {"target_structure": "target.cif", "target_chains": ["A"]}),
    ("fold", {"design_spec": "design.yaml", "generated_files": ["a.cif", "a.npz"],
              "with_target": True}),
    ("inverse_fold", {"design_spec": "design.yaml"}),
])
@pytest.mark.parametrize("workers, expected", [(None, "0"), (0, "0"), (1, None)])
def test_gpu_prediction_uses_in_process_loading_and_refuses_unsafe_workers(
    tool, supplied, workers, expected
):
    """A default call must not cross the observed stuck DataLoader queue."""
    manifest = next(m for m in load_manifests(manifest_dir())
                    if m.name == f"run_boltzgen_{tool}")
    if workers is not None:
        supplied = {**supplied, "num_workers": workers}
    if expected is None:
        with pytest.raises(ToolInputError, match="num_workers.*maximum of 0") as error:
            validate_and_fill(manifest, supplied)
        assert "multiprocessing queue stalls" in str(error.value)
        return
    params = validate_and_fill(manifest, supplied)
    adapter = importlib.import_module(f"protein_design_mcp.adapters.boltzgen_{tool}")
    argv = adapter.build_args(manifest, params)
    if tool == "fold":
        assert f"data.cfg.num_workers={expected}" in argv
    else:
        assert argv[argv.index("--num_workers") + 1] == expected


@pytest.mark.parametrize("returncode", [0, 7])
def test_design_wrapper_exposes_both_streams_before_engine_exit(tmp_path, returncode):
    """Capturing and replaying only on exit hides errors during a loader stall."""
    wrapper = Path(__file__).resolve().parents[1] / "scripts/engines/boltzgen_design.py"
    engine = tmp_path / "boltzgen"
    engine.write_text(
        f"#!{sys.executable}\n"
        "import pathlib, sys, time\n"
        "print('engine progress', flush=True)\n"
        "print('loader diagnostic', file=sys.stderr, flush=True)\n"
        "pathlib.Path('ready').touch()\n"
        "deadline = time.monotonic() + 15\n"
        "while not pathlib.Path('release').exists() and time.monotonic() < deadline:\n"
        "    time.sleep(0.01)\n"
        f"sys.exit({returncode})\n"
    )
    engine.chmod(0o755)
    stdout, stderr = tmp_path / "stdout.log", tmp_path / "stderr.log"
    env = {**os.environ, "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
           "PYTHONUNBUFFERED": "1"}
    with stdout.open("w") as out, stderr.open("w") as err:
        proc = subprocess.Popen(
            [sys.executable, str(wrapper), "--target-structure", "target.cif",
             "--target-chains", "A", "--binder-length-min", "60",
             "--binder-length-max", "100"], cwd=tmp_path, env=env,
            stdout=out, stderr=err,
        )
        try:
            deadline = time.monotonic() + 5
            while not (tmp_path / "ready").exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert (tmp_path / "ready").exists(), stderr.read_text()
            assert proc.poll() is None
            visible_stdout, visible_stderr = stdout.read_text(), stderr.read_text()
        finally:
            (tmp_path / "release").touch()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
    assert "engine progress" in visible_stdout
    assert "loader diagnostic" in visible_stderr
    assert proc.returncode == returncode
