"""Exercise the wrapper's process boundary without loading Boltz or a GPU."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml


WRAPPER = Path(__file__).resolve().parents[1] / "scripts" / "engines" / "boltz.py"
JOB = {
    "chains": [{"sequence": "ACDEFG", "copies": 1, "msa": None}],
    "recycling_steps": 4, "sampling_steps": 17, "diffusion_samples": 2,
    "step_scale": 1.25, "output_format": "mmcif", "max_msa_seqs": 512,
    "num_subsampled_msa": 128, "seed": 19,
    "use_potentials": True, "subsample_msa": True,
}


@pytest.fixture
def fake_boltz(tmp_path, monkeypatch):
    executable = tmp_path / "boltz"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys, time\n"
        "from pathlib import Path\n"
        "Path('argv.json').write_text(json.dumps(sys.argv[1:]))\n"
        "print('prediction progress', flush=True)\n"
        "print('prediction diagnostic', file=sys.stderr, flush=True)\n"
        "Path('ready').touch()\n"
        "while not Path('release').exists(): time.sleep(0.01)\n"
        "sys.exit(int(os.environ.get('BOLTZ_TEST_EXIT_CODE', '0')))\n"
    )
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    return tmp_path


@pytest.mark.parametrize("exit_code", [0, 7])
def test_worker_free_loading_preserves_prediction_arguments_and_exit_code(fake_boltz, monkeypatch, exit_code):
    monkeypatch.setenv("BOLTZ_TEST_EXIT_CODE", str(exit_code))
    (fake_boltz / "release").touch()
    result = subprocess.run(
        [sys.executable, str(WRAPPER), json.dumps(JOB)], cwd=fake_boltz,
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == exit_code
    assert result.stdout == "prediction progress\n"
    assert result.stderr == "prediction diagnostic\n"
    argv = json.loads((fake_boltz / "argv.json").read_text())
    assert argv[:2] == ["predict", "job.yaml"]
    assert "--num_workers" in argv
    assert argv[argv.index("--num_workers") + 1] == "0"
    for option, expected in {
        "--model": "boltz2", "--accelerator": "gpu", "--devices": "1",
        "--recycling_steps": "4", "--sampling_steps": "17", "--diffusion_samples": "2",
        "--step_scale": "1.25", "--output_format": "mmcif", "--max_msa_seqs": "512",
        "--num_subsampled_msa": "128", "--seed": "19", "--out_dir": "out",
    }.items():
        assert argv[argv.index(option) + 1] == expected
    assert "--use_potentials" in argv
    assert "--subsample_msa" in argv
    assert yaml.safe_load((fake_boltz / "job.yaml").read_text()) == {
        "version": 1,
        "sequences": [{"protein": {"id": "A", "sequence": "ACDEFG", "msa": "empty"}}],
    }


def test_progress_is_visible_before_prediction_exits(fake_boltz):
    stdout_path, stderr_path = fake_boltz / "stdout.log", fake_boltz / "stderr.log"
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(
            [sys.executable, str(WRAPPER), json.dumps(JOB)], cwd=fake_boltz,
            stdout=stdout, stderr=stderr, start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 5
            while not (fake_boltz / "ready").exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert (fake_boltz / "ready").exists(), "Fake predictor did not start"
            assert process.poll() is None
            assert stdout_path.read_text() == "prediction progress\n"
            assert stderr_path.read_text() == "prediction diagnostic\n"
        finally:
            (fake_boltz / "release").touch()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
