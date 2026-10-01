"""Exercise real wrapper/child process I/O without launching model inference."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _flags(**values):
    return [item for key, value in values.items() for item in ("--" + key.replace("_", "-"), str(value))]


CASES = [
    ("rf3", _flags(chains='[{"sequence":"ACDE","chain_id":"A"}]', msa="null", n_recycles=1, diffusion_batch_size=1, num_steps=1)),
    ("rfd3", [json.dumps({"job": {}, "engine": {"diffusion_batch_size": 1, "num_timesteps": 1, "step_scale": 1}})]),
    ("promera", _flags(schema="{}", msa="null", recycling_steps=1, diffusion_samples=1, diffusion_steps=1, num_seeds=1)),
    ("protenix", [json.dumps({"chains": [], "seeds": [1], "cycle": 1, "step": 1, "sample": 1, "dtype": "bf16", "need_atom_confidence": True})]),
    ("openfold3", [json.dumps({"chains": [], "seeds": [1], "num_diffusion_samples": 1})]),
    ("alphafold3", [json.dumps({"chains": [], "seeds": [1], "num_recycles": 1, "num_diffusion_samples": 1, "max_template_date": "2020-01-01", "resolve_msa_overlaps": True, "flash_attention_implementation": "xla", "save_embeddings": False, "save_distogram": False, "buckets": [256], "conformer_max_iterations": None})]),
    ("alphafold2_multimer", _flags(sequences='["ACDE","FGHI"]', msa_mode='single_sequence', num_recycle=1, num_models=1, num_seeds=1, random_seed=1, num_ensemble=1, pair_mode="unpaired", pair_strategy="greedy", rank="auto", stop_at_score=100)),
    ("genie3_scaffold", _flags(model_variant="v1", min_length=20, max_length=20, length_step=1, num_samples=1, batch_size=1, direction_scale=1, eta=1, n_sample_step=1, noise_scale=1, predict_sidechain="false", seed=1)),
    ("genie3_binder", _flags(target_pdb="target.pdb", hotspot_residues='["A1"]', binder_min_length=20, binder_max_length=20, num_samples=1, model_variant="v1", direction_scale=1, eta=1, n_sample_step=1, noise_scale=1, predict_sidechain="false", seed=1, expand_interface="false", interface_cutoff_angstrom=10, interface_rsa_threshold=1, interface_abs_sasa_threshold=1)),
    ("protpardelle", _flags(target_pdb="target.pdb", contig="A1-4/20-20", total_lengths="[[24,24]]", hotspots="null", model="cc83", step_scale=1, schurn=1, crop_cond_start=1, translation="[0,0,0]", num_samples=1, batch_size=1)),
    ("rfdiffusion2", [json.dumps({"target_pdb": "target.pdb", "contig": "A1-4/20-20", "num_designs": 1, "diffusion_steps": 1, "noise_scale_ca": 1, "noise_scale_frame": 1, "ckpt_variant": "140"})]),
    ("multiflow", []),
]


@pytest.mark.parametrize("wrapper,args", CASES, ids=[case[0] for case in CASES])
def test_wrapper_streams_child_output_before_exit_and_preserves_failure(tmp_path, wrapper, args):
    worker = tmp_path / "worker.py"
    worker.write_text(
        "import sys,time\nfrom pathlib import Path\n"
        "print('worker progress', flush=True)\n"
        "print('worker diagnostic', file=sys.stderr, flush=True)\n"
        "Path('ready').touch()\n"
        "while not Path('release').exists(): time.sleep(.01)\n"
        "sys.exit(7)\n"
    )
    # Replace only the external model command. Run real Popen and the wrapper's
    # actual subprocess.run arguments, stream handling, and error propagation.
    launcher = tmp_path / "launch.py"
    launcher.write_text(
        "import runpy, subprocess, sys\n"
        "real_popen = subprocess.Popen\n"
        f"worker = {str(worker)!r}\n"
        "class WorkerProcess(real_popen):\n"
        "    def __init__(self, args, *a, **kw):\n"
        "        super().__init__([sys.executable, worker], *a, **kw)\n"
        "subprocess.Popen = WorkerProcess\n"
        "sys.argv = sys.argv[1:]\n"
        "sys.path.insert(0, str(__import__('pathlib').Path(sys.argv[0]).parent))\n"
        "runpy.run_path(sys.argv[0], run_name='__main__')\n"
    )
    stdout_path, stderr_path = tmp_path / "stdout", tmp_path / "stderr"
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(
            [sys.executable, str(launcher), str(ROOT / "scripts" / "engines" / f"{wrapper}.py"), *args],
            cwd=tmp_path, stdout=stdout, stderr=stderr, start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 5
            while not (tmp_path / "ready").exists() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(.01)
            assert (tmp_path / "ready").exists(), stderr_path.read_text()
            assert process.poll() is None
            assert stdout_path.read_text() == "worker progress\n"
            assert stderr_path.read_text() == "worker diagnostic\n"
            (tmp_path / "release").touch()
            assert process.wait(timeout=5) == 7
        finally:
            (tmp_path / "release").touch()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)


def test_stderr_tee_keeps_only_a_bounded_tail_and_full_log(tmp_path):
    log = tmp_path / "stderr.log"
    launcher = (
        f"import sys; sys.path.insert(0, {str(ROOT / 'scripts')!r}); "
        "from engines._streaming import run_with_stderr_tail; "
        "result = run_with_stderr_tail([sys.executable, '-c', "
        "\"import sys; sys.stderr.write('x'*3000000 + 'final error'); sys.exit(7)\"]); "
        "print(result.returncode, len(result.stderr), result.stderr.endswith('final error'))"
    )
    with log.open("wb") as stderr:
        result = subprocess.run([sys.executable, "-c", launcher], stderr=stderr, stdout=subprocess.PIPE, text=True, timeout=10)
    assert result.returncode == 0
    assert result.stdout.strip() == "7 1048576 True"
    assert log.stat().st_size == 3000011
    assert log.read_bytes().endswith(b"final error")


def test_mmseqs_step_error_preserves_child_diagnostic(tmp_path):
    launcher = (
        f"import sys; sys.path.insert(0, {str(ROOT / 'scripts')!r}); "
        "from engines.run_mmseqs_search import _run; "
        "_run(sys.executable, ['-c', \"import sys; print('search failed', file=sys.stderr); sys.exit(7)\"], 'search')"
    )
    result = subprocess.run([sys.executable, "-c", launcher], cwd=tmp_path, capture_output=True, text=True, timeout=10)
    assert result.returncode != 0
    assert result.stderr.startswith("search failed\n")
    assert "mmseqs search exited 7" in result.stderr
    assert "stderr (tail):\nsearch failed" in result.stderr


def test_multiflow_keeps_existing_partial_generation_policy(tmp_path):
    worker = tmp_path / "worker.py"
    worker.write_text(
        "import sys\nfrom pathlib import Path\n"
        "Path('predict_out').mkdir()\n"
        "Path('predict_out/sample.pdb').write_text('ATOM')\n"
        "print(\"ModuleNotFoundError: No module named 'deepspeed'\", file=sys.stderr)\n"
        "sys.exit(7)\n"
    )
    launcher = (
        f"import sys; sys.path.insert(0, {str(ROOT / 'scripts')!r}); "
        "from engines import multiflow; "
        f"multiflow._INFERENCE_SCRIPT = {str(worker)!r}; multiflow.main()"
    )
    result = subprocess.run([sys.executable, "-c", launcher], cwd=tmp_path, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert "No module named 'deepspeed'" in result.stderr
    assert "no self-consistency score" in result.stderr
    assert json.loads((tmp_path / "self_consistency_summary.json").read_text()) == {}
