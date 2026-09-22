"""Wrapper for RFdiffusion2 (`run_rfdiffusion2`). Supports two backends,
chosen per call by the `backend` parameter (see the manifest's schema and
doc for the full explanation of the trade-off -- summarised here for the
implementation):

- `"conda"` (default; matches this tool's own `engine:` dispatch, and every
  other engine in this project): runs directly in this host's `rfd2_fixed`
  conda environment via the current interpreter (`sys.executable`), which
  already has `PYTHONPATH` pointed at the RFdiffusion2 checkout by the
  manifest's `engine.env_vars`. Needs no capability beyond what every other
  tool in this server already needs. **CONFIRMED LIVE to complete a
  generation on this host**, 2026-09-22, through `ServerApp.call_tool` end
  to end on GPU 7. `rfd2_fixed` is a clone of `rfd2_src` (left untouched)
  with the gaps a real run reaches but a bare `import rf_diffusion` does
  not closed: `pydantic` (`dgl`'s own import-time dependency, missing
  outright), a scipy/numpy pairing left broken by an earlier, unrelated
  `pip install` (fixed by a clean `numpy==1.26.4` reinstall), and `fire`
  (imported directly by `rf_diffusion/run_inference.py`, missing outright).
- `"docker"`: launches the OFFICIAL upstream container image
  (`rfdiffusion2-sif:converted`, converted from RFdiffusion2's own
  Apptainer `.sif` -- Apptainer itself cannot run on this host, a kernel
  `apparmor` policy, confirmed, not a misconfiguration) as a SIBLING
  Docker container over the host's Docker socket. CONFIRMED LIVE working
  end to end, including through this server's own real dispatch path.
  **This is a root-equivalent capability** (anything that reaches
  `/var/run/docker.sock` has root-equivalent control of the host) and,
  combined with C9 (the HTTP transport has no authentication and every
  tool already takes a caller-supplied path), is a direct path from
  unauthenticated network access to root if this server's own deployment
  grants it by default. It must therefore be an explicit, opt-in choice
  by whoever deploys this server for a specific host, never the default --
  this wrapper does not fall back to it silently when `"conda"` fails.

Bind-mounted paths (`-v host:container`) in the docker backend are
resolved by the Docker DAEMON against the true host filesystem, not
against whatever filesystem view the PROCESS issuing `docker run` happens
to see -- this is exactly what makes Docker-outside-of-Docker work at
all: this wrapper does not itself need `/home/jk661/projects/RFdiffusion2`
or the scratch workdir mounted into its own container for those `-v`
flags to resolve correctly on the real host.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

IMAGE = "rfdiffusion2-sif:converted"
REPO_ROOT = "/home/jk661/projects/RFdiffusion2"
GPU_DEVICE = "7"  # this host's GPU-7-only policy -- see module docstring.


def _common_overrides(job: dict, out_dir: Path, ckpt_path: str) -> list[str]:
    """Hydra overrides shared by both backends -- everything except how the
    engine itself is invoked. All CONFIRMED LIVE this wave (see the
    manifest's doc for each override's own rationale):

    - `--config-name=aa` (never `aa_ppi`, whose own default checkpoint is
      not present on this host).
    - `contigmap.has_termini`: required whenever `contigmap.contigs`
      describes more than one chain; derived from `contig`'s own
      chain-segment count rather than asked of the caller separately.
    - `+transforms.configs.CenterPostTransform.center_type=is_not_diffused`:
      without hotspot conditioning (not available with the checkpoints on
      this host -- see the manifest's doc), the default centering strategy
      needs an `ORI` HETATM record this tool cannot supply; centering on
      the target's own center of mass instead needs no such record.
    """
    contig = job["contig"]
    n_chains = contig.count("_") + 1
    has_termini = "[" + ",".join(["True"] * n_chains) + "]"
    return [
        "--config-name=aa",
        f"inference.ckpt_path={ckpt_path}",
        f"inference.input_pdb={job['target_pdb']}",
        f"inference.output_prefix={out_dir}/design",
        f"inference.num_designs={job['num_designs']}",
        f"diffuser.T={job['diffusion_steps']}",
        f"contigmap.contigs=['{contig}']",
        f"contigmap.has_termini={has_termini}",
        f"denoiser.noise_scale_ca={job['noise_scale_ca']}",
        f"denoiser.noise_scale_frame={job['noise_scale_frame']}",
        "+transforms.configs.CenterPostTransform.center_type=is_not_diffused",
    ]


def _run_conda(job: dict, out_dir: Path, ckpt_path: str) -> subprocess.CompletedProcess:
    """Default backend: invoke `run_inference.py` directly with the current
    (rfd2_fixed) interpreter. `PYTHONPATH` is already set by the manifest's
    `engine.env_vars`, so `rf_diffusion` resolves the same way this
    module's own docstring confirms it does.
    """
    script = f"{REPO_ROOT}/rf_diffusion/run_inference.py"
    cmd = [sys.executable, script, *_common_overrides(job, out_dir, ckpt_path)]
    return subprocess.run(cmd, capture_output=True, text=True)


def _run_docker(job: dict, out_dir: Path, ckpt_path: str, workdir: Path) -> subprocess.CompletedProcess:
    """Opt-in backend: launch the official image as a sibling container.
    Only reached when the caller explicitly set `backend: "docker"` --
    never a silent fallback from `_run_conda` (see module docstring for
    why: that capability must be an explicit, opt-in deployment choice).
    """
    # `target_pdb` is a caller-supplied absolute HOST path and, unlike the
    # conda backend (a plain subprocess that sees the whole host
    # filesystem automatically), this nested container only sees what is
    # explicitly bind-mounted -- CONFIRMED LIVE: target_pdb outside
    # `workdir` (the common case) produced FileNotFoundError without this.
    target_pdb = Path(job["target_pdb"])
    extra_mounts: list[str] = []
    if target_pdb.parent != workdir and not str(target_pdb).startswith(str(workdir) + "/"):
        extra_mounts = ["-v", f"{target_pdb.parent}:{target_pdb.parent}:ro"]

    docker_cmd = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        f"device={GPU_DEVICE}",
        # Image runs as root by default, which would leave every output
        # file root-owned and unreadable by the dispatcher's own
        # output-collection step (confirmed both ways).
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-v",
        f"{REPO_ROOT}:{REPO_ROOT}:ro",
        "-v",
        f"{workdir}:{workdir}:rw",
        *extra_mounts,
        "-e",
        f"PYTHONPATH={REPO_ROOT}",
        # RFdiffusion2's own config resolution interpolates ${env:USER}
        # for an unrelated wandb_dir default even on pure inference and
        # raises InterpolationResolutionError if unset.
        "-e",
        "USER=rfdiffusion2mcp",
        # So the one-time IGSO3 rotation-schedule cache this engine builds
        # on first use writes into the (disposable) scratch workdir rather
        # than wherever $HOME would otherwise resolve to.
        "-e",
        f"HOME={workdir}",
        "-w",
        str(workdir),
        IMAGE,
        "python3",
        f"{REPO_ROOT}/rf_diffusion/run_inference.py",
        *_common_overrides(job, out_dir, ckpt_path),
    ]
    return subprocess.run(docker_cmd, capture_output=True, text=True)


def main() -> None:
    job = json.loads(sys.argv[1])

    workdir = Path.cwd()
    out_dir = workdir / "out"
    out_dir.mkdir(exist_ok=True)

    ckpt_path = f"{REPO_ROOT}/rf_diffusion/model_weights/RFD_{job['ckpt_variant']}.pt"

    backend = job.get("backend", "conda")
    if backend == "docker":
        result = _run_docker(job, out_dir, ckpt_path, workdir)
    else:
        result = _run_conda(job, out_dir, ckpt_path)

    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
