"""Wrapper for RFdiffusion2 (`run_rfdiffusion2`).

RFdiffusion2's own documented invocation path runs everything inside an
Apptainer/Singularity image. Apptainer itself is unusable on this host --
diagnosed, not guessed: it installs (conda env `apptainer`, v1.5.3) but
`apptainer exec` fails with `Could not write info to setgroups: Permission
denied` because the kernel has `apparmor_restrict_unprivileged_userns=1`;
plain `unshare --user` fails the same way, confirming it is kernel policy,
not an apptainer misconfiguration, and fixing it needs root, which this
host does not have for this work. The original `.sif`
(`rf_diffusion/exec/bakerlab_rf_diffusion_aa.sif`) was converted instead,
via a path that needs no user namespaces (`unsquashfs` + `docker import`),
into a working Docker image: `rfdiffusion2-sif:converted` (21.7G). THIS IS
now the official upstream environment (same container the apptainer image
would have run), not a workaround -- it is simply reached through `docker
run` instead of `apptainer run`.

## Why this wrapper shells out to `docker run` -- a sibling container, launched from inside this server
No other tool in this project does this; every other GPU engine runs as a
plain subprocess inside a HOST conda environment that is mounted into this
server's own container (`engine.prefix`, dispatched as `micromamba run -p`).
RFdiffusion2 cannot work that way: none of the mounted host conda
environments have PyTorch/dgl/pydantic versions consistent with what this
container image ships (CONFIRMED LIVE: running RFdiffusion2 directly in the
`rfd2_src` conda env fails with `ModuleNotFoundError: No module named
'pydantic'`, transitively required by `dgl`, which the model's own
`rf2aa.util_module` imports). The only environment that actually has a
consistent, complete dependency set is this container image, so this
wrapper launches it directly with `docker run --rm`.

This means: **`docker` must be reachable from wherever this wrapper itself
runs, with access to a Docker daemon that also has GPU 7.** This wrapper's
own `engine.prefix` (`rfd2_src`) is a completely ordinary conda environment
otherwise -- it contributes nothing to the docker invocation except a
`python` interpreter to run this script and (via inherited `PATH`) the
`/usr/bin/docker` client. If this MCP server is itself deployed inside a
container (per this project's own substrate design), THAT outer container
needs the host's Docker socket bind-mounted and the `docker` CLI installed
(the "Docker-outside-of-Docker" pattern) for this tool to work there --
this wave's file scope ("three files, no edits to shared infrastructure")
cannot provision that, so it is stated here as a real deployment
prerequisite rather than silently assumed. It was NOT possible to verify
from within this wave whether the currently-deployed server container has
that access; every live confirmation in this tool's manifest doc was run
directly on the host, invoking `docker` the same way this wrapper does.

Bind-mounted paths (`-v host:container`) are resolved by the Docker
DAEMON against the true host filesystem, not against whatever filesystem
view the PROCESS issuing `docker run` happens to see -- this is exactly
what makes Docker-outside-of-Docker work at all: this wrapper does not
itself need `/home/jk661/projects/RFdiffusion2` or the scratch workdir
mounted into its own container for those `-v` flags to resolve correctly
on the real host.

## What this wrapper actually builds, and why each piece is there (all CONFIRMED LIVE this wave)
- `--gpus device=7`: this host's GPU-7-only policy, applied explicitly
  because a nested `docker run` is a PEER call to the host daemon, not
  something inherited from whatever GPU restriction the outer container
  (if any) was started with.
- `--user <uid>:<gid>`: the image runs as root by default, which would
  leave every output file root-owned and unreadable by this dispatcher's
  own output-collection step (confirmed: without this flag, `design_0*.pdb`
  came out `-rw-r--r-- root root`).
- `-e USER=...`, `-e HOME=<workdir>`: RFdiffusion2's own config resolution
  interpolates `${env:USER}` for an unrelated `wandb_dir` default even on a
  pure inference run and raises `InterpolationResolutionError` if it is
  unset; `HOME` is set so the one-time IGSO3 rotation-schedule cache this
  engine builds on first use writes into the (disposable) scratch workdir
  rather than wherever `$HOME` would otherwise resolve to.
- `-e PYTHONPATH=<repo>`: `rf_diffusion` is not pip-installed even inside
  this image -- the image supplies the DEPENDENCIES, the repo checkout
  supplies the CODE, exactly like the original apptainer invocation's own
  shebang script did.
- `--config-name=aa` (never `aa_ppi`): `aa_ppi.yaml`'s own default
  checkpoint (`.../ppi2024-10-10.../RFD_7.pt`) is not present on this host
  -- only the two general-purpose checkpoints `RFD_140.pt`/`RFD_173.pt`
  are. CONFIRMED LIVE that this has a real consequence, not just a config
  difference: those two checkpoints were never trained with
  `FindHotspotsTrainingTransform`, so `ppi.hotspot_res` conditioning
  raises `AssertionError: Model not set up for hotspots` regardless of how
  it is passed -- this is why `run_rfdiffusion2`'s schema has no
  hotspot parameter at all (see the manifest's doc).
- `contigmap.has_termini=[True, ...]`: required whenever `contigmap.contigs`
  describes more than one chain (an assertion the engine raises otherwise);
  this wrapper derives its length from `contig`'s own chain-segment count
  rather than asking the caller for a second, easy-to-desync parameter.
- `+transforms.configs.CenterPostTransform.center_type=is_not_diffused`:
  without hotspot conditioning, the default centering strategy
  (`is_diffused`) requires an `ORI` HETATM record in the input PDB that
  this tool has no way to supply; centering on the NON-diffused (target)
  region's center of mass instead needs no such record and was CONFIRMED
  LIVE to work.
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


def main() -> None:
    job = json.loads(sys.argv[1])

    workdir = Path.cwd()
    out_dir = workdir / "out"
    out_dir.mkdir(exist_ok=True)

    contig = job["contig"]
    n_chains = contig.count("_") + 1
    has_termini = "[" + ",".join(["True"] * n_chains) + "]"

    ckpt_path = f"{REPO_ROOT}/rf_diffusion/model_weights/RFD_{job['ckpt_variant']}.pt"

    # `target_pdb` is a caller-supplied absolute HOST path and, unlike every
    # other engine in this project (which run as a plain subprocess and so
    # see the whole host filesystem automatically), this nested container
    # only sees what is explicitly bind-mounted -- CONFIRMED LIVE:
    # target_pdb outside `workdir` (the common case; the dispatcher's own
    # scratch directory is not where a caller's real input files live)
    # produced `FileNotFoundError` inside the container without this.
    # Mounting its parent directory read-only (rather than the file itself)
    # keeps this a one-line addition even if a future caller also needs a
    # sibling file (e.g. a matching `.trb`) from the same directory.
    target_pdb = Path(job["target_pdb"])
    extra_mounts = []
    if target_pdb.parent != workdir and not str(target_pdb).startswith(str(workdir) + "/"):
        extra_mounts = ["-v", f"{target_pdb.parent}:{target_pdb.parent}:ro"]

    docker_cmd = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        f"device={GPU_DEVICE}",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-v",
        f"{REPO_ROOT}:{REPO_ROOT}:ro",
        "-v",
        f"{workdir}:{workdir}:rw",
        *extra_mounts,
        "-e",
        f"PYTHONPATH={REPO_ROOT}",
        "-e",
        "USER=rfdiffusion2mcp",
        "-e",
        f"HOME={workdir}",
        "-w",
        str(workdir),
        IMAGE,
        "python3",
        f"{REPO_ROOT}/rf_diffusion/run_inference.py",
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

    result = subprocess.run(docker_cmd, capture_output=True, text=True)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
