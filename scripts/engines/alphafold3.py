"""Wrapper for AlphaFold 3 (``run_alphafold3``). Runs inside the ``scoring``
environment (nothing beyond the Python standard library is needed -- see
the manifest's own comment on why).

Unlike every other engine in this server, AlphaFold 3 is NOT a process
inside a conda environment mounted into this server's own container: it
runs from its own Docker image. This wrapper therefore shells out to
``docker run``, launching a SIBLING container next to this server's own --
see the manifest's "How this tool is dispatched" section for the
deployment requirements that creates (Docker CLI + socket access from
inside this server's own container; a scratch workdir that is valid at an
IDENTICAL path on the Docker daemon's host).

Ground truth for the invocation shape is the user's own checkout,
``~/projects/af3-mmseqs-gpu`` (read-only) --
``benchmarks/run_inference_original_db.sh`` for the ``docker run``
arguments and in-container activation sequence, ``docs/input.md`` for the
AlphaFold 3 JSON schema, ``docs/output.md`` for the output directory
layout this wrapper's manifest ``outputs:`` patterns are built from.

The image is ``romerolabduke/alphafast:latest`` (12.1GB, present on this
host; verified live 2026-09-22 that ``/alphafold3_venv`` inside it is a
real AlphaFold 3 venv) -- the RomeroLab MMseqs2-GPU fork the design spec
names, and the same image the benchmark script's own ``DOCKER_IMAGE="$2"``
takes as an argument rather than hardcoding.

**This wrapper runs the image's OWN baked-in entrypoint,
``/app/alphafold/run_alphafold.py`` -- NOT the ground-truth benchmark
script's mounted copy.** CONFIRMED LIVE, 2026-09-22: mounting the host's own
``~/projects/af3-mmseqs-gpu/run_alphafold.py`` in, exactly as
``benchmarks/run_inference_original_db.sh`` does, fails immediately with
``ModuleNotFoundError: No module named 'alphafold3.jax.attention'`` --
that host script's top-level ``from alphafold3.jax.attention import
attention`` (added by a LATER point in the same RomeroLab fork's history)
has no matching module in the ``alphafold3`` package actually baked into
this image (confirmed by listing ``/app/alphafold/src/alphafold3/jax/``
inside the image: only a ``geometry`` subpackage, no ``attention`` one).
This image's own ``/app/alphafold/run_alphafold.py`` is genuinely the same
fork family -- same flag surface (``--run_data_pipeline``,
``--num_recycles``, ``--num_diffusion_samples``, ``--resolve_msa_overlaps``,
``--flash_attention_implementation``, ``--buckets``,
``--conformer_max_iterations``, ...) built against ``tokamax``/
``ModelRunner`` instead -- and is guaranteed self-consistent with the
package actually installed here, so this wrapper uses it directly rather
than mounting anything from the host repo at all.

``MODEL_DIR_HOST`` is ``/opt/alphafold3_data/weights`` (contains
``af3.bin``/``af3.bin.zst`` -- note the identically-named
``/opt/alphafold3_data/models`` is empty and is NOT this path); AlphaFold
3's own ``params.select_model_files`` matches ``af3.bin.zst`` first and
returns just that one file, so having both the raw and compressed weight
file in the same directory does not trigger its "Multiple models matched"
error (confirmed by reading ``alphafold3/model/params.py`` directly out of
the image, 2026-09-22).

The job is always named ``"job"`` (never derived from caller input).
**Output layout, confirmed live and by reading
``alphafold3/model/inference.py`` inside the image (2026-09-22), differs
from the official AlphaFold 3 docs' ``<output_dir>/<sanitised job name>/``
convention**: this entrypoint's own ``main()``, when called with
``--json_path`` (rather than ``--input_dir``) and
``--run_data_pipeline=false``, passes ``output_dir`` (``_OUTPUT_DIR.value``,
i.e. ``/output`` here) to ``process_fold_input`` DIRECTLY, with no extra
``<job_name>/`` nesting -- so the top-ranked structure lands at
``/output/job_model.cif`` (not ``/output/job/job_model.cif``), and each
seed/sample directory (``seed-<seed>_sample-<n>/``) sits directly under
``/output/`` too. The manifest's ``outputs:`` patterns match this real
layout, not the official docs' generic one.

``--run_data_pipeline=false`` is always passed -- AlphaFold 3 must never
search for its own alignment (see the manifest's "MSA is optional"
section). GPU selection (``--gpus device=7``) is hardcoded here, never a
parameter: this sibling container is a NEW top-level container on the
host, so it does NOT inherit whatever CDI GPU restriction pins this
server's own container to GPU 7 (design §2.1) -- that restriction has to be
re-applied explicitly at this container boundary too.

Reads one argv: a JSON object (see ``adapters/alphafold3.py`` for its exact
shape). Writes ``input.json`` and an ``out/`` directory into the current
working directory (the dispatcher's scratch workdir).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

JOB_NAME = "job"
INPUT_NAME = "input.json"
OUT_DIR = "out"

DOCKER_IMAGE = "romerolabduke/alphafast:latest"
MODEL_DIR_HOST = "/opt/alphafold3_data/weights"
# The image's OWN baked-in entrypoint -- NOT a host-mounted script. See this
# module's own docstring for why the ground-truth benchmark script's mounted
# copy cannot be used against this image.
RUN_ALPHAFOLD_IN_IMAGE = "/app/alphafold/run_alphafold.py"
GPU_DEVICE = "7"

_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _chain_ids(start: int, count: int) -> tuple[list[str], int]:
    """Consecutive single-letter chain ids starting at index ``start`` --
    same convention (and same practical size assumption) as
    ``scripts/engines/boltz.py``'s own ``_chain_ids``."""
    ids = [_LETTERS[i] for i in range(start, start + count)]
    return ids, start + count


def _build_input_json(job: dict) -> dict:
    sequences = []
    next_index = 0
    for chain in job["chains"]:
        ids, next_index = _chain_ids(next_index, chain["copies"])
        sequence_id: str | list[str] = ids[0] if len(ids) == 1 else ids
        sequences.append(
            {
                "protein": {
                    "id": sequence_id,
                    "sequence": chain["sequence"],
                    "unpairedMsa": chain["unpaired_msa"],
                    "pairedMsa": chain["paired_msa"],
                    "templates": [],
                }
            }
        )
    return {
        "name": JOB_NAME,
        "modelSeeds": job["seeds"],
        "sequences": sequences,
        "dialect": "alphafold3",
        "version": 2,
    }


def main() -> None:
    job = json.loads(sys.argv[1])

    Path(INPUT_NAME).write_text(json.dumps(_build_input_json(job)))
    Path(OUT_DIR).mkdir(exist_ok=True)

    cwd = str(Path(".").resolve())
    input_json_host = f"{cwd}/{INPUT_NAME}"
    out_dir_host = f"{cwd}/{OUT_DIR}"

    run_alphafold_args = [
        "python",
        RUN_ALPHAFOLD_IN_IMAGE,
        "--json_path=/input.json",
        "--output_dir=/output",
        "--model_dir=/models",
        "--run_data_pipeline=false",
        "--run_inference=true",
        f"--num_recycles={job['num_recycles']}",
        f"--num_diffusion_samples={job['num_diffusion_samples']}",
        f"--max_template_date={job['max_template_date']}",
        f"--resolve_msa_overlaps={'true' if job['resolve_msa_overlaps'] else 'false'}",
        f"--flash_attention_implementation={job['flash_attention_implementation']}",
        f"--save_embeddings={'true' if job['save_embeddings'] else 'false'}",
        f"--save_distogram={'true' if job['save_distogram'] else 'false'}",
        "--buckets=" + ",".join(str(b) for b in job["buckets"]),
    ]
    if job["conformer_max_iterations"] is not None:
        run_alphafold_args.append(
            f"--conformer_max_iterations={job['conformer_max_iterations']}"
        )

    inner_cmd = (
        "cd /alphafold3_venv && source bin/activate && "
        + " ".join(run_alphafold_args)
    )

    cmd = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        f"device={GPU_DEVICE}",
        "-v",
        f"{input_json_host}:/input.json:ro",
        "-v",
        f"{MODEL_DIR_HOST}:/models:ro",
        "-v",
        f"{out_dir_host}:/output",
        DOCKER_IMAGE,
        "bash",
        "-c",
        inner_cmd,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
