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

The job is always named ``"job"`` (never derived from caller input) so the
output directory (``AlphaFold 3 writes into
<output_dir>/<sanitised job name>/``) is fixed and predictable, the same
convention ``scripts/engines/boltz.py`` already uses for its own job file.

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

DOCKER_IMAGE = "alphafold3-mmseqs:latest"
MODEL_DIR_HOST = "/opt/alphafold3_data/weights"
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
        "/run_alphafold.py",
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
