"""Wrapper for AlphaFold 3 (``run_alphafold3``). Runs UNDER the mounted
``/alphafold3_venv`` prefix's own python (``micromamba run -p
/alphafold3_venv python /app/scripts/engines/alphafold3.py <job-json>`` --
see the manifest's ``engine:`` block), exactly like every other GPU engine
in this server now. This wrapper itself needs nothing beyond the Python
standard library (json, subprocess, pathlib) -- it never imports
``alphafold3`` directly, it only builds the job JSON and shells out to the
mounted venv's own inference entrypoint as a plain LOCAL subprocess.

**This changed in task 16.** AlphaFold 3 used to be the one tool in this
server that shelled out to ``docker run``, launching a SIBLING container
from ``romerolabduke/alphafast:latest`` -- see git history for that
version, and ``run_alphafold3.yaml``'s own ``engine:`` comment and "How
this tool is dispatched" doc section for the full reasoning (bind-mounting
the host's Docker socket into this server's own container is a
root-equivalent capability, forbidden by Ruling 3 given this server's
unauthenticated HTTP transport). There is no docker invocation anywhere in
this file any more, and none is needed: the venv (and the entrypoint
script it now carries alongside it -- see below) was extracted from that
same image with ``docker create`` + ``docker cp`` and is mounted in like
any other GPU engine's conda environment.

Ground truth for the invocation shape is the user's own checkout,
``~/projects/af3-mmseqs-gpu`` (read-only) --
``benchmarks/run_inference_original_db.sh`` for the argument shape and
in-venv activation sequence, ``docs/input.md`` for the AlphaFold 3 JSON
schema, ``docs/output.md`` for the output directory layout this wrapper's
manifest ``outputs:`` patterns are built from.

**This wrapper runs the image's OWN baked-in entrypoint -- NOT the
ground-truth benchmark script's copy.** CONFIRMED LIVE, 2026-09-22 (and
reconfirmed after extraction, 2026-09-23): mounting the host's own
``~/projects/af3-mmseqs-gpu/run_alphafold.py`` in, exactly as
``benchmarks/run_inference_original_db.sh`` does, fails immediately with
``ModuleNotFoundError: No module named 'alphafold3.jax.attention'`` --
that host script's top-level ``from alphafold3.jax.attention import
attention`` (added by a LATER point in the same RomeroLab fork's history
than this venv was extracted from) has no matching module in the
``alphafold3`` package this venv actually carries (confirmed by listing
``<venv>/app/alphafold/src/alphafold3/jax/``: only a ``geometry``
subpackage, no ``attention`` one). The extracted entrypoint IS genuinely
the same fork family -- same flag surface (``--run_data_pipeline``,
``--num_recycles``, ``--num_diffusion_samples``, ``--resolve_msa_overlaps``,
``--flash_attention_implementation``, ``--buckets``,
``--conformer_max_iterations``, ...) built against ``tokamax``/
``ModelRunner`` instead -- and is guaranteed self-consistent with the
package actually installed in this venv, so this wrapper uses it directly
rather than mounting anything from the host repo at all.

``RUN_ALPHAFOLD_ENTRY`` is ``/alphafold3_venv/app/alphafold/run_alphafold.py``
-- NOT ``/app/alphafold/run_alphafold.py``, the path it sat at INSIDE
``romerolabduke/alphafast:latest``. Extraction moved ``/app/alphafold``
to live UNDER the venv tree instead (see the manifest's ``engine:``
comment for the full reasoning: this lets one relocated mount
(``engine.prefix_host``) cover both the venv and its entrypoint, instead
of needing a second, independently-relocated ``engine.mounts`` entry) --
CONFIRMED LIVE, 2026-09-23, both the entrypoint AND its own
``alphafold3`` package import correctly from this new location, including
after the accompanying editable-install redirect table
(``_alphafast_editable.py``/``.pth``) was hand-patched at extraction time
to match.

``MODEL_DIR`` is ``/opt/alphafold3_data/weights`` (contains
``af3.bin``/``af3.bin.zst`` -- note the identically-named
``/opt/alphafold3_data/models`` is empty and is NOT this path); AlphaFold
3's own ``params.select_model_files`` matches ``af3.bin.zst`` first and
returns just that one file, so having both the raw and compressed weight
file in the same directory does not trigger its "Multiple models matched"
error (confirmed by reading ``alphafold3/model/params.py`` directly out of
the extracted venv, 2026-09-22). This mount is IDENTICAL host and
container path (``/opt/alphafold3_data/weights`` on both sides, per
``engine.mounts`` -- see EngineSpec's own docstring on why only
``engine.prefix`` ever needs ``prefix_host``'s relocation, never
``mounts``), so no path translation is needed here at all.

The job is always named ``"job"`` (never derived from caller input).
**Output layout, confirmed live and by reading
``alphafold3/model/inference.py`` inside the extracted venv (2026-09-22),
differs from the official AlphaFold 3 docs' ``<output_dir>/<sanitised job
name>/`` convention**: this entrypoint's own ``main()``, when called with
``--json_path`` (rather than ``--input_dir``) and
``--run_data_pipeline=false``, passes ``output_dir`` (``_OUTPUT_DIR.value``)
to ``process_fold_input`` DIRECTLY, with no extra ``<job_name>/`` nesting --
so the top-ranked structure lands at ``<OUT_DIR>/job_model.cif`` (not
``<OUT_DIR>/job/job_model.cif``), and each seed/sample directory
(``seed-<seed>_sample-<n>/``) sits directly under ``<OUT_DIR>/`` too. The
manifest's ``outputs:`` patterns match this real layout, not the official
docs' generic one.

``--run_data_pipeline=false`` is always passed -- AlphaFold 3 must never
search for its own alignment (see the manifest's "MSA is optional"
section). GPU selection needs NO flag here at all any more: the sibling
``docker run`` this wrapper used to shell out to needed its OWN
``--gpus device=7`` re-applied at ITS OWN container boundary, since it was
a brand new top-level container that did not inherit this server's own
CDI GPU restriction (design §2.1). That second container boundary is
gone -- this subprocess runs directly inside THIS server's own container,
which is already pinned to exactly GPU 7 (``--device=nvidia.com/gpu=7``,
see ``scripts/container_run.py``), so the entrypoint's default
``--gpu_device=0`` (the only GPU this process can ever see) is already
correct, unconditionally.

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

MODEL_DIR = "/opt/alphafold3_data/weights"
# Lives UNDER the mounted venv now, not beside it -- see this module's own
# docstring for why extraction moved it there.
RUN_ALPHAFOLD_ENTRY = "/alphafold3_venv/app/alphafold/run_alphafold.py"

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

    cwd = Path(".").resolve()
    input_json_path = str(cwd / INPUT_NAME)
    out_dir_path = str(cwd / OUT_DIR)

    run_alphafold_args = [
        "python",
        RUN_ALPHAFOLD_ENTRY,
        f"--json_path={input_json_path}",
        f"--output_dir={out_dir_path}",
        f"--model_dir={MODEL_DIR}",
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

    # "python" resolves via inherited PATH -- micromamba run -p
    # /alphafold3_venv already put the venv's own bin/ ahead of everything
    # else for THIS process, and a subprocess inherits that same PATH, so
    # this is the SAME interpreter the venv's own console scripts use.
    result = subprocess.run(run_alphafold_args, capture_output=True, text=True)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
