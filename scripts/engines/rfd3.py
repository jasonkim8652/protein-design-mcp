"""Wrapper for RFdiffusion3 (`rfd3`, part of RosettaCommons' `rc-foundry`).
Runs inside the `foundry` environment and is shared by BOTH
`run_rfdiffusion3_binder` and `run_rfdiffusion3_scaffold` -- both tools drive
the exact same `rfd3 design inputs=<json> out_dir=<dir>` entry point, and
differ only in which fields their adapters put into the one JSON design
specification. See `adapters.rfdiffusion3_binder` and
`adapters.rfdiffusion3_scaffold` for the field-by-field shape each tool
builds; this wrapper does not know or care which tool called it.

RFdiffusion3's own conditioning schema (`rfd3.inference.input_parsing.
DesignInputSpecification`, a pydantic model with `extra="forbid"`) is one
JSON object per design job, given as a *path* to a JSON file
(`{job_name: {...spec fields...}}`) -- not inline on the CLI. Diffusion
sampling knobs (batch size, timestep count, step scale, seed) are SEPARATE
top-level Hydra overrides, not part of that JSON object at all. Reads one
argv: a JSON object `{"job": {...DesignInputSpecification fields, gaps
omitted...}, "engine": {"diffusion_batch_size": int, "num_timesteps": int,
"step_scale": float, "seed": int | null}}`. Writes `input.json` (job key
always literally "job") into the scratch workdir, then invokes `rfd3
design`. Output filenames are therefore always
`input_job_<batch>_model_<model>.cif.gz` (+ `.json` sidecar) --
deterministic because this wrapper always names the job file `input.json`
and the job key `job` -- which is why the manifests' `outputs:` patterns can
be static globs.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

INPUT_NAME = "input.json"
OUT_DIR = "out"
JOB_KEY = "job"


def main() -> None:
    payload = json.loads(sys.argv[1])
    job = payload["job"]
    engine = payload["engine"]

    Path(INPUT_NAME).write_text(json.dumps({JOB_KEY: job}))

    cmd = [
        "rfd3",
        "design",
        f"inputs={INPUT_NAME}",
        f"out_dir={OUT_DIR}",
        f"diffusion_batch_size={engine['diffusion_batch_size']}",
        f"inference_sampler.num_timesteps={engine['num_timesteps']}",
        f"inference_sampler.step_scale={engine['step_scale']}",
    ]
    if engine.get("seed") is not None:
        cmd.append(f"seed={engine['seed']}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
