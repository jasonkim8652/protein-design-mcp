"""Co-fold a target with RoseTTAFold3. Runs inside the `foundry` environment.

Builds RF3's own JSON input format (`rf3/utils/inference.py::InferenceInput.
from_json_dict`, read from source) from the adapter's JSON-serialized argv
(build_args has no access to a working directory -- see
``protein_design_mcp.adapters.rf3``'s module docstring for why this file has
to be written here instead): one `"components"` entry per chain
(`chain_type` hardcoded to `POLYPEPTIDE(L)` -- this tool is protein-only, see
the manifest), plus a top-level `"msa_paths"` dict keyed by `chain_id`,
straight from the caller-supplied paths (RF3 reads each path directly; no
hashing/staging convention like Promera's, confirmed from source).

Then invokes the `rf3` console script directly (on PATH inside this
environment -- the verified invocation from the install report).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

_TARGET_NAME = "target"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chains", required=True, help="JSON list of {chain_id, sequence}")
    parser.add_argument("--msa", required=True, help="JSON: null, or object: chain_id -> a3m path | null")
    parser.add_argument("--n-recycles", type=int, required=True)
    parser.add_argument("--diffusion-batch-size", type=int, required=True)
    parser.add_argument("--num-steps", type=int, required=True)
    args = parser.parse_args()

    chains = json.loads(args.chains)
    msa = json.loads(args.msa)

    components = [
        {"seq": chain["sequence"], "chain_type": "POLYPEPTIDE(L)", "chain_id": chain["chain_id"]}
        for chain in chains
    ]
    input_spec = {"name": _TARGET_NAME, "components": components}
    if msa is not None:
        input_spec["msa_paths"] = msa

    input_json = Path("input.json")
    input_json.write_text(json.dumps(input_spec))

    out_dir = Path("out")

    cmd = [
        "rf3",
        "fold",
        f"inputs={input_json}",
        f"out_dir={out_dir}",
        f"n_recycles={args.n_recycles}",
        f"diffusion_batch_size={args.diffusion_batch_size}",
        f"num_steps={args.num_steps}",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        sys.exit(proc.returncode)

    target_dir = out_dir / _TARGET_NAME
    cif_path = target_dir / f"{_TARGET_NAME}_model.cif"
    summary_path = target_dir / f"{_TARGET_NAME}_summary_confidences.json"
    if not cif_path.exists() or not summary_path.exists():
        raise RuntimeError(
            f"RF3 exited 0 but did not write the expected {cif_path.name}/"
            f"{summary_path.name} under {target_dir} (found: "
            f"{sorted(target_dir.iterdir()) if target_dir.exists() else 'nothing -- directory missing'})"
        )

    shutil.copy2(cif_path, "structure.cif")
    shutil.copy2(summary_path, "summary_confidences.json")


if __name__ == "__main__":
    main()
