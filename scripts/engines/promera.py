"""Co-fold a target with Promera. Runs inside the `promera` environment.

Builds Promera's own on-disk conventions from the adapter's JSON-serialized
argv (build_args has no access to a working directory -- see
``protein_design_mcp.adapters.promera``'s module docstring for why this
staging has to happen here instead):

- a single target schema JSON file, in its own input directory
  (`python -m promera` takes a DIRECTORY of `*.json` schemas, one target
  per file -- this wrapper always writes exactly one);
- an MSA directory keyed by `tinyprot.msa.hash_sequence(seq)` per chain,
  populated only for chains whose caller-supplied a3m is not null -- a
  chain absent from that directory falls back to Promera's own
  single-sequence dummy MSA automatically (verified from
  `tinyprot.msa.load_msa_from_dir`/`make_dummy_msa` source), which is
  exactly what `assert_msa=False` (always passed here) is relying on.

Then invokes `python -m promera` as a subprocess under the SAME interpreter
this script is already running under (`sys.executable`), so it dispatches
into the identical environment/weights this wrapper was launched in.
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
    parser.add_argument("--schema", required=True, help="JSON object: chain label -> schema entry")
    parser.add_argument("--msa", required=True, help="JSON: null, or object: chain label -> a3m path | null")
    parser.add_argument("--recycling-steps", type=int, required=True)
    parser.add_argument("--diffusion-samples", type=int, required=True)
    parser.add_argument("--diffusion-steps", type=int, required=True)
    parser.add_argument("--num-seeds", type=int, required=True)
    args = parser.parse_args()

    chains = json.loads(args.schema)
    msa = json.loads(args.msa)

    input_dir = Path("schemas")
    input_dir.mkdir(exist_ok=True)
    (input_dir / f"{_TARGET_NAME}.json").write_text(json.dumps(chains))

    msa_dir = Path("msa_dir")
    msa_dir.mkdir(exist_ok=True)
    if msa is not None:
        from tinyprot.msa import hash_sequence

        for chain_label, a3m_path in msa.items():
            if a3m_path is None:
                continue
            sequence = chains[chain_label]["sequence"]
            digest = hash_sequence(sequence)
            shutil.copy2(a3m_path, msa_dir / f"{digest}.a3m")

    out_dir = Path("out")

    cmd = [
        sys.executable,
        "-m",
        "promera",
        f"input={input_dir}",
        f"output={out_dir}",
        f"msa_dir={msa_dir}",
        "assert_msa=False",
        f"recycling_steps={args.recycling_steps}",
        f"diffusion_samples={args.diffusion_samples}",
        f"diffusion_steps={args.diffusion_steps}",
        f"num_seeds={args.num_seeds}",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        sys.exit(proc.returncode)

    target_dir = out_dir / _TARGET_NAME
    cif_files = sorted(target_dir.glob(f"{_TARGET_NAME}_seed*_samp*.cif"))
    conf_files = sorted(target_dir.glob(f"{_TARGET_NAME}_seed*_samp*_conf.json"))
    if not cif_files or not conf_files:
        raise RuntimeError(
            f"Promera exited 0 but wrote no structure/confidence files under "
            f"{target_dir} (found: {sorted(target_dir.iterdir()) if target_dir.exists() else 'nothing -- directory missing'})"
        )

    # skip_existing/num_seeds/diffusion_samples can all multiply the number
    # of *_seed*_samp*.cif files Promera writes for one target; this tool's
    # `outputs:` collects exactly one structure and one confidence file, so
    # the first (seed 0, sample 0, alphabetically first) is what is copied
    # out -- callers wanting every sample/seed should call this tool once
    # per seed instead.
    shutil.copy2(cif_files[0], "structure.cif")
    shutil.copy2(conf_files[0], "confidence.json")


if __name__ == "__main__":
    main()
