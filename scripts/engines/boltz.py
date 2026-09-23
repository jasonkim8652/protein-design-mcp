"""Wrapper for Boltz-2 (``run_boltz``). Runs inside the ``boltz`` environment
(the user's own editable fork at ``~/projects/lightning-boltz-dev``, not
upstream — see the tool's doc for why that matters for attribution).

Boltz takes one YAML "job" file, not a flat argv, and writes wherever
``--out_dir`` points using the job file's own stem as a directory component
(``<out_dir>/boltz_results_<stem>/predictions/<stem>/...``) — see
``boltz/main.py``'s ``out_dir = out_dir / f"boltz_results_{data.stem}"``.
This wrapper always names the job file ``job.yaml`` so that path is fixed and
the manifest's ``outputs:`` patterns can be static globs.

Boltz-2's affinity head is never invoked here (no ``properties: [affinity]``
block is ever written) — it is protein-ligand only and returns meaningless
numbers for a protein-protein interface, which is the reason this whole
tool exists as a fresh build rather than a copy of an older wrapper.

Reads one argv: a JSON object (see ``adapters/boltz.py`` for its exact
shape). Writes ``job.yaml`` into the current working directory (the
dispatcher's scratch workdir), invokes ``boltz predict``, and mirrors its
stdout/stderr/returncode so the adapter's error paths (OOM detection, etc.)
keep working unchanged.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

INPUT_NAME = "job.yaml"
OUT_DIR = "out"
_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _chain_ids(start: int, count: int) -> tuple[list[str], int]:
    """Consecutive single-letter chain ids starting at index ``start``.

    Boltz's own examples (``multimer.yaml``, ``ligand.yaml``) use bare
    uppercase letters, and every job here is small enough (a handful of
    chains) that running out of single letters is not a real concern; if it
    ever were, ``boltz predict`` would reject a malformed id itself rather
    than silently misbehaving.
    """
    ids = [_LETTERS[i] for i in range(start, start + count)]
    return ids, start + count


def _build_yaml(chains: list[dict]) -> dict:
    """Translate the adapter's normalised chain list into Boltz's YAML
    schema. Each ``chains[]`` entry with ``copies`` > 1 becomes ONE sequence
    block with a list of ids (Boltz's own homo-oligomer convention — see
    ``examples/ligand.yaml``'s ``id: [C, D]``), which also satisfies Boltz's
    own rule that "all proteins with the same sequence must share the same
    MSA" since every copy comes from the one chain entry.
    """
    sequences = []
    next_index = 0
    for chain in chains:
        ids, next_index = _chain_ids(next_index, chain["copies"])
        sequence_id: str | list[str] = ids[0] if len(ids) == 1 else ids
        msa = chain["msa"]
        sequences.append(
            {
                "protein": {
                    "id": sequence_id,
                    "sequence": chain["sequence"],
                    # "empty" is Boltz's own explicit single-sequence-mode
                    # value (see boltz/data/parse/schema.py) -- never the
                    # bare absence of the key, which Boltz reads as "auto"
                    # and would try to reach its own MSA server for.
                    "msa": msa if msa is not None else "empty",
                }
            }
        )
    return {"version": 1, "sequences": sequences}


def main() -> None:
    job = json.loads(sys.argv[1])

    yaml_doc = _build_yaml(job["chains"])
    Path(INPUT_NAME).write_text(yaml.safe_dump(yaml_doc, sort_keys=False))

    cmd = [
        "boltz",
        "predict",
        INPUT_NAME,
        "--out_dir",
        OUT_DIR,
        # Pinned: this tool is Boltz-2 specifically (spec: "Boltz-2 2.2.1"),
        # never Boltz-1. Never exposed as a parameter -- see the manifest.
        "--model",
        "boltz2",
        "--accelerator",
        "gpu",
        "--devices",
        "1",
        "--recycling_steps",
        str(job["recycling_steps"]),
        "--sampling_steps",
        str(job["sampling_steps"]),
        "--diffusion_samples",
        str(job["diffusion_samples"]),
        "--step_scale",
        str(job["step_scale"]),
        "--output_format",
        job["output_format"],
        "--max_msa_seqs",
        str(job["max_msa_seqs"]),
        "--num_subsampled_msa",
        str(job["num_subsampled_msa"]),
        "--seed",
        str(job["seed"]),
        # Never let Boltz reach its own MSA server or run local ColabFold
        # search: every chain's alignment (or deliberate absence) is
        # already fully specified in job.yaml above (policy: no tool here
        # may build its own MSA).
    ]
    if job["use_potentials"]:
        cmd.append("--use_potentials")
    # --subsample_msa is a plain click is_flag with no explicit default,
    # which click resolves to False when the flag is omitted -- despite the
    # CLI's own --help text claiming "Default is True." (verified against
    # boltz/main.py: @click.option("--subsample_msa", is_flag=True) with no
    # default=, immediately followed by a `predict()` signature default of
    # True that click's own flag parsing overrides). We trust the verified
    # runtime behaviour, not the stale help string.
    if job["subsample_msa"]:
        cmd.append("--subsample_msa")

    result = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
