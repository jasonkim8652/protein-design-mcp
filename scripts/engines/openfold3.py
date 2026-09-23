"""Wrapper for OpenFold3 (``run_openfold3``). Runs inside the ``openfold3``
environment (plain PyPI install, ``pip install openfold3`` -> 0.5.0 today,
not editable).

OpenFold3 takes a query-set JSON (``InferenceQuerySet`` /
``openfold3/projects/of3_all_atom/config/inference_query_format.py``), not
a flat argv. Each ``Chain`` has its own optional ``main_msa_file_paths``,
but a supplied a3m cannot be handed to it under an arbitrary filename:
``core/data/io/sequence/msa.py:parse_msas_direct`` silently SKIPS any file
whose basename (stem) is not one of a fixed set of recognised database
names (``dataset_config_components.py``'s ``MSASettings.max_seq_counts``
keys -- ``uniref90_hits``, ``colabfold_main``, etc.), which leaves the
parsed MSA dict empty and crashes downstream with an ``IndexError`` on
``sorted(all_msas_per_chain.keys())[0]`` -- CONFIRMED live: passing an
arbitrarily-named a3m reproduces exactly this crash. This wrapper therefore
always copies the caller's a3m into a file named ``colabfold_main.a3m``
(the generic "one merged unpaired alignment" entry in that recognised set)
before referencing it.

``--use_msa_server`` is passed as ``false`` unconditionally: the ONLY place
OpenFold3 ever fetches an alignment itself is
``PredictDataModule.prepare_data``'s ``if self.use_msa_server:
preprocess_colabfold_msas(...)`` (verified from
``openfold3/core/data/framework/data_module.py``), so with that flag off a
chain simply runs single-sequence whenever it has no
``main_msa_file_paths`` -- there is no other native fetch path to guard
against here, unlike Boltz or Protenix.

Reads one argv: a JSON object (see ``adapters/openfold3.py`` for its shape).
"""

from __future__ import annotations

import json
import shutil
import string
import subprocess
import sys
from pathlib import Path

INPUT_JSON = "query.json"
OUT_DIR = "out"
MSA_DIR = Path("msas")
QUERY_NAME = "job"
_LETTERS = string.ascii_uppercase
# The one entry in MSASettings.max_seq_counts (see this module's docstring)
# that best matches "one merged, unpaired alignment from an external
# search" -- the shape run_mmseqs_search's unpaired_a3m produces.
_RECOGNISED_MAIN_MSA_BASENAME = "colabfold_main"


def _chain_ids(start: int, count: int) -> tuple[list[str], int]:
    ids = [_LETTERS[i] for i in range(start, start + count)]
    return ids, start + count


def _stage_main_msa(sequence_index: int, source_path: str) -> str:
    """Copy the caller's a3m into a recognised-basename file (see this
    module's docstring) and return that copy's path."""
    chain_dir = MSA_DIR / f"chain{sequence_index}"
    chain_dir.mkdir(parents=True, exist_ok=True)
    dest = chain_dir / f"{_RECOGNISED_MAIN_MSA_BASENAME}.a3m"
    shutil.copyfile(source_path, dest)
    return str(dest.resolve())


def _build_chains(chains: list[dict]) -> list[dict]:
    out = []
    next_index = 0
    for sequence_index, chain in enumerate(chains):
        ids, next_index = _chain_ids(next_index, chain["copies"])
        entry = {
            "molecule_type": "PROTEIN",
            "chain_ids": ids,
            "sequence": chain["sequence"],
        }
        if chain["msa"] is not None:
            entry["main_msa_file_paths"] = [
                _stage_main_msa(sequence_index, chain["msa"])
            ]
        out.append(entry)
    return out


def main() -> None:
    job = json.loads(sys.argv[1])

    query_doc = {
        "seeds": job["seeds"],
        "queries": {
            QUERY_NAME: {
                "chains": _build_chains(job["chains"]),
                "use_msas": True,
                "use_paired_msas": True,
                "use_main_msas": True,
            }
        },
    }
    Path(INPUT_JSON).write_text(json.dumps(query_doc))

    cmd = [
        "run_openfold",
        "predict",
        f"--query_json={INPUT_JSON}",
        f"--output_dir={OUT_DIR}",
        f"--num_diffusion_samples={job['num_diffusion_samples']}",
        # Never let OpenFold3 reach ColabFold's MSA server -- every
        # chain's alignment (or deliberate absence) is already fully
        # specified in query.json above.
        "--use_msa_server=false",
        "--use_templates=false",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
