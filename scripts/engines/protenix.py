"""Wrapper for Protenix v1 (``run_protenix``). Runs inside the ``protenix``
environment (plain PyPI install, ``pip install protenix`` -> 2.0.0 today,
not editable).

Protenix takes an AF3-server-style JSON job list, not a flat argv, and its
own ``--use_msa`` flag is a single job-wide switch: if ANY protein chain in
the job lacks a valid ``unpairedMsaPath`` (or ``unpairedMsa``/``pairedMsaPath``)
AND ``--use_msa`` is true, Protenix's own ``runner.msa_search.msa_search``
calls out to ``https://protenix-server.com/api/msa`` -- a REMOTE service
(verified from ``protenix/web_service/colab_request_parser.py``'s
``MMSEQS_SERVICE_HOST_URL``). That is exactly the kind of native alignment
fetch this server's tools must never trigger.

So every protein chain here ALWAYS gets an ``unpairedMsaPath`` written --
either the caller's own supplied a3m (resolved to an absolute path by the
adapter) or a placeholder single-record, query-only a3m this wrapper writes
for a deliberately MSA-free chain (``msa: null``). Because every chain then
has an existing path, Protenix's own ``need_msa_search`` check
(``runner/msa_search.py``) never fires and ``--use_msa true`` is safe to
pass unconditionally.

Reads one argv: a JSON object (see ``adapters/protenix.py`` for its shape).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

INPUT_JSON = "job.json"
OUT_DIR = "out"
MSA_DIR = Path("msas")
JOB_NAME = "job"
MODEL_NAME = "protenix_base_default_v1.0.0"


def _write_placeholder_a3m(sequence: str, index: int) -> str:
    """A query-only a3m record, so Protenix's own need_msa_search() sees an
    existing, valid path and never falls back to its remote search -- see
    this module's docstring."""
    MSA_DIR.mkdir(exist_ok=True)
    path = MSA_DIR / f"empty_{index}.a3m"
    path.write_text(f">query\n{sequence}\n")
    return str(path.resolve())


def _build_job(chains: list[dict]) -> dict:
    sequences = []
    for index, chain in enumerate(chains):
        msa_path = chain["msa"] or _write_placeholder_a3m(chain["sequence"], index)
        sequences.append(
            {
                "proteinChain": {
                    "sequence": chain["sequence"],
                    "count": chain["copies"],
                    "unpairedMsaPath": msa_path,
                }
            }
        )
    return {"name": JOB_NAME, "sequences": sequences}


def main() -> None:
    job = json.loads(sys.argv[1])

    job_doc = [_build_job(job["chains"])]
    Path(INPUT_JSON).write_text(json.dumps(job_doc))

    seeds_csv = ",".join(str(s) for s in job["seeds"])

    cmd = [
        "protenix",
        "pred",
        "-i",
        INPUT_JSON,
        "-o",
        OUT_DIR,
        # Pinned explicitly, never the bare default -- see the manifest's
        # doc for why (pip install protenix's default model_name can
        # change in a future release; the proprietary v2 must never be
        # selected).
        "-n",
        MODEL_NAME,
        "-c",
        str(job["cycle"]),
        "-p",
        str(job["step"]),
        "-e",
        str(job["sample"]),
        "-d",
        job["dtype"],
        "-s",
        seeds_csv,
        "--use_msa",
        "true",
        "--use_template",
        "false",
        "--use_rna_msa",
        "false",
        "--need_atom_confidence",
        "true" if job["need_atom_confidence"] else "false",
        # Never --use_msa false: every chain already has an unpairedMsaPath
        # (real or placeholder) above, so this only controls whether those
        # paths are honoured, never whether Protenix reaches its own
        # server (see this module's docstring).
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
