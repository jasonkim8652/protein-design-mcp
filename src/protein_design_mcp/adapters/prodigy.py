"""Adapter for PRODIGY (prodigy-prot).

PRODIGY writes its results to stdout as ``[+] key: value`` lines rather than a
machine-readable file, so the adapter parses stdout.
"""

from __future__ import annotations

import re
from typing import Any

from protein_design_mcp.dispatch.env import CompletedRun

CAVEAT = (
    "PRODIGY is calibrated on natural complexes and systematically mis-ranks "
    "de novo designed binders. Use it as a sanity floor, not as a ranking "
    "criterion."
)

_AFFINITY_RE = re.compile(r"binding affinity \(kcal\.mol-1\):\s*(-?[\d.]+)", re.I)
_KD_RE = re.compile(r"dissociation constant \(M\)[^:]*:\s*([\d.eE+-]+)", re.I)
_CONTACTS_RE = re.compile(r"intermolecular contacts:\s*(\d+)", re.I)


def build_args(params: dict[str, Any]) -> list[str]:
    """Translate validated parameters into PRODIGY's argv."""
    return [
        str(params["complex_pdb"]),
        "--selection",
        str(params["chain_a"]),
        str(params["chain_b"]),
        "--temperature",
        str(params["temperature"]),
    ]


def parse_output(run: CompletedRun) -> dict[str, Any]:
    """Extract PRODIGY's numbers from stdout."""
    affinity = _AFFINITY_RE.search(run.stdout)
    if affinity is None:
        raise ValueError(
            "PRODIGY produced no binding affinity line. Output was:\n"
            f"{run.stdout.strip()[-1000:]}"
        )

    kd = _KD_RE.search(run.stdout)
    contacts = _CONTACTS_RE.search(run.stdout)
    return {
        "binding_affinity_kcal_per_mol": float(affinity.group(1)),
        "dissociation_constant_M": float(kd.group(1)) if kd else None,
        "intermolecular_contacts": int(contacts.group(1)) if contacts else None,
        "caveat": CAVEAT,
    }
