"""Adapter for PRODIGY (prodigy-prot).

PRODIGY writes its results to stdout as ``[+] key: value`` lines rather than a
machine-readable file, so the adapter parses stdout.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.schema import Manifest

logger = logging.getLogger(__name__)

CAVEAT = (
    "PRODIGY is calibrated on natural complexes and systematically mis-ranks "
    "de novo designed binders. Use it as a sanity floor, not as a ranking "
    "criterion."
)

_AFFINITY_RE = re.compile(r"binding affinity \(kcal\.mol-1\):\s*(-?[\d.]+)", re.I)
_KD_RE = re.compile(r"dissociation constant \(M\)[^:]*:\s*([\d.eE+-]+)", re.I)
_CONTACTS_RE = re.compile(r"intermolecular contacts:\s*(\d+)", re.I)


def build_args(manifest: Manifest, params: dict[str, Any]) -> list[str]:
    """Translate validated parameters into PRODIGY's argv.

    ``manifest`` is unused here — PRODIGY backs exactly one tool — but is
    part of every adapter's signature (app.ADAPTERS is keyed on
    manifest.name, not manifest.engine.repo) so a future adapter module
    backing several tools on one engine repo can branch on
    ``manifest.name`` without changing the call site.
    """
    del manifest
    return [
        str(params["complex_pdb"]),
        "--selection",
        str(params["chain_a"]),
        str(params["chain_b"]),
        "--temperature",
        str(params["temperature"]),
    ]


def parse_output(manifest: Manifest, run: CompletedRun) -> dict[str, Any]:
    """Extract PRODIGY's numbers from stdout. ``manifest`` is unused (see
    ``build_args``)."""
    del manifest
    affinity = _AFFINITY_RE.search(run.stdout)
    if affinity is None:
        raise ValueError(
            "PRODIGY produced no binding affinity line. Output was:\n"
            f"{run.stdout.strip()[-1000:]}"
        )

    kd = _KD_RE.search(run.stdout)
    if kd is None:
        logger.warning(
            "PRODIGY output included binding affinity but not dissociation "
            "constant; the output format may have changed."
        )
    contacts = _CONTACTS_RE.search(run.stdout)
    return {
        "binding_affinity_kcal_per_mol": float(affinity.group(1)),
        "dissociation_constant_M": float(kd.group(1)) if kd else None,
        "intermolecular_contacts": int(contacts.group(1)) if contacts else None,
        "caveat": CAVEAT,
    }
