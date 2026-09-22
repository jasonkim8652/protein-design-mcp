"""Adapter for ipSAE (DunbrackLab/IPSAE, PyPI `ipsae`).

ipSAE prints a whitespace-aligned table to stdout with one row per ordered
chain pair. We return the first row; a caller wanting a specific pair should
score that pair's structure.
"""

from __future__ import annotations

import re
from typing import Any

from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.schema import Manifest

# Chn1 Chn2  ipSAE  ipTM_af  pDockQ  LIS
_ROW_RE = re.compile(
    r"^\s*([A-Za-z0-9]+)\s+([A-Za-z0-9]+)\s+"
    r"([\d.]+)\s+([\d.]+)\s+([\d.]+)",
    re.M,
)


def build_args(manifest: Manifest, params: dict[str, Any]) -> list[str]:
    """Translate validated parameters into ipSAE's argv.

    ``manifest`` is unused here — ipSAE backs exactly one tool — but is part
    of every adapter's signature (app.ADAPTERS is keyed on manifest.name, not
    manifest.engine.repo) so a future adapter module backing several tools on
    one engine repo can branch on ``manifest.name`` without changing the call
    site.
    """
    del manifest
    return [
        str(params["pae_json"]),
        str(params["structure"]),
        str(params["pae_cutoff"]),
        str(params["dist_cutoff"]),
    ]


def parse_output(manifest: Manifest, run: CompletedRun) -> dict[str, Any]:
    """Extract the first chain-pair row from ipSAE's table. ``manifest`` is
    unused (see ``build_args``)."""
    del manifest
    for match in _ROW_RE.finditer(run.stdout):
        chain1, chain2, ipsae, iptm, pdockq = match.groups()
        if chain1.lower() == "chn1":
            continue
        return {
            "chain_pair": f"{chain1}_{chain2}",
            "ipsae": float(ipsae),
            "iptm_af": float(iptm),
            "pdockq": float(pdockq),
        }
    raise ValueError(
        "ipSAE produced no chain-pair row. Output was:\n"
        f"{run.stdout.strip()[-1000:]}"
    )
