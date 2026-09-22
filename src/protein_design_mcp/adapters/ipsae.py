"""Adapter for ipSAE (DunbrackLab/IPSAE, PyPI `ipsae`).

ipSAE prints a whitespace-aligned table to stdout with one row per ordered
chain pair, and a wide, version-dependent column set (real header, e.g.:
``Chn1 Chn2  PAE Dist  Type   ipSAE  ipSAE_d0chn  ipSAE_d0dom  ipTM_af
ipTM_d0chn  pDockQ  pDockQ2  LIS  n0res  n0chn  n0dom  d0res  d0chn  d0dom
nres1  nres2  dist1  dist2  Model``). We locate the header structurally and
read columns by NAME rather than by fixed position, because column order and
count are not part of any stable contract, and some columns (e.g. ``PAE``,
``Type``) are not numeric. We return the first data row; a caller wanting a
specific pair should score that pair's structure. For a complex with more
than two chains, ipSAE emits one row per chain pair and only the first is
returned here (also documented in the manifest's "What you get back").
"""

from __future__ import annotations

from typing import Any

from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.schema import Manifest

# Columns this adapter needs by name. ipSAE's real output carries many more
# columns (PAE, Dist, Type, ipSAE_d0chn, ipSAE_d0dom, ipTM_d0chn, pDockQ2,
# LIS, n0res, ...) that we deliberately ignore rather than assume are numeric
# or present in a fixed position.
_CHAIN1_COL = "Chn1"
_CHAIN2_COL = "Chn2"
_IPSAE_COL = "ipSAE"
_IPTM_COL = "ipTM_af"
_PDOCKQ_COL = "pDockQ"
_REQUIRED_COLUMNS = (_CHAIN1_COL, _CHAIN2_COL, _IPSAE_COL, _IPTM_COL, _PDOCKQ_COL)


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


def _find_header(lines: list[str]) -> tuple[int, list[str]]:
    """Find the header line: the first line whose whitespace-split tokens
    include every required column name exactly (not as a substring — e.g.
    ``pDockQ2`` must not satisfy a search for ``pDockQ``)."""
    for index, line in enumerate(lines):
        tokens = line.split()
        if all(col in tokens for col in _REQUIRED_COLUMNS):
            return index, tokens
    joined = "\n".join(lines).strip()[-1000:]
    raise ValueError(
        "ipSAE produced no header row containing the expected columns "
        f"({', '.join(_REQUIRED_COLUMNS)}). Output was:\n{joined}"
    )


def parse_output(manifest: Manifest, run: CompletedRun) -> dict[str, Any]:
    """Extract the first chain-pair row from ipSAE's table, reading columns
    by name. ``manifest`` is unused (see ``build_args``)."""
    del manifest
    lines = run.stdout.splitlines()
    header_line_index, columns = _find_header(lines)
    column_index = {name: position for position, name in enumerate(columns)}
    required_index = {name: column_index[name] for name in _REQUIRED_COLUMNS}
    min_len = max(required_index.values()) + 1

    for line in lines[header_line_index + 1 :]:
        tokens = line.split()
        if len(tokens) < min_len:
            continue
        try:
            ipsae = float(tokens[required_index[_IPSAE_COL]])
            iptm = float(tokens[required_index[_IPTM_COL]])
            pdockq = float(tokens[required_index[_PDOCKQ_COL]])
        except ValueError:
            # A data row whose required cells aren't numeric (rare, but the
            # non-required columns like PAE/Type are allowed to be
            # placeholders) — keep looking rather than crash on it.
            continue
        chain1 = tokens[required_index[_CHAIN1_COL]]
        chain2 = tokens[required_index[_CHAIN2_COL]]
        return {
            "chain_pair": f"{chain1}_{chain2}",
            "ipsae": ipsae,
            "iptm_af": iptm,
            "pdockq": pdockq,
        }

    raise ValueError(
        "ipSAE produced no chain-pair row. Output was:\n"
        f"{run.stdout.strip()[-1000:]}"
    )
