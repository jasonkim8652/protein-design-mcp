"""Adapter for ipSAE (DunbrackLab/IPSAE, PyPI `ipsae`).

ipSAE writes a whitespace-aligned table with one row per ordered chain pair,
and a wide, version-dependent column set (real header, e.g.:
``Chn1 Chn2  PAE Dist  Type   ipSAE  ipSAE_d0chn  ipSAE_d0dom  ipTM_af
ipTM_d0chn  pDockQ  pDockQ2  LIS  n0res  n0chn  n0dom  d0res  d0chn  d0dom
nres1  nres2  dist1  dist2  Model``). We locate the header structurally and
read columns by NAME rather than by fixed position, because column order and
count are not part of any stable contract, and some columns (e.g. ``PAE``,
``Type``) are not numeric. We return the first data row; a caller wanting a
specific pair should score that pair's structure. For a complex with more
than two chains, ipSAE emits one row per chain pair and only the first is
returned here (also documented in the manifest's "What you get back").

SETTLED LIVE (Task 7 fix round 1): ``ipsae==1.0.1``'s only entry point
(``ipsae.cli:main``) never prints that table to stdout at all — it WRITES
it, along with a by-residue detail file and a PyMOL script, into the
STRUCTURE file's own directory (``_save_outputs`` in ``ipsae/core.py``
always uses ``structure_file.parent`` — there is no flag to redirect it).
The manifest now stages ``structure`` into the dispatcher's scratch
directory first (``engine.stage: ["structure"]`` — see
``protein_design_mcp.staging``) and declares ``results_txt`` (``multiple:
true``, since the by-residue file lands next to it and cannot be excluded
by a static glob) as an output, so ``run.outputs["results_txt"]`` is a list
of collected file paths. This adapter now reads THOSE files instead of
``run.stdout``. The header-name column lookup below (``_find_header`` and
the row-extraction loop) is otherwise unchanged from what was already
verified correct against the real table format — only where the text comes
from changed. Reading by header name doubles as the mechanism that picks
the right file out of the two collected: the by-residue file's columns
never satisfy ``_REQUIRED_COLUMNS``, so ``_find_header`` naturally skips it.
"""

from __future__ import annotations

from pathlib import Path
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


def _extract_row(lines: list[str], header_line_index: int, columns: list[str]) -> dict[str, Any]:
    """Extract the first data row after the header, reading columns by name.

    Unchanged from the pre-staging version except that ``lines`` now comes
    from a collected file's text instead of ``run.stdout``.
    """
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

    raise ValueError("ipSAE produced no chain-pair row.")


def parse_output(manifest: Manifest, run: CompletedRun) -> dict[str, Any]:
    """Extract the first chain-pair row from ipSAE's results file, reading
    columns by name. ``manifest`` is unused (see ``build_args``).

    ``run.outputs["results_txt"]`` is always a list (the manifest's
    ``results_txt`` output sets ``multiple: true``, since ipSAE also writes
    a by-residue detail file next to the results table and no static glob
    can exclude it by name). Each collected file is tried in turn; the
    by-residue file is skipped because its columns never satisfy
    ``_REQUIRED_COLUMNS``, so ``_find_header`` raises for it and this loop
    moves on. It is a genuine failure — not "try the next file" — if the
    table file IS found but contains no valid data row: that means ipSAE's
    own output is malformed, not that we picked the wrong candidate.
    """
    del manifest
    result_paths = run.outputs.get("results_txt")
    if not result_paths:
        raise ValueError(
            "ipSAE's declared 'results_txt' output was not collected — no "
            f"results file was found. run.outputs was: {run.outputs}"
        )

    header_errors: list[str] = []
    for path in result_paths:
        lines = Path(path).read_text().splitlines()
        try:
            header_line_index, columns = _find_header(lines)
        except ValueError as exc:
            header_errors.append(f"{path}: {exc}")
            continue
        return _extract_row(lines, header_line_index, columns)

    raise ValueError(
        "ipSAE produced no header row containing the expected columns "
        f"({', '.join(_REQUIRED_COLUMNS)}) in any collected output file: "
        + "; ".join(header_errors)
    )
