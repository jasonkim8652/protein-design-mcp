"""The persistent home for files engines produce.

An engine writes into a scratch directory that is removed after the run. Any
file the caller needs must therefore be declared in the manifest's ``outputs:``
and copied out first. Undeclared files go away with the workdir, which is what
makes the declaration load-bearing rather than documentation.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Sequence
from pathlib import Path
from tempfile import gettempdir

from protein_design_mcp.manifest.schema import OutputSpec


def results_dir() -> Path:
    """Where collected outputs live. Override with PROTEIN_MCP_RESULTS_DIR."""
    override = os.environ.get("PROTEIN_MCP_RESULTS_DIR")
    if override:
        return Path(override)
    return Path(gettempdir()) / "pdmcp-results"


def collect_outputs(
    specs: Sequence[OutputSpec],
    workdir: Path,
    run_id: str,
) -> dict[str, str]:
    """Copy each declared output out of ``workdir``. Returns name -> path.

    Raises FileNotFoundError naming the output whose pattern matched nothing,
    so the caller can keep the workdir for diagnosis.
    """
    if not specs:
        return {}

    destination = results_dir() / run_id
    destination.mkdir(parents=True, exist_ok=True)

    collected: dict[str, str] = {}
    for spec in specs:
        matches = sorted(workdir.glob(spec.pattern))
        if not matches:
            raise FileNotFoundError(
                f"declared output {spec.name!r} matched no file for pattern "
                f"{spec.pattern!r} in the engine's working directory"
            )
        source = matches[0]
        target = destination / source.name
        shutil.copy2(source, target)
        collected[spec.name] = str(target)
    return collected
