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
) -> dict[str, str | list[str]]:
    """Copy each declared output out of ``workdir``. Returns name -> path(s).

    A spec with ``multiple=False`` (the default) returns a single path string
    and requires its pattern to match exactly one file: zero matches raises
    FileNotFoundError naming the missing output, and more than one match
    raises FileNotFoundError naming every match (an ambiguous match is the
    same failure class as a missing one — resolving it silently would return
    a plausible wrong answer). A spec with ``multiple=True`` returns a list
    of every matched path instead, and still requires at least one match.

    Each spec's matches are copied into their own ``destination/<spec.name>/``
    subdirectory, so two specs whose patterns match files with the same
    basename in different subdirectories of the workdir never collide.
    """
    if not specs:
        return {}

    destination = results_dir() / run_id

    collected: dict[str, str | list[str]] = {}
    for spec in specs:
        matches = sorted(workdir.glob(spec.pattern))
        if not matches:
            raise FileNotFoundError(
                f"declared output {spec.name!r} matched no file for pattern "
                f"{spec.pattern!r} in the engine's working directory"
            )
        if not spec.multiple and len(matches) > 1:
            names = [str(m.relative_to(workdir)) for m in matches]
            raise FileNotFoundError(
                f"declared output {spec.name!r} matched {len(matches)} files "
                f"for pattern {spec.pattern!r} in the engine's working "
                f"directory, expected exactly one: {names}. Narrow the "
                "pattern, or set multiple: true on this output if more than "
                "one file is expected."
            )

        spec_dest = destination / spec.name
        spec_dest.mkdir(parents=True, exist_ok=True)

        if spec.multiple:
            copied: list[str] = []
            for source in matches:
                target = spec_dest / source.name
                shutil.copy2(source, target)
                copied.append(str(target))
            collected[spec.name] = copied
        else:
            source = matches[0]
            target = spec_dest / source.name
            shutil.copy2(source, target)
            collected[spec.name] = str(target)
    return collected
