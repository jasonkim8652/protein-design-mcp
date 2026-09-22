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


class AmbiguousOutputError(OSError):
    """A single-valued output's pattern matched more than one file.

    Distinct from FileNotFoundError (genuinely no match): "too many files
    matched" is a different failure than "file not found", and conflating
    them misleads anyone reading a traceback or narrowly catching
    FileNotFoundError elsewhere. It subclasses OSError so callers that
    handle collection failures broadly (dispatch/env.py's ``except
    OSError``) still catch it without change.
    """


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
    raises AmbiguousOutputError naming every match (an ambiguous match is a
    kin failure to a missing one in that resolving it silently would return
    a plausible wrong answer, but it is not itself a missing file, so it gets
    its own exception type). A spec with ``multiple=True`` returns a list of
    every matched path instead, and still requires at least one match.

    Each spec's matches are copied into their own ``destination/<spec.name>/``
    subdirectory, so two specs whose patterns match files with the same
    basename in different subdirectories of the workdir never collide. Within
    a single ``multiple=True`` spec, each match is copied to its path
    *relative to the workdir* under that subdirectory (not just its
    basename), so two matches of the same spec that share a basename in
    different subdirectories don't collide with each other either — a path
    relative to the workdir is unique by construction.
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
            raise AmbiguousOutputError(
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
                target = spec_dest / source.relative_to(workdir)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                copied.append(str(target))
            collected[spec.name] = copied
        else:
            source = matches[0]
            target = spec_dest / source.name
            shutil.copy2(source, target)
            collected[spec.name] = str(target)
    return collected
