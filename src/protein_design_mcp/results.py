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


class OutputPathEscapeError(OSError):
    """A glob match resolves outside the workdir once symlinks are followed.

    ``Path.relative_to`` raises a bare ``ValueError`` when its argument
    isn't actually a prefix of the path being compared. That can happen
    either because the scratch root itself is reached through a symlink (so
    the workdir string and a glob match's string can resolve to different
    real prefixes even though the match came from that workdir) or because
    a matched file is itself a symlink pointing outside the workdir. Either
    way, an unhandled ValueError would reach dispatch/env.py as a bare
    traceback instead of the diagnosable EngineError every other collection
    failure produces. Wrapping it in an OSError subclass fixes that; and a
    match that genuinely resolves outside the workdir is refused here
    rather than silently copied from wherever it actually points.
    """


def _relative_to_workdir(source: Path, workdir: Path, spec_name: str) -> Path:
    """Return ``source``'s path relative to ``workdir``, resolved first.

    Resolving both sides before comparing means a symlinked scratch root
    cannot make this raise merely because pathlib built the two path
    strings through different-looking (but equivalent) prefixes. A match
    that genuinely resolves outside the workdir still raises — as
    OutputPathEscapeError, not a bare ValueError — rather than being
    silently copied from wherever it points.
    """
    try:
        return source.resolve().relative_to(workdir.resolve())
    except ValueError as exc:
        raise OutputPathEscapeError(
            f"declared output {spec_name!r} matched {source}, which "
            f"resolves outside the working directory {workdir} (likely a "
            "symlink); refusing to copy a file from outside the workdir"
        ) from exc


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

    A match that resolves outside the workdir (e.g. via a symlink) raises
    OutputPathEscapeError rather than being copied from wherever it points.
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
            names = [str(_relative_to_workdir(m, workdir, spec.name)) for m in matches]
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
                rel = _relative_to_workdir(source, workdir, spec.name)
                target = spec_dest / rel
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
