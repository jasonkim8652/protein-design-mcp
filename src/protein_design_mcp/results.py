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
from dataclasses import dataclass
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


@dataclass(frozen=True)
class _CheckedMatch:
    """A glob match that has already passed the containment check.

    This type exists to make containment structural rather than a habit.
    ``_check_containment`` below is the only function that constructs one,
    and ``_copy_checked_match`` — the only ``shutil.copy2`` call in this
    module — accepts nothing else. So a copy is reachable only via a value
    that could only have come from the check. Adding a new branch to
    ``collect_outputs`` (a third output mode, say) cannot skip the check
    without also having to invent its own copy call, which is a visible act
    rather than an easy omission.

    ``relative`` is the match's path relative to the *resolved* workdir, and
    is used verbatim as the copy target under the spec's subdirectory. It is
    relative and free of ``..`` by construction (``Path.relative_to``
    guarantees that when it succeeds), so joining it onto a destination
    cannot walk back out of that destination either.
    """

    source: Path
    relative: Path


def _check_containment(source: Path, resolved_workdir: Path, spec_name: str) -> _CheckedMatch:
    """THE containment choke point. Every copied file passes through here.

    Resolving both sides before comparing means a symlinked scratch root
    cannot make this fail merely because pathlib built the two path strings
    through different-looking (but equivalent) prefixes. A match that
    genuinely resolves outside the workdir — most obviously a symlink inside
    the workdir pointing elsewhere — is refused, as OutputPathEscapeError
    (an OSError) rather than the bare ValueError ``relative_to`` raises,
    so dispatch/env.py's ``except OSError`` turns it into a diagnosable
    EngineError.

    Raises:
        OutputPathEscapeError: ``source`` resolves outside ``resolved_workdir``.
    """
    resolved_source = source.resolve()
    try:
        relative = resolved_source.relative_to(resolved_workdir)
    except ValueError as exc:
        raise OutputPathEscapeError(
            f"declared output {spec_name!r} matched {source}, which "
            f"resolves to {resolved_source}, outside the working directory "
            f"{resolved_workdir} (likely a symlink); refusing to copy a file "
            "from outside the workdir"
        ) from exc
    return _CheckedMatch(source=source, relative=relative)


def _checked_matches(spec: OutputSpec, workdir: Path, resolved_workdir: Path) -> list[_CheckedMatch]:
    """Resolve ``spec``'s glob into checked matches, or raise.

    Applies the arity rules (zero matches is always an error; more than one
    match is an error unless ``multiple``) and then runs *every* surviving
    match through ``_check_containment``, irrespective of ``multiple``.

    Raises:
        FileNotFoundError: the pattern matched nothing.
        AmbiguousOutputError: ``multiple=False`` and the pattern matched >1.
        OutputPathEscapeError: any match resolves outside the workdir.
    """
    matches = sorted(workdir.glob(spec.pattern))
    if not matches:
        raise FileNotFoundError(
            f"declared output {spec.name!r} matched no file for pattern "
            f"{spec.pattern!r} in the engine's working directory"
        )
    if not spec.multiple and len(matches) > 1:
        names = [
            str(_check_containment(m, resolved_workdir, spec.name).relative) for m in matches
        ]
        raise AmbiguousOutputError(
            f"declared output {spec.name!r} matched {len(matches)} files "
            f"for pattern {spec.pattern!r} in the engine's working "
            f"directory, expected exactly one: {names}. Narrow the "
            "pattern, or set multiple: true on this output if more than "
            "one file is expected."
        )
    return [_check_containment(m, resolved_workdir, spec.name) for m in matches]


def _copy_checked_match(match: _CheckedMatch, spec_dest: Path) -> str:
    """The module's only copy. Takes a _CheckedMatch, so never an unchecked one."""
    target = spec_dest / match.relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(match.source, target)
    return str(target)


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
    one spec, each match is copied to its path *relative to the workdir*
    under that subdirectory (not just its basename), so two matches of the
    same spec that share a basename in different subdirectories don't
    collide with each other either — a path relative to the workdir is
    unique by construction. Both modes use the same rule; ``multiple`` only
    decides whether the result is a list or the single element.

    A match that resolves outside the workdir (e.g. a symlink inside the
    workdir pointing elsewhere) raises OutputPathEscapeError rather than
    being copied from wherever it points. This holds for every match of
    every spec in both modes: matching and containment-checking happens as
    a complete first pass, and only once every spec has passed does any
    copying begin, so a failure part-way through leaves nothing copied.
    """
    if not specs:
        return {}

    # Pass 1 -- match and check EVERYTHING. `multiple` selects only how the
    # results are shaped below, never whether a match is checked, and no
    # copy has happened yet when this pass raises.
    resolved_workdir = workdir.resolve()
    plan: list[tuple[OutputSpec, list[_CheckedMatch]]] = [
        (spec, _checked_matches(spec, workdir, resolved_workdir)) for spec in specs
    ]

    # Pass 2 -- copy. Reachable only with _CheckedMatch values from pass 1.
    destination = results_dir() / run_id
    collected: dict[str, str | list[str]] = {}
    for spec, matches in plan:
        spec_dest = destination / spec.name
        spec_dest.mkdir(parents=True, exist_ok=True)
        copied = [_copy_checked_match(match, spec_dest) for match in matches]
        collected[spec.name] = copied if spec.multiple else copied[0]
    return collected
