"""Copy declared input files into an engine's scratch working directory.

An engine's own outputs are collected from its scratch working directory
(``results.collect_outputs``), relative to that directory. That works for an
engine that writes wherever its process's cwd is (the dispatcher already sets
``cwd`` to the scratch directory). It does NOT work for an engine that
instead writes beside one of its INPUT files — ipSAE is the first example:
it always writes its results next to the structure file it was given,
regardless of cwd. ``app._resolve_path_params`` resolves a caller-supplied
path to an ABSOLUTE path (deliberately — see its own docstring), which for
such an engine means "beside the input" is outside the scratch directory
entirely: unreachable by a relative ``outputs:`` pattern, and refused by
``results.collect_outputs``'s containment check even if the pattern somehow
reached it.

``stage_inputs`` fixes this the same way the dispatcher already solves the
analogous OUTPUT problem, just running in the opposite direction: copy the
named input file(s) INTO the scratch directory first, so "beside the input"
becomes "inside the scratch directory" too, and the existing ``outputs:``
machinery just works. A manifest opts an engine into this via
``engine.stage`` (see ``manifest.schema.EngineSpec.stage``), naming which
``format: path`` schema parameters need it — most engines name none.
"""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def stage_inputs(
    names: Sequence[str],
    params: dict[str, Any],
    workdir: Path,
    subdirs: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Copy each named path parameter's file into its own subdirectory of
    ``workdir``, returning a NEW params dict with those entries rewritten to
    the staged copy's absolute path. ``params`` itself is not mutated.

    Each staged name gets its own ``workdir/<name>/`` subdirectory — the
    same "namespace by spec name" choice ``results.collect_outputs`` already
    makes for outputs, here applied to inputs. This is what makes two staged
    parameters safe even if their SOURCE files happen to share a basename
    (e.g. two different callers' inputs both named ``model.pdb``): they are
    copied to ``workdir/<name_a>/model.pdb`` and ``workdir/<name_b>/model.pdb``
    respectively, which can never collide, rather than both landing directly
    in ``workdir`` where the second copy would silently overwrite the first —
    the exact class of bug ``collect_outputs`` was already hardened against
    on the output side.

    A name absent from ``params`` (or whose value is ``None`` — an optional
    path parameter the caller didn't supply) is skipped rather than staged.
    A name whose source file does not exist raises ``FileNotFoundError`` from
    the underlying ``shutil.copy2`` call, exactly as a missing file already
    fails at the point an engine subprocess would have tried to read it.

    A ``params[name]`` that is a ``list`` (an array-of-``format: path``
    schema parameter — see ``manifest.schema._validate_stage``) stages EVERY
    item into that SAME ``workdir/<name>/`` directory, rather than each
    getting its own subdirectory. This is the opposite tradeoff from two
    DIFFERENT staged names (kept apart specifically so same-basename inputs
    never collide): here, landing together in one directory is the whole
    point — an engine like BoltzGen's ``fold``/``analyze`` reads each
    design's ``.cif`` and its matching ``.npz`` out of the SAME directory,
    matched by shared basename, so the caller's one file list (e.g. a prior
    tool call's own combined ``outputs`` list) has to end up in one place.
    Two list ITEMS that happen to share a basename overwrite each other
    (last write wins, exactly like ``shutil.copy2`` always does) rather than
    being deduplicated or renamed — a same-named collision within one list
    is a caller error (a duplicate file), not something this function can
    safely paper over. An empty list stages nothing and returns an empty
    list, not ``None`` — this is a legitimate value (e.g. "nothing survived
    upstream filtering"), never treated as "parameter absent".

    ``subdirs`` optionally maps a staged name to an explicit relative path
    (which may itself contain more segments) used INSTEAD of the default
    ``<name>`` — see ``manifest.schema.EngineSpec.stage_subdir`` for why:
    several staged names can then share one parent tree, each landing at
    the specific relative location a downstream engine's own convention
    expects (e.g. ``"design_dir"`` for one name and
    ``"design_dir/refold_cif"`` for another, so both end up under the same
    ``workdir/design_dir/``). A name absent from ``subdirs`` (or when
    ``subdirs`` itself is ``None``) keeps the original ``workdir/<name>/``
    placement.
    """
    staged = dict(params)
    for name in names:
        value = params.get(name)
        if value is None:
            continue
        dest_dir = workdir / (subdirs.get(name, name) if subdirs else name)
        dest_dir.mkdir(parents=True, exist_ok=True)
        if isinstance(value, list):
            staged_items = []
            for item in value:
                source = Path(str(item))
                dest = dest_dir / source.name
                shutil.copy2(source, dest)
                staged_items.append(str(dest))
            staged[name] = staged_items
        else:
            source = Path(str(value))
            dest = dest_dir / source.name
            shutil.copy2(source, dest)
            staged[name] = str(dest)
    return staged
