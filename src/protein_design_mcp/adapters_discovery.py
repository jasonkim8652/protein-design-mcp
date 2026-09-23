"""Discover adapter modules by filename convention instead of a hand-maintained registry.

THE NAMING RULE (read this before adding a tool): a manifest named
``run_<x>`` is served by the module ``protein_design_mcp/adapters/<x>.py`` —
the manifest's ``name`` with the leading ``run_`` stripped. That module must
define two module-level functions:

    def build_args(manifest, params) -> list[str]: ...
    def parse_output(manifest, run) -> dict: ...

each taking exactly two positional parameters. Adding a new tool is then:
add its manifest YAML, add this one module — nothing else. No shared file
to edit, so no merge conflict and no "I wrote the adapter but forgot to
register it" bug.

This is the existing convention the four shipped adapters already follow
(``prodigy.py`` serves ``run_prodigy``, ``openmm_minimize.py`` serves
``run_openmm_minimize``); this module makes it load-bearing (enforced by
:func:`discover_adapters`) instead of merely a naming habit nobody checked.

Isolation: each candidate file is imported on its own, wrapped in its own
try/except. One adapter module with a syntax error, a bad import, a missing
symbol, or the wrong arity is recorded in ``broken`` (or omitted, for
non-Python files and ``__init__.py``) and never prevents any *other*
adapter module in the same directory from loading — the same principle
``manifest.loader.load_manifests_resilient`` applies to manifests. Nothing
here imports ``protein_design_mcp.adapters`` as a package (which would run
its ``__init__.py`` and could cascade), only the individual candidate files
found by scanning the directory.

This module does not know about manifests at all — :func:`discover_adapters`
only scans a directory of ``.py`` files. Tying that result to a set of
manifest tool names (to decide which manifests lack an adapter, and which
adapter modules are stranded without a manifest) is :func:`missing_adapter_reasons`
and :func:`stranded_adapter_reasons`, used by ``protein_design_mcp.app.build_registry``.

THIS FILE LIVES OUTSIDE ``protein_design_mcp/adapters/`` ON PURPOSE. If the
discovery machinery lived inside the directory it scans, it would be a
``.py`` file in that directory too, and would scan *itself* as a broken
candidate adapter (it defines neither ``build_args`` nor ``parse_output``)
— reported as a false "stranded adapter" at every single startup. Mirrors
how ``manifest/loader.py`` lives in the ``manifest`` package, never inside
the ``manifests/`` directory of ``*.yaml`` data files it loads.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Callable, NamedTuple

_TOOL_PREFIX = "run_"
_REQUIRED_SYMBOLS = ("build_args", "parse_output")
_REQUIRED_ARITY = 2
_PACKAGE_QUALNAME = "protein_design_mcp.adapters"


class AdapterFunctions(NamedTuple):
    """The two functions a valid adapter module exposes. Also unpacks as a
    plain 2-tuple (``build_args, parse_output = adapters[name]``), the same
    shape the hand-written ``ADAPTERS`` dict used to hold, so nothing that
    reads from the result needs to know this is a NamedTuple rather than a
    bare tuple."""

    build_args: Callable[..., Any]
    parse_output: Callable[..., Any]


class AdapterDiscoveryResult(NamedTuple):
    """What :func:`discover_adapters` found in one pass over a directory.

    ``adapters``: tool name -> (build_args, parse_output), for every module
    that loaded cleanly and exposes both required symbols with the right
    arity.

    ``broken``: tool name -> a message naming the module and what is wrong
    with it (import failure, missing symbol, wrong arity), for every ``.py``
    file that was found but did NOT qualify. A module here is never also in
    ``adapters``.

    ``module_names``: every candidate module's bare name (its filename minus
    ``.py``, so ``prodigy.py`` -> ``"prodigy"``), valid or broken, found in
    the directory. Used to detect a *stranded* adapter — one with no
    manifest of the corresponding name at all — which is a different
    failure mode than either of the above and is reported separately (see
    :func:`stranded_adapter_reasons`).
    """

    adapters: dict[str, AdapterFunctions]
    broken: dict[str, str]
    module_names: frozenset[str]


def module_name_for_tool(tool_name: str) -> str:
    """``run_prodigy`` -> ``prodigy``. Raises for anything not shaped like a
    ``run_*`` tool name (``describe_tool`` and ``get_job_status`` are meta
    tools, never adapter-backed, and must never be passed here)."""
    if not tool_name.startswith(_TOOL_PREFIX):
        raise ValueError(
            f"not an adapter-backed tool name (must start with {_TOOL_PREFIX!r}): "
            f"{tool_name!r}"
        )
    return tool_name[len(_TOOL_PREFIX) :]


def tool_name_for_module(module_name: str) -> str:
    """``prodigy`` -> ``run_prodigy``. Inverse of :func:`module_name_for_tool`."""
    return f"{_TOOL_PREFIX}{module_name}"


def _candidate_files(directory: Path) -> list[Path]:
    """Every plain ``.py`` file directly inside ``directory``, sorted, other
    than ``__init__.py``. Non-recursive: ``directory.glob("*.py")`` only
    matches direct children, so it never descends into ``__pycache__`` (which
    holds ``.pyc`` files anyway, not ``.py``) or any other subdirectory. A
    non-``.py`` file (a stray ``.txt``, a ``README``, …) never matches the
    glob at all, so it is skipped without special-casing."""
    return sorted(
        p for p in directory.glob("*.py") if p.is_file() and p.name != "__init__.py"
    )


def _load_module(path: Path, module_name: str) -> Any:
    """Import ``path`` as ``protein_design_mcp.adapters.<module_name>``, by
    file location rather than ``import_module`` — this works whether
    ``directory`` is the real installed adapters package or a scratch
    directory a test points at, and it isolates the exec of each file so a
    broken one can be caught here without disturbing the others."""
    qualified = f"{_PACKAGE_QUALNAME}.{module_name}"
    spec = importlib.util.spec_from_file_location(qualified, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not build an import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        # Don't leave a half-initialized module registered under this name —
        # a later, unrelated import of the same qualified name (unlikely,
        # but possible if a test retries discovery) must not pick up a
        # module that crashed partway through exec.
        sys.modules.pop(qualified, None)
        raise
    return module


def _check_symbols(module: Any, module_name: str) -> AdapterFunctions | str:
    """Validate ``module`` exposes both required symbols, callable, each
    accepting exactly two positional arguments. Returns the pair on success,
    or an error string naming the module and the specific problem."""
    location = f"protein_design_mcp/adapters/{module_name}.py"

    missing = [s for s in _REQUIRED_SYMBOLS if not hasattr(module, s)]
    if missing:
        return (
            f"adapter module '{location}' is missing required symbol(s): "
            f"{', '.join(missing)} (an adapter must define both build_args "
            "and parse_output)"
        )

    functions: dict[str, Callable[..., Any]] = {}
    for symbol in _REQUIRED_SYMBOLS:
        fn = getattr(module, symbol)
        if not callable(fn):
            return f"adapter module '{location}': '{symbol}' is defined but is not callable"
        try:
            signature = inspect.signature(fn)
            signature.bind(*([object()] * _REQUIRED_ARITY))
        except TypeError as exc:
            other = "params" if symbol == "build_args" else "run"
            return (
                f"adapter module '{location}': '{symbol}' does not accept "
                f"exactly {_REQUIRED_ARITY} positional arguments "
                f"(manifest, {other}): {exc}"
            )
        functions[symbol] = fn

    return AdapterFunctions(build_args=functions["build_args"], parse_output=functions["parse_output"])


def discover_adapters(directory: Path) -> AdapterDiscoveryResult:
    """Scan ``directory`` for adapter modules, one file at a time.

    A missing directory is not an error here (mirrors
    ``manifest.loader.load_manifests_resilient``'s own tolerance) — it just
    yields an empty result; the caller (``app.build_registry``) is what
    decides whether an empty adapters directory is itself worth logging.
    """
    directory = Path(directory)

    adapters: dict[str, AdapterFunctions] = {}
    broken: dict[str, str] = {}
    module_names: set[str] = set()

    if not directory.is_dir():
        return AdapterDiscoveryResult({}, {}, frozenset())

    for path in _candidate_files(directory):
        module_name = path.stem
        module_names.add(module_name)
        tool_name = tool_name_for_module(module_name)

        try:
            module = _load_module(path, module_name)
        except Exception as exc:  # isolation: one broken module, not the rest
            broken[tool_name] = (
                f"adapter module 'protein_design_mcp/adapters/{module_name}.py' "
                f"failed to import: {type(exc).__name__}: {exc}"
            )
            continue

        result = _check_symbols(module, module_name)
        if isinstance(result, str):
            broken[tool_name] = result
        else:
            adapters[tool_name] = result

    return AdapterDiscoveryResult(adapters, broken, frozenset(module_names))


def missing_adapter_reasons(
    tool_names: Iterable[str], discovery: AdapterDiscoveryResult
) -> dict[str, str]:
    """For every ``tool_name`` with no usable adapter, a model-facing reason
    naming the manifest and the module path that was looked for (or, if the
    module exists but is broken, what is wrong with it) — never an
    instruction to create a file, since a model reading this cannot do
    that; it can only be told the tool is unavailable and why.

    Keyed by tool name so it plugs directly into ``ToolRegistry``'s
    ``load_failures``, the same per-tool exclusion mechanism
    ``manifest.loader.load_manifests_resilient`` already uses for a
    duplicate name or a bad cross-reference: one exclusion, one tool,
    everything else keeps loading.
    """
    reasons: dict[str, str] = {}
    for tool_name in tool_names:
        if tool_name in discovery.adapters:
            continue
        module_name = module_name_for_tool(tool_name)
        if tool_name in discovery.broken:
            reasons[tool_name] = f"{tool_name} is unavailable: {discovery.broken[tool_name]}"
        else:
            reasons[tool_name] = (
                f"{tool_name} is unavailable: no adapter module was found at "
                f"protein_design_mcp/adapters/{module_name}.py, so it cannot "
                "be dispatched to an engine."
            )
    return reasons


def stranded_adapter_reasons(
    known_tool_names: Iterable[str], discovery: AdapterDiscoveryResult
) -> dict[str, str]:
    """Adapter modules whose corresponding tool name has no manifest at all
    (``known_tool_names`` should include every manifest that was even
    *attempted* to load, not just the ones that ended up available — a
    manifest excluded for an unrelated reason, e.g. requires GPU, still
    "exists" and its adapter is not stranded).

    Keyed by module name (there is no tool name to key by — that is
    precisely the problem). Not routed through ``ToolRegistry`` — there is
    no manifest to exclude — just reported at startup by the caller.

    Wording distinguishes a module that DOES look like a valid adapter
    (``discovery.adapters``) from one that is also broken
    (``discovery.broken``) — a module can be stranded and broken at once,
    and the message must not claim it "implements build_args/parse_output"
    when it doesn't.
    """
    known = set(known_tool_names)
    reasons: dict[str, str] = {}
    for module_name in sorted(discovery.module_names):
        tool_name = tool_name_for_module(module_name)
        if tool_name in known:
            continue
        location = f"protein_design_mcp/adapters/{module_name}.py"
        if tool_name in discovery.adapters:
            what = f"implements build_args/parse_output for {tool_name!r}"
        else:
            what = f"was meant to implement {tool_name!r} but is itself broken ({discovery.broken.get(tool_name, 'unknown reason')})"
        reasons[module_name] = (
            f"adapter module '{location}' {what}, but no manifest named "
            f"{tool_name!r} exists. Either the manifest was deleted/renamed, "
            "or this module was misnamed."
        )
    return reasons
