"""Per-engine argument translation and output parsing.

ADDING A NEW TOOL? Read this first.

Every module directly in this directory is discovered automatically by
:mod:`protein_design_mcp.adapters_discovery` — there is no registry to
edit. The naming rule: a manifest named ``run_<x>`` is served by the module
``<x>.py`` right here (the tool name with the leading ``run_`` stripped —
e.g. ``prodigy.py`` serves ``run_prodigy``, ``openmm_minimize.py`` serves
``run_openmm_minimize``). That module must define, at module level:

    def build_args(manifest, params) -> list[str]: ...
    def parse_output(manifest, run) -> dict: ...

each taking exactly those two positional parameters. Add the manifest YAML
and this one module; nothing else needs to change. Get the name wrong (or
forget one of the two functions) and the manifest is excluded at startup
with a reason naming this file and what's missing — see
``adapters_discovery.missing_adapter_reasons`` — rather than the tool
silently having no adapter.

This file itself is intentionally NOT the registry: it must never import
the adapter modules below eagerly, or one broken adapter's ImportError
here would take every other adapter down with it at package-import time.
Discovery imports each candidate module on its own, in isolation.

The discovery machinery (``discover_adapters`` and friends) deliberately
does NOT live in this directory, even though it is about this directory —
see the module docstring of ``protein_design_mcp.adapters_discovery`` for
why: a ``.py`` file living where it scans would scan itself.
"""
