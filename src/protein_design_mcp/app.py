"""Transport-independent MCP application logic.

Keeping this out of ``server.py`` means the listing and dispatch behaviour can
be tested without standing up a transport.
"""

from __future__ import annotations

import importlib.resources
import json
import logging
import os
from pathlib import Path
from typing import Any

from mcp.types import CallToolResult, TextContent, Tool

from protein_design_mcp.adapters import ipsae, mpnn, openmm_minimize, prodigy
from protein_design_mcp.dispatch.env import EngineError, EnvDispatcher
from protein_design_mcp.dispatch.serialize import to_jsonable
from protein_design_mcp.manifest.loader import ManifestLoadResult, load_manifests_resilient
from protein_design_mcp.manifest.registry import ToolNotAvailable, ToolRegistry, json_schema_for
from protein_design_mcp.manifest.schema import Manifest, ManifestError
from protein_design_mcp.meta_tools import DESCRIBE_TOOL_MANIFEST, describe_tool
from protein_design_mcp.staging import stage_inputs
from protein_design_mcp.validation import ToolInputError, validate_and_fill

logger = logging.getLogger(__name__)

# Tool name -> (build_args, parse_output). Keyed on manifest.name, NOT
# manifest.engine.repo: several tools can share one engine repo (e.g. a
# future run_boltzgen_design / run_boltzgen_inverse_fold / run_boltzgen_filter
# all with engine.repo == "boltzgen"), and keying on repo would make them all
# resolve to the same adapter functions, building argv for the wrong tool.
# Both functions receive the manifest so one adapter module can still serve
# several tools sharing a repo by branching on manifest.name.
ADAPTERS = {
    "run_prodigy": (prodigy.build_args, prodigy.parse_output),
    "run_ipsae": (ipsae.build_args, ipsae.parse_output),
    "run_openmm_minimize": (openmm_minimize.build_args, openmm_minimize.parse_output),
    "run_mpnn": (mpnn.build_args, mpnn.parse_output),
}


def manifest_dir() -> Path:
    """Directory holding ``*.yaml`` tool manifests.

    Ships as package data at ``protein_design_mcp/manifests/`` so it is
    present in both the wheel and the Docker image (whatever copies
    ``src/`` picks it up automatically). Resolved via ``importlib.resources``
    so it also works from a zipped/namespace install, not just a plain
    directory checkout. ``PROTEIN_MCP_MANIFEST_DIR`` always overrides it.
    """
    override = os.environ.get("PROTEIN_MCP_MANIFEST_DIR")
    if override:
        return Path(override)
    return Path(importlib.resources.files("protein_design_mcp") / "manifests")


def build_registry(device: str = "cuda") -> ToolRegistry:
    """Load manifests from disk into a registry.

    A manifest problem must degrade the server, not prevent import: if the
    directory is missing, log a loud diagnostic naming the directory and
    serve an empty registry (plus ``describe_tool``, which is not
    manifest-backed) instead of raising. Individual manifest failures
    (unreadable file, invalid YAML, schema violation, duplicate tool name,
    bad cross-reference) never reach this except block at all —
    ``load_manifests_resilient`` already excluded each of them on its own,
    with a reason, so the OTHER manifests always keep loading. Only a
    missing/unreadable manifest DIRECTORY (or STRICT_MANIFESTS=1 in CI)
    still raises.
    """
    directory = manifest_dir()
    try:
        result = load_manifests_resilient(directory)
    except ManifestError as exc:
        logger.error(
            "Could not load tool manifests from %s: %s. Serving an EMPTY "
            "tool registry (describe_tool is still available). Set "
            "PROTEIN_MCP_MANIFEST_DIR to point at a valid manifest "
            "directory and restart.",
            directory,
            exc,
        )
        result = ManifestLoadResult([], {})
    registry = ToolRegistry(result.manifests, device=device, load_failures=result.reasons)
    _log_exclusions(registry, result.manifests, result.reasons)
    return registry


def _log_exclusions(
    registry: ToolRegistry,
    manifests: list[Manifest],
    load_failures: dict[str, str] | None = None,
) -> None:
    """Log the full exclusion table at startup.

    Two independent sources feed this table. First, ``ToolRegistry`` itself
    excludes any manifest declaring ``requires.weights`` or
    ``requires.license_gated`` (``build_registry`` never passes
    ``available_weights``/``licensed``, so both default to empty), as well
    as GPU-only and composite tools — that's ``registry.excluded()`` below.
    Second, ``load_failures`` covers manifests that never made it into
    ``manifests`` at all: a duplicate tool name, a bad cross-reference, or a
    file that failed to parse (see ``load_manifests_resilient``). Without
    logging both, a tool vanishing from the listing looked mysterious
    rather than visible.
    """
    exclusions = {m.name: registry.excluded(m.name) for m in manifests}
    exclusions = {name: reason for name, reason in exclusions.items() if reason}
    exclusions.update(load_failures or {})
    if not exclusions:
        return
    table = "\n".join(
        f"  - {name}: {reason}" for name, reason in sorted(exclusions.items())
    )
    logger.warning(
        "%d of %d tool manifest(s) excluded from the registry at startup:\n%s",
        len(exclusions),
        len(manifests) + len(load_failures or {}),
        table,
    )


def _resolve_path_params(manifest: Manifest, params: dict[str, Any]) -> dict[str, Any]:
    """Resolve caller-supplied path parameters to absolute paths.

    Every engine subprocess runs with ``cwd`` set to a freshly created,
    EMPTY scratch directory (see ``dispatch.env.EnvDispatcher``). A relative
    path the caller supplied — including the exact relative path a
    manifest's own ``example:`` shows, e.g. ``run_prodigy.yaml``'s
    ``complex.pdb`` — would otherwise resolve against that empty directory
    and fail with a bare "No such file or directory", with no hint that the
    path was resolved somewhere else than the caller intended.

    Done centrally, once, on every schema entry marked ``format: path``,
    rather than in each adapter's ``build_args``: ~28 more adapters are
    planned on this seam and would otherwise each have to rediscover this.
    Resolution happens here, in the main server process, before the
    dispatcher ever changes into the scratch directory, so it is always
    relative to the server's own working directory (or already absolute).
    """
    resolved = dict(params)
    for key, spec in manifest.schema.items():
        if spec.get("format") == "path" and isinstance(resolved.get(key), str):
            resolved[key] = str(Path(resolved[key]).resolve())
    return resolved


def _error(message: str) -> CallToolResult:
    """Build an error result.

    Returns a ``CallToolResult`` with ``isError=True`` rather than a bare
    list of content, because with server-side JSON-schema validation
    disabled (see server.py) this is now the ONLY signal a client has to
    distinguish a refusal from a normal result: the SDK passes a
    ``CallToolResult`` through unchanged (see ``mcp.server.Server.call_tool``),
    but a plain ``list[TextContent]`` is always wrapped as ``isError=False``.
    """
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps({"error": message}, indent=2))],
        isError=True,
    )


def _ok(payload: Any) -> list[TextContent]:
    return [
        TextContent(type="text", text=json.dumps(to_jsonable(payload), indent=2))
    ]


class ServerApp:
    """Owns the registry and turns MCP calls into engine runs."""

    def __init__(
        self,
        registry: ToolRegistry,
        dispatcher: EnvDispatcher | None = None,
    ) -> None:
        self._registry = registry
        self._dispatcher = dispatcher or EnvDispatcher()

    async def list_tools(self) -> list[Tool]:
        tools = self._registry.tools()
        tools.append(
            Tool(
                name=DESCRIBE_TOOL_MANIFEST.name,
                description=DESCRIBE_TOOL_MANIFEST.summary,
                inputSchema=json_schema_for(DESCRIBE_TOOL_MANIFEST),
            )
        )
        return tools

    async def call_tool(
        self, name: str, arguments: dict[str, Any] | None
    ) -> list[TextContent] | CallToolResult:
        arguments = arguments or {}
        logger.info("tool call: %s %s", name, arguments)

        if name == DESCRIBE_TOOL_MANIFEST.name:
            # Routed through validate_and_fill like any other tool: with the
            # SDK's own schema validation disabled (server.py sets
            # validate_input=False so OUR messages reach the client), this is
            # the only input validation describe_tool gets.
            try:
                params = validate_and_fill(DESCRIBE_TOOL_MANIFEST, arguments)
            except ToolInputError as exc:
                return _error(str(exc))
            return _ok(
                describe_tool(
                    self._registry,
                    name=params.get("name"),
                    category=params.get("category"),
                )
            )

        try:
            manifest = self._registry.resolve(name)
        except ToolNotAvailable as exc:
            return _error(str(exc))

        try:
            params = validate_and_fill(manifest, arguments)
        except ToolInputError as exc:
            return _error(str(exc))
        params = _resolve_path_params(manifest, params)

        adapter = ADAPTERS.get(manifest.name)
        if adapter is None:
            return _error(
                f"{name} has no adapter registered (engine "
                f"{manifest.engine.repo!r})"
            )

        build_args, parse_output = adapter
        try:
            # Most engines write wherever their subprocess's cwd is, which
            # the dispatcher already sets to a scratch workdir it creates
            # itself. An engine declared in manifest.engine.stage instead
            # writes beside one of its INPUT files — that input was just
            # resolved to an absolute path outside any scratch directory
            # (see _resolve_path_params), so its workdir has to be created
            # early enough to copy that input into it FIRST, rewriting the
            # parameter to the staged copy, before build_args ever sees it.
            workdir = None
            if manifest.engine.stage:
                workdir = self._dispatcher.new_workdir()
                # Staging can fail (most obviously: a caller-supplied path
                # that doesn't exist) after the workdir already exists but
                # before dispatcher.run() — which owns the "preserve on
                # failure, remove on success, and SAY SO" contract — is ever
                # reached. Without this, a staging failure would silently
                # orphan the workdir: not removed (nothing said it should
                # be), and not mentioned either. Preserving it and naming it
                # in the error, in the exact wording run()'s own failure
                # branches already use, keeps that contract unbroken across
                # this earlier span too.
                try:
                    params = stage_inputs(manifest.engine.stage, params, workdir)
                except OSError as exc:
                    raise EngineError(
                        f"could not stage input(s) for {name}: {exc}. "
                        f"Working directory preserved for diagnosis: {workdir}"
                    ) from exc
            run = await self._dispatcher.run(
                manifest.engine,
                build_args(manifest, params),
                timeout=manifest.timeout_s,
                outputs=manifest.outputs,
                workdir=workdir,
            )
            payload = parse_output(manifest, run)
            if "outputs" in payload:
                raise ValueError(
                    f"adapter for {name} returned 'outputs' key, which is "
                    f"reserved by the dispatcher contract; rename this field"
                )
            if run.outputs:
                payload = {**payload, "outputs": run.outputs}
            return _ok(payload)
        except EngineError as exc:
            return _error(str(exc))
        except Exception as exc:
            # Catches ValueError (e.g. "could not parse engine output") and,
            # critically, KeyError: the most likely adapter mistake, since
            # validate_and_fill omits optional parameters that have no
            # default. Either must produce a clear error payload, not an
            # opaque protocol-level failure.
            return _error(f"adapter for {name} failed: {exc}")
