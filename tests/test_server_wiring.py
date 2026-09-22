import json
from pathlib import Path

import pytest

import protein_design_mcp.app as app_module
from protein_design_mcp.app import ServerApp, manifest_dir
from protein_design_mcp.dispatch.env import CompletedRun, EnvDispatcher
from protein_design_mcp.manifest.registry import ToolRegistry
from protein_design_mcp.manifest.schema import parse_manifest

MANIFEST_DIR = manifest_dir()


def _text(result):
    """Extract the first content block's text, whether ``call_tool``
    returned a plain ``list[TextContent]`` (success) or a ``CallToolResult``
    with ``isError=True`` (FIX 3: error paths now carry isError so a client
    can tell a refusal from a result)."""
    if isinstance(result, list):
        return result[0].text
    return result.content[0].text


def _manifest_sharing_a_repo(name):
    """Two manifests that would collide if ADAPTERS were keyed on
    engine.repo instead of manifest.name (FIX 2)."""
    return parse_manifest(
        {
            "name": name,
            "category": "generation",
            "engine": {"repo": "sharedrepo", "env": "e", "entry": ["x"]},
            "summary": "Summary.",
            "doc": "## What this is\nDoc.\n",
            "schema": {},
        }
    )


class _FakeDispatcher:
    """Records the args it was called with instead of spawning a subprocess."""

    def __init__(self):
        self.calls = []

    async def run(self, engine, args, *, timeout, outputs=(), workdir=None):
        self.calls.append((engine, list(args)))
        return CompletedRun(returncode=0, stdout="ok", stderr="", workdir=Path("/tmp"))


def _composite():
    return parse_manifest(
        {
            "name": "run_boltzgen_run",
            "category": "generation",
            "composite": True,
            "engine": {"repo": "boltzgen", "env": "boltzgen", "entry": ["boltzgen", "run"]},
            "summary": "Full pipeline.",
            "doc": "## What this is\nFull pipeline.\n",
            "schema": {"spec": {"type": "string", "required": True, "example": "s.yaml"}},
        }
    )


@pytest.mark.asyncio
async def test_list_tools_includes_describe_tool():
    app = ServerApp(ToolRegistry([]))
    assert "describe_tool" in {t.name for t in await app.list_tools()}


@pytest.mark.asyncio
async def test_composite_tool_is_absent_from_the_listing():
    app = ServerApp(ToolRegistry([_composite()]))
    assert "run_boltzgen_run" not in {t.name for t in await app.list_tools()}


@pytest.mark.asyncio
async def test_calling_a_composite_tool_by_name_is_refused_with_a_reason():
    app = ServerApp(ToolRegistry([_composite()]))
    result = await app.call_tool("run_boltzgen_run", {})
    assert result.isError is True
    payload = json.loads(_text(result))
    assert "composite" in payload["error"]


@pytest.mark.asyncio
async def test_calling_an_unknown_tool_reports_it():
    app = ServerApp(ToolRegistry([]))
    result = await app.call_tool("run_nope", {})
    assert result.isError is True
    payload = json.loads(_text(result))
    assert "unknown" in payload["error"]


@pytest.mark.asyncio
async def test_describe_tool_is_dispatched_without_a_subprocess():
    registry = ToolRegistry([])
    app = ServerApp(registry)
    payload = json.loads(
        (await app.call_tool("describe_tool", {"name": "describe_tool"}))[0].text
    )
    assert payload["name"] == "describe_tool"


@pytest.mark.asyncio
async def test_invalid_input_returns_a_correctable_error_not_a_crash():
    registry = ToolRegistry(
        [m for m in _load_real() if m.name == "run_prodigy"]
    )
    app = ServerApp(registry)
    result = await app.call_tool("run_prodigy", {"complex_pdb": "notes.txt",
                                                  "chain_a": "A", "chain_b": "B"})
    assert result.isError is True
    payload = json.loads(_text(result))
    assert "complex_pdb" in payload["error"]
    assert "complex.pdb" in payload["error"]


def _load_real():
    from protein_design_mcp.manifest.loader import load_manifests

    return load_manifests(MANIFEST_DIR)


def test_real_manifests_all_load():
    assert {m.name for m in _load_real()} >= {"run_prodigy"}


@pytest.mark.asyncio
async def test_a_relative_path_the_caller_supplied_is_resolved_before_dispatch():
    """Regression for FIX 4: the engine subprocess's cwd is a freshly
    created, empty scratch directory, so a relative path like run_prodigy's
    own ``example: complex.pdb`` must be made absolute (relative to the
    SERVER's cwd) before it reaches argv, or it silently resolves against
    that empty scratch directory instead."""
    registry = ToolRegistry([m for m in _load_real() if m.name == "run_prodigy"])
    dispatcher = _FakeDispatcher()
    app = ServerApp(registry, dispatcher=dispatcher)

    await app.call_tool(
        "run_prodigy",
        {"complex_pdb": "complex.pdb", "chain_a": "A", "chain_b": "B"},
    )

    argv = dispatcher.calls[0][1]
    complex_arg = argv[0]
    assert complex_arg != "complex.pdb"
    assert Path(complex_arg).is_absolute()
    assert complex_arg.endswith("complex.pdb")


@pytest.mark.asyncio
async def test_an_already_absolute_path_is_left_alone():
    registry = ToolRegistry([m for m in _load_real() if m.name == "run_prodigy"])
    dispatcher = _FakeDispatcher()
    app = ServerApp(registry, dispatcher=dispatcher)

    await app.call_tool(
        "run_prodigy",
        {"complex_pdb": "/tmp/complex.pdb", "chain_a": "A", "chain_b": "B"},
    )

    assert dispatcher.calls[0][1][0] == "/tmp/complex.pdb"


@pytest.mark.asyncio
async def test_a_non_path_parameter_is_never_touched():
    """chain_a/chain_b are plain strings, not paths — resolving them would
    silently corrupt a bare chain identifier like "A" into an absolute,
    nonexistent path."""
    registry = ToolRegistry([m for m in _load_real() if m.name == "run_prodigy"])
    dispatcher = _FakeDispatcher()
    app = ServerApp(registry, dispatcher=dispatcher)

    await app.call_tool(
        "run_prodigy",
        {"complex_pdb": "/tmp/complex.pdb", "chain_a": "A", "chain_b": "B"},
    )

    argv = dispatcher.calls[0][1]
    assert "A" in argv and "B" in argv


@pytest.mark.asyncio
async def test_describe_tool_is_routed_through_validate_and_fill():
    """Regression for FIX 3: describe_tool used to be dispatched before any
    validation. With SDK schema validation disabled, this manifest's own
    validate_and_fill call is the only input-checking describe_tool gets."""
    app = ServerApp(ToolRegistry([]))
    result = await app.call_tool("describe_tool", {"bogus": "x"})
    assert result.isError is True
    payload = json.loads(_text(result))
    assert "bogus" in payload["error"] or "unexpected" in payload["error"]


@pytest.mark.asyncio
async def test_adapters_are_keyed_by_tool_name_not_engine_repo(monkeypatch):
    """Regression for FIX 2: two tools sharing one engine.repo must each get
    the argv their OWN adapter builds, not whichever adapter happened to be
    registered first for that repo."""
    design = _manifest_sharing_a_repo("run_sharedrepo_design")
    filter_ = _manifest_sharing_a_repo("run_sharedrepo_filter")

    def design_build_args(manifest, params):
        return ["design-argv"]

    def filter_build_args(manifest, params):
        return ["filter-argv"]

    monkeypatch.setattr(
        app_module,
        "ADAPTERS",
        {
            "run_sharedrepo_design": (design_build_args, lambda m, r: {"ok": True}),
            "run_sharedrepo_filter": (filter_build_args, lambda m, r: {"ok": True}),
        },
    )

    dispatcher = _FakeDispatcher()
    app = ServerApp(ToolRegistry([design, filter_]), dispatcher=dispatcher)

    await app.call_tool("run_sharedrepo_design", {})
    await app.call_tool("run_sharedrepo_filter", {})

    assert dispatcher.calls[0][1] == ["design-argv"]
    assert dispatcher.calls[1][1] == ["filter-argv"]


@pytest.mark.asyncio
async def test_adapter_keyerror_produces_a_clear_error_not_a_crash(monkeypatch):
    """Regression for FIX 2: an adapter KeyError (the likely mistake when
    validate_and_fill omits an optional parameter with no default) must not
    escape call_tool as an opaque, unhandled exception."""
    manifest = _manifest_sharing_a_repo("run_sharedrepo_broken")

    def broken_build_args(manifest, params):
        raise KeyError("some_optional_param")

    monkeypatch.setattr(
        app_module,
        "ADAPTERS",
        {"run_sharedrepo_broken": (broken_build_args, lambda m, r: {})},
    )

    app = ServerApp(ToolRegistry([manifest]), dispatcher=_FakeDispatcher())
    result = await app.call_tool("run_sharedrepo_broken", {})
    assert result.isError is True
    payload = json.loads(_text(result))
    assert "error" in payload
    assert "run_sharedrepo_broken" in payload["error"]
    assert "adapter" in payload["error"]


class _RecordingDispatcher:
    """Captures what the app asked the dispatcher to do."""

    def __init__(self):
        self.timeout = None
        self.outputs = None

    async def run(self, engine, args, *, timeout, outputs=(), workdir=None):
        self.timeout = timeout
        self.outputs = tuple(outputs)
        return CompletedRun(
            returncode=0, stdout="", stderr="", workdir=Path("/tmp"), outputs={}
        )


def _timeout_manifest():
    return parse_manifest(
        {
            "name": "run_prodigy",
            "category": "scoring",
            "engine": {"repo": "prodigy", "env": "scoring", "entry": ["prodigy"]},
            "summary": "Timeout probe.",
            "doc": "## What this is\nProbe.\n",
            "timeout_s": 45,
            "outputs": [{"name": "o", "pattern": "o.txt"}],
            "schema": {
                "complex_pdb": {"type": "string", "required": True,
                                "example": "c.pdb", "format": "path"},
                "chain_a": {"type": "string", "required": True, "example": "A"},
                "chain_b": {"type": "string", "required": True, "example": "B"},
                "temperature": {"type": "number", "default": 25.0},
            },
        }
    )


@pytest.mark.asyncio
async def test_manifest_timeout_reaches_the_dispatcher(tmp_path):
    dispatcher = _RecordingDispatcher()
    app = ServerApp(ToolRegistry([_timeout_manifest()]), dispatcher)
    pdb = tmp_path / "c.pdb"
    pdb.write_text("ATOM\n")

    await app.call_tool(
        "run_prodigy",
        {"complex_pdb": str(pdb), "chain_a": "A", "chain_b": "B"},
    )
    assert dispatcher.timeout == 45


@pytest.mark.asyncio
async def test_manifest_outputs_reach_the_dispatcher(tmp_path):
    dispatcher = _RecordingDispatcher()
    app = ServerApp(ToolRegistry([_timeout_manifest()]), dispatcher)
    pdb = tmp_path / "c.pdb"
    pdb.write_text("ATOM\n")

    await app.call_tool(
        "run_prodigy",
        {"complex_pdb": str(pdb), "chain_a": "A", "chain_b": "B"},
    )
    assert [o.name for o in dispatcher.outputs] == ["o"]


@pytest.mark.asyncio
async def test_adapter_returning_outputs_key_fails_loudly(monkeypatch):
    """An adapter that returns 'outputs' in its payload causes a collision.
    Must fail loudly with an error naming the tool and the reserved key,
    not silently overwrite the dispatcher's outputs."""
    manifest = _manifest_sharing_a_repo("run_broken_adapter")

    def broken_build_args(manifest, params):
        return ["arg"]

    def broken_parse_output(manifest, run):
        # This adapter incorrectly returns an 'outputs' key
        return {"result": "value", "outputs": "adapter_outputs"}

    monkeypatch.setattr(
        app_module,
        "ADAPTERS",
        {"run_broken_adapter": (broken_build_args, broken_parse_output)},
    )

    app = ServerApp(ToolRegistry([manifest]), dispatcher=_FakeDispatcher())
    result = await app.call_tool("run_broken_adapter", {})
    assert result.isError is True
    payload = json.loads(_text(result))
    assert "error" in payload
    assert "run_broken_adapter" in payload["error"]
    assert "outputs" in payload["error"]
    assert "reserved" in payload["error"]


@pytest.mark.asyncio
async def test_adapter_output_merges_with_dispatcher_outputs(tmp_path, monkeypatch):
    """An adapter that returns ordinary keys alongside a manifest with
    declared outputs should merge both successfully into the final result."""
    manifest = _timeout_manifest()

    def merging_build_args(manifest, params):
        return ["arg"]

    def merging_parse_output(manifest, run):
        # This adapter returns normal result keys (not 'outputs')
        return {"affinity": "1.5", "kd": "1e-8"}

    monkeypatch.setattr(
        app_module,
        "ADAPTERS",
        {"run_prodigy": (merging_build_args, merging_parse_output)},
    )

    class _OutputDispatcher(_RecordingDispatcher):
        async def run(self, engine, args, *, timeout, outputs=(), workdir=None):
            await super().run(engine, args, timeout=timeout, outputs=outputs)
            # Return some outputs from the dispatcher
            from protein_design_mcp.dispatch.env import CompletedRun
            return CompletedRun(
                returncode=0, stdout="", stderr="", workdir=Path("/tmp"),
                outputs={"o": "output_value.txt"}
            )

    dispatcher = _OutputDispatcher()
    app = ServerApp(ToolRegistry([manifest]), dispatcher=dispatcher)
    pdb = tmp_path / "c.pdb"
    pdb.write_text("ATOM\n")

    result = await app.call_tool(
        "run_prodigy",
        {"complex_pdb": str(pdb), "chain_a": "A", "chain_b": "B"},
    )
    payload = json.loads(_text(result))
    # Both adapter output and dispatcher outputs should be present
    assert payload["affinity"] == "1.5"
    assert payload["kd"] == "1e-8"
    assert payload["outputs"] == {"o": "output_value.txt"}


class _StagingRecordingDispatcher:
    """A fake dispatcher supporting the staging protocol (new_workdir() +
    run(..., workdir=...)), for testing ServerApp.call_tool's staging
    wiring without a real subprocess or a real engine."""

    def __init__(self, tmp_path):
        self._tmp_path = tmp_path
        self.workdir = None
        self.run_args = None

    def new_workdir(self):
        self.workdir = self._tmp_path / "work"
        self.workdir.mkdir()
        return self.workdir

    async def run(self, engine, args, *, timeout, outputs=(), workdir=None):
        assert workdir is self.workdir, "run() must receive the SAME workdir new_workdir() made"
        self.run_args = list(args)
        return CompletedRun(
            returncode=0, stdout="", stderr="", workdir=workdir, outputs={}
        )


def _staging_manifest():
    return parse_manifest(
        {
            "name": "run_prodigy",
            "category": "scoring",
            "engine": {
                "repo": "prodigy",
                "env": "scoring",
                "entry": ["prodigy"],
                "stage": ["structure"],
            },
            "summary": "Staging probe.",
            "doc": "## What this is\nProbe.\n",
            "schema": {
                "structure": {
                    "type": "string",
                    "format": "path",
                    "required": True,
                    "example": "s.pdb",
                },
            },
        }
    )


@pytest.mark.asyncio
async def test_engine_stage_copies_the_input_and_rewrites_the_argument(tmp_path, monkeypatch):
    """A manifest declaring engine.stage must: (1) create a workdir BEFORE
    build_args runs, (2) copy the named input into it, and (3) call
    build_args with the STAGED path, not the caller's original one."""
    source_dir = tmp_path / "caller_supplied"
    source_dir.mkdir()
    source = source_dir / "model.pdb"
    source.write_text("ATOM original content\n")

    seen_params = {}

    def recording_build_args(manifest, params):
        seen_params.update(params)
        return [params["structure"]]

    monkeypatch.setattr(
        app_module,
        "ADAPTERS",
        {"run_prodigy": (recording_build_args, lambda m, r: {"ok": True})},
    )

    dispatcher = _StagingRecordingDispatcher(tmp_path)
    app = ServerApp(ToolRegistry([_staging_manifest()]), dispatcher=dispatcher)

    result = await app.call_tool("run_prodigy", {"structure": str(source)})

    assert not (isinstance(result, object) and getattr(result, "isError", False))
    staged_path = Path(seen_params["structure"])
    assert staged_path != source, "build_args must see the STAGED copy, not the original"
    assert staged_path == dispatcher.workdir / "structure" / "model.pdb"
    assert staged_path.read_text() == "ATOM original content\n"
    assert dispatcher.run_args == [str(staged_path)]
    # The original, caller-supplied file must be untouched (a copy, not a move).
    assert source.read_text() == "ATOM original content\n"


@pytest.mark.asyncio
async def test_a_manifest_without_engine_stage_never_touches_new_workdir(tmp_path, monkeypatch):
    """The common case (no staging) must not call new_workdir() at all —
    manifests that don't opt in must be byte-for-byte unaffected."""
    manifest = _timeout_manifest()

    def build_args(manifest, params):
        return ["arg"]

    monkeypatch.setattr(
        app_module, "ADAPTERS", {"run_prodigy": (build_args, lambda m, r: {"ok": True})}
    )

    class _AssertNoStagingDispatcher(_RecordingDispatcher):
        def new_workdir(self):
            raise AssertionError("new_workdir() must not be called for a non-staging manifest")

    dispatcher = _AssertNoStagingDispatcher()
    app = ServerApp(ToolRegistry([manifest]), dispatcher=dispatcher)
    pdb = tmp_path / "c.pdb"
    pdb.write_text("ATOM\n")

    result = await app.call_tool(
        "run_prodigy", {"complex_pdb": str(pdb), "chain_a": "A", "chain_b": "B"}
    )
    payload = json.loads(_text(result))
    assert payload["ok"] is True


@pytest.mark.asyncio
async def test_a_staging_failure_preserves_its_workdir_like_run_does(tmp_path, monkeypatch):
    """FIX 1 (coordinator review round 2): a REAL EnvDispatcher, not a fake
    one — a fake dispatcher's new_workdir()/run() are just recorded calls,
    so they can't prove anything about what happens to the directory ON
    DISK when staging fails between them. A manifest declaring `stage`
    plus a caller-supplied path that does not exist must produce the same
    diagnosable-error contract run()'s own failure branches already give:
    the workdir is NOT removed, and the error names it as "preserved for
    diagnosis", in that exact wording, rather than silently orphaning a
    scratch directory nobody was told about."""

    def build_args(manifest, params):
        raise AssertionError("build_args must never run — staging failed first")

    monkeypatch.setattr(
        app_module, "ADAPTERS", {"run_prodigy": (build_args, lambda m, r: {})}
    )

    dispatcher = EnvDispatcher(runner=None, scratch_root=tmp_path)
    app = ServerApp(ToolRegistry([_staging_manifest()]), dispatcher=dispatcher)

    missing = tmp_path / "does_not_exist.pdb"
    result = await app.call_tool("run_prodigy", {"structure": str(missing)})

    assert result.isError is True
    payload = json.loads(_text(result))
    assert "preserved for diagnosis" in payload["error"]

    workdirs = list(tmp_path.glob("pdmcp-*"))
    assert len(workdirs) == 1, "the workdir new_workdir() created must survive the failure"
    assert workdirs[0].is_dir()
    assert str(workdirs[0]) in payload["error"]
