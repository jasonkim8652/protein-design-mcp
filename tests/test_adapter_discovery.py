"""Task 4: adapter discovery over adapters/ replaces the hand-maintained
ADAPTERS dict in app.py.

Covers the discovery primitives (``discover_adapters``,
``missing_adapter_reasons``, ``stranded_adapter_reasons``) in isolation
against scratch directories, then the end-to-end wiring through
``app.build_registry`` / ``ServerApp``.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from protein_design_mcp.adapters_discovery import (
    discover_adapters,
    missing_adapter_reasons,
    module_name_for_tool,
    stranded_adapter_reasons,
    tool_name_for_module,
)
from protein_design_mcp.manifest.registry import ToolNotAvailable, ToolRegistry
from protein_design_mcp.manifest.schema import parse_manifest

VALID_ADAPTER = """\
    def build_args(manifest, params):
        return ["--x"]


    def parse_output(manifest, run):
        return {"ok": True}
    """

MISSING_PARSE_OUTPUT = """\
    def build_args(manifest, params):
        return ["--x"]
    """

WRONG_ARITY = """\
    def build_args(manifest, params):
        return ["--x"]


    def parse_output(manifest):
        return {"ok": True}
    """

RAISES_ON_IMPORT = """\
    raise RuntimeError("boom at import time")
    """


def _write(directory: Path, filename: str, body: str) -> Path:
    path = directory / filename
    path.write_text(textwrap.dedent(body))
    return path


def _manifest(name: str, category: str = "binder_generation"):
    return parse_manifest(
        {
            "name": name,
            "category": category,
            "engine": {"repo": name, "env": "e", "entry": ["x"]},
            "summary": "Summary.",
            "doc": "## What this is\nDoc.\n",
            "schema": {},
        }
    )


# --- naming helpers -----------------------------------------------------


def test_module_name_for_tool_strips_run_prefix():
    assert module_name_for_tool("run_prodigy") == "prodigy"
    assert module_name_for_tool("run_openmm_minimize") == "openmm_minimize"


def test_tool_name_for_module_adds_run_prefix():
    assert tool_name_for_module("prodigy") == "run_prodigy"


def test_module_name_for_tool_rejects_non_run_name():
    with pytest.raises(ValueError):
        module_name_for_tool("describe_tool")


# --- the four shipped adapters still resolve -----------------------------


def test_the_four_shipped_adapters_are_discovered_and_valid():
    from protein_design_mcp.app import adapters_dir

    result = discover_adapters(adapters_dir())
    expected = {"run_prodigy", "run_ipsae", "run_openmm_minimize", "run_mpnn"}
    assert expected <= set(result.adapters)
    assert not (expected & set(result.broken))


def test_shipped_adapter_functions_are_callable_two_arg_functions():
    from protein_design_mcp.app import adapters_dir

    result = discover_adapters(adapters_dir())
    build_args, parse_output = result.adapters["run_prodigy"]
    assert callable(build_args)
    assert callable(parse_output)


def test_the_discovery_module_never_scans_itself():
    """Regression: the discovery machinery must not live inside the directory
    it scans. It used to (adapters/discovery.py), which made discovery.py a
    candidate '.py' file in its own scan — a false 'stranded adapter'
    (module_name='discovery', no run_discovery manifest) reported at every
    startup. The code now lives at protein_design_mcp.adapters_discovery.

    Asserted as an invariant rather than against a hardcoded module list:
    this file is scanned on every tool addition, and a snapshot of the shipped
    set would have to be edited ~30 more times while catching nothing.
    """
    from protein_design_mcp.app import adapters_dir

    result = discover_adapters(adapters_dir())

    for forbidden in ("discovery", "adapters_discovery"):
        assert forbidden not in result.module_names, (
            f"{forbidden!r} was scanned as an adapter; the discovery module "
            "has moved back inside the directory it scans"
        )
    assert result.module_names, "discovery found no adapters at all"


def test_every_adapter_module_matches_a_manifest():
    """The real invariant behind adapter discovery: the adapter set and the
    manifest set are the same set. A stranded adapter means a manifest was
    deleted or misnamed; a manifest with no adapter means a tool cannot run.
    Derived at runtime so it keeps holding as tools are added.
    """
    from protein_design_mcp.app import adapters_dir, manifest_dir

    modules = discover_adapters(adapters_dir()).module_names
    expected = {
        p.stem[len("run_"):] if p.stem.startswith("run_") else p.stem
        for p in manifest_dir().glob("*.yaml")
    }
    assert modules == expected, (
        f"adapters without a manifest: {sorted(modules - expected)}; "
        f"manifests without an adapter: {sorted(expected - modules)}"
    )

def test_build_registry_reports_no_stranded_adapters_for_the_real_shipped_tree(caplog):
    """End-to-end: booting the real server must not emit a false
    'adapter module(s) have no matching manifest' error for its own
    discovery machinery or any other shipped file."""
    import logging

    from protein_design_mcp.app import build_registry

    with caplog.at_level(logging.ERROR, logger="protein_design_mcp.app"):
        build_registry(device="cpu")

    assert "have no matching manifest" not in caplog.text


# --- corner case: empty directory ----------------------------------------


def test_empty_adapters_directory_yields_empty_result_not_an_error(tmp_path):
    result = discover_adapters(tmp_path)
    assert result.adapters == {}
    assert result.broken == {}
    assert result.module_names == frozenset()


def test_nonexistent_adapters_directory_yields_empty_result_not_an_error(tmp_path):
    result = discover_adapters(tmp_path / "does-not-exist")
    assert result.adapters == {}
    assert result.broken == {}


# --- corner case: non-module files are skipped ----------------------------


def test_init_pycache_and_stray_files_are_skipped_not_treated_as_broken(tmp_path):
    _write(tmp_path, "__init__.py", '"""Package docstring."""\n')
    _write(tmp_path, "notes.txt", "not python at all\n")
    pycache = tmp_path / "__pycache__"
    pycache.mkdir()
    (pycache / "prodigy.cpython-311.pyc").write_bytes(b"\x00\x01")
    _write(tmp_path, "prodigy.py", VALID_ADAPTER)

    result = discover_adapters(tmp_path)

    assert result.module_names == frozenset({"prodigy"})
    assert set(result.adapters) == {"run_prodigy"}
    assert result.broken == {}


# --- corner case: module name collides with a Python builtin -------------


def test_module_name_colliding_with_a_builtin_is_discovered_normally(tmp_path):
    _write(tmp_path, "list.py", VALID_ADAPTER)

    result = discover_adapters(tmp_path)

    assert "run_list" in result.adapters
    assert result.broken == {}


# --- a module missing parse_output is rejected by name --------------------


def test_module_missing_parse_output_is_rejected_by_name(tmp_path):
    _write(tmp_path, "half.py", MISSING_PARSE_OUTPUT)

    result = discover_adapters(tmp_path)

    assert "run_half" not in result.adapters
    assert "run_half" in result.broken
    reason = result.broken["run_half"]
    assert "half.py" in reason
    assert "parse_output" in reason


def test_module_missing_build_args_is_rejected_by_name(tmp_path):
    _write(tmp_path, "onlyparse.py", "def parse_output(manifest, run):\n    return {}\n")

    result = discover_adapters(tmp_path)

    assert "run_onlyparse" in result.broken
    assert "build_args" in result.broken["run_onlyparse"]


def test_module_with_wrong_arity_is_rejected_by_name(tmp_path):
    _write(tmp_path, "badarity.py", WRONG_ARITY)

    result = discover_adapters(tmp_path)

    assert "run_badarity" not in result.adapters
    assert "run_badarity" in result.broken
    reason = result.broken["run_badarity"]
    assert "badarity.py" in reason
    assert "parse_output" in reason


def test_module_that_raises_on_import_is_isolated(tmp_path):
    """One broken module (here: raises at import time) must not prevent a
    sibling module in the same directory from being discovered — the same
    isolation principle as load_manifests_resilient."""
    _write(tmp_path, "broken.py", RAISES_ON_IMPORT)
    _write(tmp_path, "prodigy.py", VALID_ADAPTER)

    result = discover_adapters(tmp_path)

    assert "run_prodigy" in result.adapters
    assert "run_broken" in result.broken
    assert "broken.py" in result.broken["run_broken"]


# --- a manifest with no adapter excludes only itself -----------------------


def test_manifest_with_no_adapter_excludes_only_itself(tmp_path):
    _write(tmp_path, "prodigy.py", VALID_ADAPTER)
    discovery = discover_adapters(tmp_path)

    reasons = missing_adapter_reasons(["run_prodigy", "run_missing"], discovery)

    assert "run_prodigy" not in reasons
    assert "run_missing" in reasons
    message = reasons["run_missing"]
    assert "run_missing" in message
    assert "missing.py" in message
    # Must not instruct a model to create the file — it can't.
    assert "add" not in message.lower()
    assert "create" not in message.lower()


def test_manifest_with_broken_adapter_module_is_excluded_with_the_reason(tmp_path):
    _write(tmp_path, "prodigy.py", VALID_ADAPTER)
    _write(tmp_path, "half.py", MISSING_PARSE_OUTPUT)
    discovery = discover_adapters(tmp_path)

    reasons = missing_adapter_reasons(["run_prodigy", "run_half"], discovery)

    assert "run_prodigy" not in reasons
    assert "run_half" in reasons
    assert "parse_output" in reasons["run_half"]


def test_missing_adapter_reason_flows_through_the_registry_exclusion_mechanism(tmp_path):
    """End-to-end at the registry seam: the surviving manifest resolves
    normally, the one lacking an adapter is excluded — not with 'unknown
    tool', but a reason naming it."""
    _write(tmp_path, "prodigy.py", VALID_ADAPTER)
    discovery = discover_adapters(tmp_path)

    prodigy = _manifest("run_prodigy", category="scoring")
    missing = _manifest("run_missing", category="binder_generation")

    reasons = missing_adapter_reasons([m.name for m in (prodigy, missing)], discovery)
    manifests = [m for m in (prodigy, missing) if m.name not in reasons]

    registry = ToolRegistry(manifests, load_failures=reasons)

    assert registry.resolve("run_prodigy").name == "run_prodigy"
    assert "run_missing" not in {t.name for t in registry.tools()}
    with pytest.raises(ToolNotAvailable) as excinfo:
        registry.resolve("run_missing")
    assert "run_missing" in str(excinfo.value)
    assert "unknown tool" not in str(excinfo.value)


# --- a stranded adapter is reported ----------------------------------------


def test_stranded_adapter_is_reported(tmp_path):
    _write(tmp_path, "prodigy.py", VALID_ADAPTER)
    _write(tmp_path, "orphan.py", VALID_ADAPTER)
    discovery = discover_adapters(tmp_path)

    reasons = stranded_adapter_reasons(["run_prodigy"], discovery)

    assert "prodigy" not in reasons
    assert "orphan" in reasons
    message = reasons["orphan"]
    assert "orphan.py" in message
    assert "run_orphan" in message


def test_no_stranded_adapters_when_every_module_has_a_manifest(tmp_path):
    _write(tmp_path, "prodigy.py", VALID_ADAPTER)
    discovery = discover_adapters(tmp_path)

    reasons = stranded_adapter_reasons(["run_prodigy"], discovery)

    assert reasons == {}
