"""Runtime launch contracts for the integrated image, without loading models."""
import ast
import json
import sys
from pathlib import Path

import pytest
import yaml

from protein_design_mcp.dispatch.env import EnvDispatcher
from protein_design_mcp.manifest.schema import EngineSpec

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "src/protein_design_mcp/manifests"


def documents():
    return {path.stem: yaml.safe_load(path.read_text()) for path in MANIFESTS.glob("*.yaml")}


def test_integrated_launch_settings_have_no_personal_runtime_dependencies():
    for name, document in documents().items():
        engine = document["engine"]
        serialized = json.dumps(engine)
        assert "/home/" not in serialized, name
        assert "/file_server/" not in serialized, name
        assert "prefix_host" not in engine, name
        if "prefix" in engine:
            assert engine["prefix"].startswith("/opt/conda/envs/") or engine["prefix"] == "/alphafold3_venv", name
        for mount in engine.get("mounts", []):
            assert mount.startswith(("/data/databases/", "/data/models/", "/data/licenses/")), (name, mount)


def test_model_and_database_locations_are_explicit():
    docs = documents()
    assert docs["run_colabfold_search"]["engine"]["env"] == "colabfold"
    assert docs["run_boltz"]["engine"]["env_vars"]["BOLTZ_CACHE"] == "/opt/models/boltz"
    assert docs["run_protenix"]["engine"]["env_vars"]["PROTENIX_ROOT_DIR"] == "/opt/models/protenix"
    assert docs["run_chai1"]["engine"]["env_vars"]["CHAI_DOWNLOADS_DIR"] == "/opt/conda/envs/chai1/lib/python3.10/site-packages/downloads"
    for name in ("run_rf3", "run_rfdiffusion3_binder", "run_rfdiffusion3_scaffold"):
        assert docs[name]["engine"]["env_vars"]["FOUNDRY_CHECKPOINT_DIRS"] == "/opt/models/foundry"
    assert docs["run_promera"]["engine"]["env_vars"]["TINYPROT_CACHE"] == "/data/databases/tinyprot"
    assert "PYTHONPATH" not in docs["run_rosetta_interface"]["engine"].get("env_vars", {})
    assert docs["run_rosetta_interface"]["requires"]["files"] == ["/data/licenses/pyrosetta/pyrosetta/__init__.py"]
    assert docs["run_boltzgen_analyze"]["schema"]["foldseek_binary"]["default"] == "/usr/local/bin/foldseek"


def test_engine_code_has_no_personal_path_constants():
    for directory in (ROOT / "scripts/engines", ROOT / "src/protein_design_mcp/adapters"):
        for path in directory.glob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Constant) and isinstance(node.value, str) and "\n" not in node.value:
                    assert "/home/" not in node.value and "/file_server/" not in node.value, (path, node.lineno)


@pytest.mark.asyncio
@pytest.mark.parametrize("repo", ["promera", "rf3", "rfd3"])
async def test_dispatch_keeps_writable_home_and_explicit_model_paths(tmp_path, monkeypatch, repo):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    dispatcher = EnvDispatcher(runner=None, scratch_root=tmp_path)
    engine = EngineSpec(repo=repo, env="bundled", entry=(sys.executable,),
                        env_vars={"FOUNDRY_CHECKPOINT_DIRS": "/opt/models/foundry"})
    result = await dispatcher.run(engine, ["-c", "import json,os; print(json.dumps({k:os.environ[k] for k in ('HOME','FOUNDRY_CHECKPOINT_DIRS','TRITON_CACHE_DIR')}))"], timeout=10)
    env = json.loads(result.stdout)
    assert env["HOME"] == str(home)
    assert env["FOUNDRY_CHECKPOINT_DIRS"] == "/opt/models/foundry"
    assert Path(env["TRITON_CACHE_DIR"]).is_relative_to(tmp_path)


def _colabfold_wrapper():
    import importlib.util
    spec = importlib.util.spec_from_file_location("portable_colabfold_search", ROOT / "scripts/engines/colabfold_search.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_colabfold_remote_does_not_require_local_databases(tmp_path, monkeypatch):
    from types import SimpleNamespace
    module = _colabfold_wrapper()
    monkeypatch.setattr(module, "_parse_args", lambda: SimpleNamespace(backend="remote", sequence="MKT", use_env=True, filter=1))
    calls = []
    monkeypatch.setattr(module, "_run_remote", lambda *args: calls.append(args))
    module.main()
    assert calls == [("MKT", True, 1)]
    assert documents()["run_colabfold_search"]["engine"].get("mounts", []) == []


def test_colabfold_local_missing_database_fails_before_subprocess(tmp_path, monkeypatch):
    from types import SimpleNamespace
    module = _colabfold_wrapper()
    monkeypatch.setattr(module, "_parse_args", lambda: SimpleNamespace(backend="local", db_root=str(tmp_path), db1="uniref", db3="envdb", use_env=True))
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **kw: pytest.fail("search launched without databases"))
    with pytest.raises(SystemExit, match="Local ColabFold databases unavailable.*uniref"):
        module.main()


def test_colabfold_local_checks_only_selected_databases(tmp_path):
    module = _colabfold_wrapper()
    for suffix in (".dbtype", ".index", ".0"):
        (tmp_path / ("uniref" + suffix)).write_bytes(b"data")
    module._require_local_databases(str(tmp_path), ["uniref"])
    with pytest.raises(SystemExit, match="envdb"):
        module._require_local_databases(str(tmp_path), ["uniref", "envdb"])
