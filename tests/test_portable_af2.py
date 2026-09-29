"""Deployment regressions: inaccessible host paths and portable AF2 dispatch."""
import importlib.util
from pathlib import Path
import sys

import pytest
import yaml

from protein_design_mcp.manifest.loader import load_manifests_resilient
from protein_design_mcp.manifest.registry import ToolNotAvailable, ToolRegistry

ROOT = Path(__file__).resolve().parents[1]


def test_unavailable_alternative_does_not_hide_a_working_tool(tmp_path, monkeypatch):
    monkeypatch.setenv("STRICT_MANIFESTS", "0")
    for name, mounts, doc in [
        ("run_good", [], "Alternative: run_private."),
        ("run_private", ["/missing/private/weights"], "Alternative: run_good."),
    ]:
        (tmp_path / f"{name}.yaml").write_text(yaml.safe_dump(dict(
            name=name, category="scoring", summary="Score", schema={},
            doc="## When to use this instead of the alternatives\n" + doc,
            engine=dict(repo="test", env="test", entry=["python"], mounts=mounts))))
    result = load_manifests_resilient(tmp_path)
    assert {m.name for m in result.manifests} == {"run_good", "run_private"}
    assert not result.reasons
    registry = ToolRegistry(result.manifests)
    assert [tool.name for tool in registry.tools()] == ["run_good"]
    with pytest.raises(ToolNotAvailable, match="/missing/private/weights"):
        registry.resolve("run_private")


@pytest.mark.parametrize("strict", [False, True])
def test_permission_denied_mount_is_a_runtime_availability_failure(tmp_path, monkeypatch, strict):
    monkeypatch.setenv("STRICT_MANIFESTS", "1" if strict else "0")
    template = dict(category="scoring", summary="Score", schema={},
                    doc="## When to use this instead of the alternatives\n"
                        "Use run_good for public inputs and run_private for private inputs.")
    for name, mounts in [("run_good", []), ("run_private", ["/private/weights"])]:
        (tmp_path / f"{name}.yaml").write_text(yaml.safe_dump(dict(
            template, name=name, engine=dict(repo="test", env="test", entry=["python"], mounts=mounts))))
    exists = Path.exists

    def probe(path):
        if str(path) == "/private/weights":
            raise PermissionError("permission denied: /private/weights")
        return exists(path)

    monkeypatch.setattr(Path, "exists", probe)
    result = load_manifests_resilient(tmp_path)
    assert {m.name for m in result.manifests} == {"run_good", "run_private"}
    assert not result.reasons
    registry = ToolRegistry(result.manifests)
    assert [tool.name for tool in registry.tools()] == ["run_good"]
    with pytest.raises(ToolNotAvailable, match="missing or inaccessible"):
        registry.resolve("run_private")


def test_af2_wrapper_uses_deployment_weights_path(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("af2_wrapper", ROOT / "scripts/engines/alphafold2_multimer.py")
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("COLABFOLD_WEIGHTS_DIR", "/opt/custom-weights")
    monkeypatch.setattr(sys, "argv", ["af2", "--sequences", '["AAAA", "GGGG"]',
        "--msa-mode", "single_sequence", "--num-recycle", "0", "--num-models", "1",
        "--num-seeds", "1", "--random-seed", "0", "--num-ensemble", "1",
        "--pair-mode", "unpaired_paired", "--pair-strategy", "greedy",
        "--rank", "auto", "--stop-at-score", "100"])
    commands = []

    def run(cmd, **kwargs):
        from types import SimpleNamespace
        commands.append(cmd)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(wrapper.subprocess, "run", run)
    wrapper.main()
    assert commands[0][commands[0].index("--data") + 1] == "/opt/custom-weights"
    assert (tmp_path / "query.fasta").read_text() == ">query\nAAAA:GGGG\n"
