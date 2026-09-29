"""The image asset copier must honor its allowlist without leaking host files."""

import json
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/prepare_integrated_assets.py"


def run_manifest(tmp_path, entries, *args):
    manifest = tmp_path / "assets.json"
    manifest.write_text(json.dumps({"staging_entries": entries}))
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--manifest", str(manifest),
         "--rootfs", str(tmp_path / "rootfs"), *args],
        capture_output=True, text=True,
    )


def test_dry_run_does_not_copy_and_execution_makes_independent_files(tmp_path):
    source = tmp_path / "model.pt"
    source.write_bytes(b"public model")
    entries = [{"kind": "file", "source": str(source), "destination": "/opt/models/model.pt"}]
    result = run_manifest(tmp_path, entries)
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "rootfs").exists()
    result = run_manifest(tmp_path, entries, "--execute")
    assert result.returncode == 0, result.stderr
    copied = tmp_path / "rootfs/opt/models/model.pt"
    assert copied.read_bytes() == source.read_bytes()
    assert copied.stat().st_ino != source.stat().st_ino
    copied.write_bytes(b"changed")
    assert source.read_bytes() == b"public model"


def test_git_source_only_includes_tracked_safe_files_and_notices(tmp_path):
    repo = tmp_path / "checkout"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    for name in ["engine.py", "LICENSE", ".env", "token", "untracked.pt"]:
        (repo / name).write_text(name)
    subprocess.run(["git", "-C", str(repo), "add", "engine.py", "LICENSE", ".env", "token"], check=True)
    result = run_manifest(tmp_path, [{"kind": "git", "source": str(repo), "destination": "/opt/engines/demo"}], "--execute")
    assert result.returncode == 0, result.stderr
    copied = tmp_path / "rootfs/opt/engines/demo"
    assert sorted(p.name for p in copied.iterdir()) == ["LICENSE", "engine.py"]


def test_hf_only_copies_requested_revision_and_materializes_shared_blobs(tmp_path):
    model = tmp_path / "hub/models--public--model"
    shared = tmp_path / "hub/blobs"
    shared.mkdir(parents=True)
    (shared / "good").write_bytes(b"checkpoint")
    for rev in ["active", "old"]:
        snapshot = model / "snapshots" / rev
        snapshot.mkdir(parents=True)
        (snapshot / "model.pt").symlink_to(shared / "good")
    (model / "refs").mkdir()
    (model / "refs/main").write_text("active")
    result = run_manifest(tmp_path, [{"kind": "hf_snapshot", "source": str(model),
        "revision": "active", "destination": "/opt/models/huggingface/hub/models--public--model",
        "allowed_blob_roots": [str(shared)]}], "--execute")
    assert result.returncode == 0, result.stderr
    copied = tmp_path / "rootfs/opt/models/huggingface/hub/models--public--model"
    assert (copied / "snapshots/active/model.pt").read_bytes() == b"checkpoint"
    assert not (copied / "snapshots/active/model.pt").is_symlink()
    assert not (copied / "snapshots/old").exists()
    assert not (copied / "blobs").exists()
    assert (copied / "refs/main").read_text() == "active"


@pytest.mark.parametrize("destination", ["../../escape", "/opt/../escape", "/data/models/alphafold3/af3.bin"])
def test_rejects_invalid_or_restricted_destinations_before_copying(tmp_path, destination):
    source = tmp_path / "safe.txt"
    source.write_text("safe")
    result = run_manifest(tmp_path, [{"kind": "file", "source": str(source), "destination": destination}], "--execute")
    assert result.returncode != 0
    assert not (tmp_path / "rootfs").exists()


def test_tree_rejects_symlink_outside_allowlisted_source(tmp_path):
    tree = tmp_path / "tree"
    tree.mkdir()
    secret = tmp_path / "private.txt"
    secret.write_text("private")
    (tree / "weights.pt").symlink_to(secret)
    result = run_manifest(tmp_path, [{"kind": "tree", "source": str(tree), "destination": "/opt/models/tree"}], "--execute")
    assert result.returncode != 0
    assert not (tmp_path / "rootfs").exists()


def test_git_internal_directory_symlink_preserves_runtime_resources(tmp_path):
    repo = tmp_path / "checkout"
    (repo / "config").mkdir(parents=True)
    (repo / "config/runtime.yaml").write_text("enabled: true")
    (repo / "tools").mkdir()
    (repo / "tools/config").symlink_to("../config", target_is_directory=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    result = run_manifest(tmp_path, [{"kind": "git", "source": str(repo), "destination": "/opt/engines/demo"}], "--execute")
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "rootfs/opt/engines/demo/tools/config/runtime.yaml").read_text() == "enabled: true"


def test_rejects_destination_symlink_escape(tmp_path):
    source = tmp_path / "model.pt"
    source.write_bytes(b"model")
    outside = tmp_path / "outside"
    outside.mkdir()
    rootfs = tmp_path / "rootfs"
    rootfs.mkdir()
    (rootfs / "opt").symlink_to(outside)
    result = run_manifest(tmp_path, [{"kind": "file", "source": str(source), "destination": "/opt/model.pt"}], "--execute")
    assert result.returncode != 0
    assert not list(outside.iterdir())


def test_rewrite_only_changes_staged_text_without_copying_weights_or_host_files(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    config = source / "config.yaml"
    config.write_text("weights: /home/person/models/model.pt\n")
    (source / "model.pt").write_bytes(b"/home/person/models/model.pt")
    entries = [{"kind": "tree", "source": str(source), "destination": "/opt/engines/demo"}]
    assert run_manifest(tmp_path, entries, "--execute").returncode == 0
    path_map = tmp_path / "paths.json"
    path_map.write_text(json.dumps({"/home/person/models": "/opt/models/demo"}))
    result = run_manifest(tmp_path, entries, "--rewrite-only", "--path-map", str(path_map))
    assert result.returncode == 0, result.stderr
    staged = tmp_path / "rootfs/opt/engines/demo"
    assert (staged / "config.yaml").read_text() == "weights: /opt/models/demo/model.pt\n"
    assert (staged / "model.pt").read_bytes() == b"/home/person/models/model.pt"
    assert config.read_text() == "weights: /home/person/models/model.pt\n"
