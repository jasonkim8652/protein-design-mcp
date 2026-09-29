"""Exercise portable staging against real files, without copying GPU environments."""

import importlib.util
import json
import os
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def staging():
    script = Path(__file__).parents[1] / "scripts" / "prepare_integrated_envs.py"
    assert script.exists(), "portable environment staging script is missing"
    spec = importlib.util.spec_from_file_location("prepare_integrated_envs", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def put(root, name, content):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def prefix(tmp_path, name="toy"):
    root = tmp_path / "host" / name
    put(root, "conda-meta/history", "")
    put(root, "bin/tool", f"#!/bin/sh\necho '{root}/share/payload'\n").chmod(0o755)
    put(root, "share/payload", "hello\n")
    put(root, "share/licenses/toy/LICENSE", "Keep this package license.\n")
    record = {
        "name": "toy", "version": "1", "build": "0", "url": "https://example.org/toy",
        "files": ["bin/tool", "share/payload", "share/licenses/toy/LICENSE"],
        "link": {"source": str(tmp_path / "missing-cache")},
        "paths_data": {"paths": [{"_path": "bin/tool", "file_mode": "text",
                                    "prefix_placeholder": "/build/placeholder"}]},
    }
    put(root, "conda-meta/toy-1-0.json", json.dumps(record))
    return root


def test_stage_relocates_without_host_mutation_or_shared_inodes(staging, tmp_path):
    pytest.importorskip("conda_pack")
    source = prefix(tmp_path)
    site = "lib/python3.11/site-packages/"
    put(source, site + "editable.pth", "/home/example/project/src\n")
    put(source, site + "__editable___model_finder.py",
        "MAPPING = {'model': '/home/example/project/src/model'}\n")
    old = (source / "bin/tool").read_bytes()
    result = staging.stage_environment(
        {"source": str(source), "destination": "/opt/conda/envs/toy"},
        tmp_path / "rootfs", {"/home/example/project": "/opt/engines/project"})
    output = tmp_path / "rootfs/opt/conda/envs/toy"
    assert result["files"] > 0
    assert "/opt/conda/envs/toy/share/payload" in (output / "bin/tool").read_text()
    assert (output / (site + "editable.pth")).read_text() == "/opt/engines/project/src\n"
    assert "/opt/engines/project/src/model" in (
        output / (site + "__editable___model_finder.py")).read_text()
    assert (source / "bin/tool").read_bytes() == old
    assert os.stat(source / "share/payload").st_ino != os.stat(output / "share/payload").st_ino
    assert (output / "share/licenses/toy/LICENSE").read_text() == "Keep this package license.\n"
    assert not list(output.glob("tmp*")), "conda-pack scratch files must not enter the image"


def test_rejects_missing_managed_runtime_without_pip_replacement(staging, tmp_path):
    source = prefix(tmp_path)
    (source / "share/payload").unlink()
    with pytest.raises(ValueError, match="missing managed"):
        staging.stage_environment({"source": str(source), "destination": "/opt/conda/envs/toy"},
                                  tmp_path / "rootfs", {}, dry_run=True)


def test_verified_pip_replacement_drops_only_stale_conda_record(staging, tmp_path):
    pytest.importorskip("conda_pack")
    source = prefix(tmp_path)
    (source / "share/payload").unlink()
    site = "lib/python3.11/site-packages/"
    put(source, site + "toy/__init__.py", "VALUE = 2\n")
    put(source, site + "toy-2.dist-info/METADATA", "Name: toy\nVersion: 2\n")
    put(source, site + "toy-2.dist-info/RECORD", "toy/__init__.py,,\ntoy-2.dist-info/METADATA,,\n")
    staging.stage_environment({"source": str(source), "destination": "/opt/conda/envs/toy"},
                              tmp_path / "rootfs", {})
    out = tmp_path / "rootfs/opt/conda/envs/toy"
    assert not (out / "conda-meta/toy-1-0.json").exists()
    assert (out / (site + "toy/__init__.py")).exists()
    assert (source / "conda-meta/toy-1-0.json").exists()


def test_missing_pip_record_file_is_not_a_valid_replacement(staging, tmp_path):
    source = prefix(tmp_path)
    (source / "share/payload").unlink()
    site = "lib/python3.11/site-packages/"
    put(source, site + "toy-2.dist-info/METADATA", "Name: toy\nVersion: 2\n")
    put(source, site + "toy-2.dist-info/RECORD", "toy/missing.py,,\n")
    with pytest.raises(ValueError, match="missing managed"):
        staging.stage_environment({"source": str(source), "destination": "/opt/conda/envs/toy"},
                                  tmp_path / "rootfs", {}, dry_run=True)


def test_numpy_base_conda_record_recognizes_complete_numpy_wheel(staging, tmp_path):
    source = prefix(tmp_path)
    record = source / "conda-meta/toy-1-0.json"
    data = json.loads(record.read_text())
    data["name"] = "numpy-base"
    record.write_text(json.dumps(data))
    (source / "share/payload").unlink()
    site = "lib/python3.11/site-packages/"
    put(source, site + "numpy/__init__.py", "VALUE = 2\n")
    put(source, site + "numpy-2.dist-info/METADATA", "Name: numpy\nVersion: 2\n")
    put(source, site + "numpy-2.dist-info/RECORD", "numpy/__init__.py,,\n")
    result = staging.stage_environment(
        {"source": str(source), "destination": "/opt/conda/envs/toy"}, tmp_path / "rootfs", {},
        dry_run=True)
    assert result["repairs"][0]["package"] == "numpy-base"


def test_bindcraft_drops_pyrosetta_but_preserves_other_licenses(staging, tmp_path):
    pytest.importorskip("conda_pack")
    source = prefix(tmp_path, "BindCraft")
    site = "lib/python3.11/site-packages/"
    put(source, site + "pyrosetta/__init__.py", "restricted = True\n")
    put(source, site + "rosetta/__init__.py", "from pyrosetta import *\n")
    put(source, site + "pyrosetta-1.dist-info/METADATA", "Name: pyrosetta\n")
    put(source, "share/licenses/pyrosetta/LICENSE", "restricted\n")
    staging.stage_environment({"source": str(source), "destination": "/opt/conda/envs/BindCraft"},
                              tmp_path / "rootfs", {})
    out = tmp_path / "rootfs/opt/conda/envs/BindCraft"
    assert not list(out.rglob("*pyrosetta*"))
    assert not (out / (site + "rosetta")).exists()
    assert (out / "share/licenses/toy/LICENSE").exists()
    assert (source / (site + "pyrosetta/__init__.py")).exists()


@pytest.mark.parametrize("dry_run", [True, False])
def test_unknown_host_path_fails_without_publishing_partial_output(staging, tmp_path, dry_run):
    pytest.importorskip("conda_pack")
    source = prefix(tmp_path)
    put(source, "lib/python3.11/site-packages/model.pth", "/home/unknown/private/src\n")
    with pytest.raises(ValueError, match="unmapped host path"):
        staging.stage_environment({"source": str(source), "destination": "/opt/conda/envs/toy"},
                                  tmp_path / "rootfs", {}, dry_run=dry_run)
    assert not (tmp_path / "rootfs/opt/conda/envs/toy").exists()


@pytest.mark.parametrize("builder", ["task_123", "task_a9LjGuP7SLSzS_0"])
def test_packager_build_paths_become_system_tools_not_private_dependencies(staging, tmp_path, builder):
    pytest.importorskip("conda_pack")
    source = prefix(tmp_path)
    put(source, "bin/c_rehash", f"#!/home/{builder}/croot/openssl_42/_build_env/bin/perl\n")
    put(source, "bin/freetype-config", "#!/bin/sh\n"
        "/home/conda/feedstock_root/build_artifacts/freetype_42/_build_env/bin/"
        "x86_64-conda-linux-gnu-pkg-config --version\n")
    staging.stage_environment({"source": str(source), "destination": "/opt/conda/envs/toy"},
                              tmp_path / "rootfs", {})
    out = tmp_path / "rootfs/opt/conda/envs/toy"
    assert (out / "bin/c_rehash").read_text() == "#!/usr/bin/perl\n"
    assert "/usr/bin/pkg-config --version" in (out / "bin/freetype-config").read_text()


def test_private_install_configuration_and_history_are_not_published(staging, tmp_path):
    pytest.importorskip("conda_pack")
    source = prefix(tmp_path)
    put(source, "conda-meta/history", "private installation command\n")
    put(source, "pip.conf", "[global]\nindex-url = https://example.invalid/private\n")
    put(source, ".condarc", "channels: [private]\n")
    staging.stage_environment({"source": str(source), "destination": "/opt/conda/envs/toy"},
                              tmp_path / "rootfs", {})
    out = tmp_path / "rootfs/opt/conda/envs/toy"
    assert not (out / "conda-meta/history").exists()
    assert not (out / "pip.conf").exists()
    assert not (out / ".condarc").exists()


def test_rootfs_symlink_cannot_redirect_writes(staging, tmp_path):
    source = prefix(tmp_path)
    root = tmp_path / "rootfs"
    root.mkdir()
    (tmp_path / "outside").mkdir()
    (root / "opt").symlink_to(tmp_path / "outside", target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        staging.stage_environment({"source": str(source), "destination": "/opt/conda/envs/toy"},
                                  root, {}, dry_run=True)


@pytest.mark.parametrize("failure", [OSError("simulated write failure"), KeyboardInterrupt()])
def test_conda_pack_write_errors_never_publish_partial_environment(staging, tmp_path, monkeypatch,
                                                                 failure):
    pytest.importorskip("conda_pack")
    from conda_pack.core import Packer
    source = prefix(tmp_path)
    original = Packer.add

    def fail_during_pack(self, file):
        if file.target == "share/payload":
            raise failure
        return original(self, file)

    monkeypatch.setattr(Packer, "add", fail_during_pack)
    with pytest.raises(type(failure)):
        staging.stage_environment({"source": str(source), "destination": "/opt/conda/envs/toy"},
                                  tmp_path / "rootfs", {})
    assert not (tmp_path / "rootfs/opt/conda/envs/toy").exists()


def test_silently_omitted_archive_file_fails_completeness_audit(staging, tmp_path, monkeypatch):
    pytest.importorskip("conda_pack")
    from conda_pack.core import Packer
    source = prefix(tmp_path)
    original = Packer.add

    def omit_file(self, file):
        if file.target != "share/payload":
            return original(self, file)

    monkeypatch.setattr(Packer, "add", omit_file)
    with pytest.raises(ValueError, match="incomplete staged"):
        staging.stage_environment({"source": str(source), "destination": "/opt/conda/envs/toy"},
                                  tmp_path / "rootfs", {})
    assert not (tmp_path / "rootfs/opt/conda/envs/toy").exists()


def test_truncated_binary_fails_completeness_audit(staging, tmp_path, monkeypatch):
    pytest.importorskip("conda_pack")
    from conda_pack.core import Packer
    source = prefix(tmp_path)
    (source / "share/payload").write_bytes(b"\x7fELF\0binary payload that must survive")
    original = Packer.add

    def truncate_binary(self, file):
        original(self, file)
        if file.target == "share/payload":
            (Path(self.archive.output) / file.target).write_bytes(b"\x7fELF\0")

    monkeypatch.setattr(Packer, "add", truncate_binary)
    with pytest.raises(ValueError, match="incomplete staged.*1 damaged"):
        staging.stage_environment({"source": str(source), "destination": "/opt/conda/envs/toy"},
                                  tmp_path / "rootfs", {})
    assert not (tmp_path / "rootfs/opt/conda/envs/toy").exists()


@pytest.mark.parametrize("destination", ["../../escape", "/opt/conda/envs/../escape", "/home/user/env"])
def test_destination_cannot_escape_environment_root(staging, tmp_path, destination):
    source = prefix(tmp_path)
    with pytest.raises(ValueError, match="destination"):
        staging.stage_environment({"source": str(source), "destination": destination},
                                  tmp_path / "rootfs", {}, dry_run=True)


def test_esm_env_removes_only_unrelated_pioneer_editable(staging, tmp_path):
    pytest.importorskip("conda_pack")
    source = prefix(tmp_path, "esm_env")
    site = "lib/python3.11/site-packages/"
    put(source, site + "__editable__.pioneer-0.1.pth", "/home/example/pioneer/src\n")
    put(source, site + "pioneer-0.1.dist-info/direct_url.json", '{"url":"file:///home/example/pioneer"}')
    put(source, site + "esm/__init__.py", "MODEL = 1\n")
    staging.stage_environment({"source": str(source), "destination": "/opt/conda/envs/esm_env"},
                              tmp_path / "rootfs", {})
    out = tmp_path / "rootfs/opt/conda/envs/esm_env"
    assert not list(out.rglob("*pioneer*"))
    assert (out / (site + "esm/__init__.py")).exists()
