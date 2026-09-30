"""Layered extraction must preserve the audited payload exactly."""
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import tarfile

import pytest


def load_planner():
    script = Path(__file__).parents[1] / "scripts/plan_integrated_layers.py"
    assert script.exists(), "Missing registry-sized payload layer planner"
    spec = importlib.util.spec_from_file_location("integrated_layers", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_member_ranges_preserve_files_modes_and_cross_layer_links(tmp_path):
    planner = load_planner()
    archive = tmp_path / "payload.tar"
    with tarfile.open(archive, "w", format=tarfile.GNU_FORMAT) as tar:
        for name, data in [("opt/a", b"alpha" * 300),
                           ("opt/" + "long" * 50, b"large" * 1000),
                           ("opt/z", b"last")]:
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = len(data), 0o755, 123456
            tar.addfile(info, io.BytesIO(data))
        link = tarfile.TarInfo("opt/hard")
        link.type, link.linkname = tarfile.LNKTYPE, "opt/a"
        tar.addfile(link)
        link = tarfile.TarInfo("opt/sym")
        link.type, link.linkname = tarfile.SYMTYPE, "a"
        tar.addfile(link)
    ranges = planner.plan_ranges(archive, target_bytes=2048)
    assert len(ranges) >= 3
    assert any(item["length"] > 2048 for item in ranges)
    assert ranges[0]["offset"] == 0
    for left, right in zip(ranges, ranges[1:]):
        assert left["offset"] + left["length"] == right["offset"]
    normal, layered = tmp_path / "normal", tmp_path / "layered"
    normal.mkdir()
    layered.mkdir()
    subprocess.run(["tar", "-xf", str(archive), "-C", str(normal)], check=True)
    for item in ranges:
        with archive.open("rb") as stream:
            stream.seek(item["offset"])
            subprocess.run(["tar", "-xf", "-", "-C", str(layered)],
                           input=stream.read(item["length"]), check=True)
    for path in normal.rglob("*"):
        other = layered / path.relative_to(normal)
        assert path.lstat().st_mode == other.lstat().st_mode
        if path.is_symlink():
            assert os.readlink(path) == os.readlink(other)
        elif path.is_file():
            assert path.read_bytes() == other.read_bytes()
            assert path.stat().st_mtime == other.stat().st_mtime
    assert (layered / "opt/a").stat().st_ino == (layered / "opt/hard").stat().st_ino


def test_dockerfile_generation_requires_exact_marker_and_uses_all_ranges(tmp_path):
    planner = load_planner()
    ranges = [{"offset": 0, "length": 2048}, {"offset": 2048, "length": 512}]
    rendered = planner.render_dockerfile("FROM base\n" + planner.MARKER + "\nUSER user\n", ranges)
    assert rendered.count("from=payload,source=rootfs.tar") == 2
    assert "skip=2048 count=512" in rendered
    assert "USER user" in rendered
    with pytest.raises(ValueError, match="marker"):
        planner.render_dockerfile("FROM base", ranges)


def test_compressed_payload_and_invalid_limits_are_rejected(tmp_path):
    planner = load_planner()
    archive = tmp_path / "compressed.tar.gz"
    with tarfile.open(archive, "w:gz"):
        pass
    with pytest.raises(tarfile.ReadError):
        planner.plan_ranges(archive, target_bytes=2048)
    with pytest.raises(ValueError):
        planner.plan_ranges(archive, target_bytes=0)


@pytest.mark.parametrize("damage", ["middle_header", "truncated", "trailing_data"])
def test_damaged_archive_cannot_silently_publish_a_prefix(tmp_path, damage):
    planner = load_planner()
    archive = tmp_path / "damaged.tar"
    with tarfile.open(archive, "w", format=tarfile.GNU_FORMAT) as tar:
        for name in ("a", "b", "c"):
            info = tarfile.TarInfo(name)
            info.size = 1
            tar.addfile(info, io.BytesIO(b"x"))
    data = bytearray(archive.read_bytes())
    if damage == "middle_header":
        data[1024:1536] = b"x" * 512
    elif damage == "truncated":
        data = data[:2600]
    else:
        data.extend(b"unparsed data")
    archive.write_bytes(data)
    with pytest.raises((ValueError, tarfile.ReadError)):
        planner.plan_ranges(archive, target_bytes=2048)
