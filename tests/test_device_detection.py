"""``DEVICE=auto`` must not answer "cpu" just because torch is absent.

v1 detected CUDA with ``import torch; torch.cuda.is_available()``, falling
back to ``"cpu"`` on ImportError. v2's architecture makes that fallback always
fire: the image's ``server`` environment deliberately shares no dependencies
with any engine, so it has no torch -- each engine has its own. The result was
a container started with ``--device=nvidia.com/gpu=7``, with ``/dev/nvidia7``
present, reporting ``Device mode: cpu`` and excluding 27 of 39 tools.

The observable fact the server actually needs is not "can I run CUDA myself"
(it never runs CUDA; the engines do, each in its own environment) but "is a
GPU attached to this container". That is a numbered device node, and it is
readable without any Python package.

``/proc/driver/nvidia/gpus`` is NOT usable for this: it is the host driver's
procfs and lists all eight GPUs inside a container pinned to one.
"""

from __future__ import annotations

import pytest

from protein_design_mcp.server import detect_device


def _make_nodes(tmp_path, names):
    for name in names:
        (tmp_path / name).touch()
    return tmp_path


def test_a_numbered_device_node_means_cuda(tmp_path):
    dev = _make_nodes(tmp_path, ["nvidia7", "nvidiactl", "nvidia-uvm"])
    assert detect_device({}, dev) == "cuda"


def test_control_nodes_alone_are_not_a_gpu(tmp_path):
    """``nvidiactl``/``nvidia-uvm``/``nvidia-modeset`` can be present without
    any GPU being assigned. Only a NUMBERED node is one."""
    dev = _make_nodes(tmp_path, ["nvidiactl", "nvidia-uvm", "nvidia-uvm-tools", "nvidia-modeset"])
    assert detect_device({}, dev) == "cpu"


def test_no_nvidia_nodes_at_all_means_cpu(tmp_path):
    assert detect_device({}, _make_nodes(tmp_path, ["null", "zero"])) == "cpu"


def test_a_missing_dev_directory_means_cpu(tmp_path):
    assert detect_device({}, tmp_path / "does-not-exist") == "cpu"


@pytest.mark.parametrize("value", ["cpu", "cuda"])
def test_an_explicit_device_always_wins(tmp_path, value):
    """An operator overriding DEVICE is making a statement, not a request --
    including forcing cpu on a machine that does have a GPU."""
    dev = _make_nodes(tmp_path, ["nvidia0"])
    assert detect_device({"DEVICE": value}, dev) == value


def test_an_explicit_device_is_case_insensitive(tmp_path):
    assert detect_device({"DEVICE": "CUDA"}, tmp_path) == "cuda"


def test_auto_is_the_default_when_device_is_unset_or_blank(tmp_path):
    dev = _make_nodes(tmp_path, ["nvidia3"])
    assert detect_device({}, dev) == "cuda"
    assert detect_device({"DEVICE": ""}, dev) == "cuda"
    assert detect_device({"DEVICE": "auto"}, dev) == "cuda"
