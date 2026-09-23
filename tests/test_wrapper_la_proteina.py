"""Unit tests for scripts/engines/la_proteina.py's ``_build_repo_view``.

This is the fix for the read-only-mount defect task-13-report.md found for
``run_la_proteina``: hydra's own config-path resolution
(``compute_search_path_dir`` in hydra's ``_internal/utils.py``) calls
``realpath(dirname(calling_file))`` -- meaning a plain SYMLINK to
``generate.py`` is not enough, hydra follows it straight back to the real,
read-only-mounted checkout, and the wrapper's per-call config directory
write fails with ``OSError: [Errno 30] Read-only file system``.

``_build_repo_view`` gives ``generate.py`` a REAL (non-symlink) containing
directory under the scratch workdir instead, so hydra's ``realpath()`` call
resolves to somewhere genuinely writable. Everything else needed for
imports is a plain symlink, which ordinary Python imports follow without
this realpath problem. These tests build a small fake "repo" and assert the
mirror ``_build_repo_view`` produces has exactly this shape -- most
importantly, that a hydra-style ``realpath(dirname(...))`` call against the
mirrored ``generate.py`` lands under the scratch workdir, never under the
fake real repo.

The module is loaded by file path (mirroring
``test_wrapper_mmseqs_search.py``'s own technique), since ``scripts/engines/``
is not a package on the pytest path.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "engines" / "la_proteina.py"
)


def _load_wrapper():
    spec = importlib.util.spec_from_file_location("la_proteina_wrapper", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


wrapper = _load_wrapper()


def _make_fake_repo(root: Path) -> Path:
    """A minimal stand-in for the real la-proteina checkout: a couple of
    ordinary top-level entries, plus a `proteinfoundation/` package with
    `generate.py` and one sibling submodule, plus the four directories
    `_build_repo_view` must deliberately skip (`configs`, `inference`,
    `tmp`, `tmp_ae`) so the test also proves they are left alone.
    """
    repo = root / "fake_la_proteina"
    repo.mkdir()
    (repo / "assets").mkdir()
    (repo / "assets" / "logo.png").write_text("not a real image")
    (repo / "README.md").write_text("hello")

    pf = repo / "proteinfoundation"
    pf.mkdir()
    (pf / "generate.py").write_text("# real generate.py\n")
    (pf / "proteina.py").write_text("# real proteina.py\n")
    datasets = pf / "datasets"
    datasets.mkdir()
    (datasets / "gen_dataset.py").write_text("# real gen_dataset.py\n")

    # Pre-existing, host-side artifacts of a PRIOR run -- must never be
    # symlinked into the mirror; this run's own configs/inference/tmp/
    # tmp_ae/lightning_logs must be fresh, real, writable directories
    # under scratch.
    (repo / "configs").mkdir()
    (repo / "configs" / "inference_base.yaml").write_text("defaults: []\n")
    (repo / "inference").mkdir()
    (repo / "tmp").mkdir()
    (repo / "tmp_ae").mkdir()
    # Regression fixture: a real host repo carries this from prior
    # interactive runs (PyTorch Lightning's own default CSVLogger writes
    # here, cwd-relative) -- symlinking it (as an ordinary read-only entry)
    # let Lightning resolve through to the real, read-only mount and fail
    # with `OSError: Read-only file system` the moment it tried to create a
    # new version subdirectory, confirmed live in-container.
    (repo / "lightning_logs").mkdir()
    (repo / "lightning_logs" / "version_5").mkdir()

    return repo


@pytest.fixture
def fake_repo(tmp_path, monkeypatch):
    repo = _make_fake_repo(tmp_path)
    monkeypatch.setattr(wrapper, "_REPO_ROOT", repo)
    return repo


def test_top_level_entries_are_symlinked_into_the_scratch_workdir(tmp_path, fake_repo):
    call_workdir = tmp_path / "scratch"
    call_workdir.mkdir()

    wrapper._build_repo_view(call_workdir)

    assert (call_workdir / "assets").is_symlink()
    assert (call_workdir / "assets").resolve() == (fake_repo / "assets").resolve()
    assert (call_workdir / "README.md").is_symlink()
    assert (call_workdir / "README.md").read_text() == "hello"


def test_configs_inference_tmp_tmp_ae_lightning_logs_are_not_created_by_the_view(
    tmp_path, fake_repo
):
    """These must stay absent here -- generate.py (or a library it calls,
    e.g. PyTorch Lightning's own CSVLogger for lightning_logs) creates each
    one fresh, as a REAL directory under the scratch workdir, when it
    runs."""
    call_workdir = tmp_path / "scratch"
    call_workdir.mkdir()

    wrapper._build_repo_view(call_workdir)

    for name in ("configs", "inference", "tmp", "tmp_ae", "lightning_logs"):
        assert not (call_workdir / name).exists(), f"{name} must not be pre-created"


def test_proteinfoundation_dir_is_real_not_a_symlink(tmp_path, fake_repo):
    call_workdir = tmp_path / "scratch"
    call_workdir.mkdir()

    wrapper._build_repo_view(call_workdir)

    pf_view = call_workdir / "proteinfoundation"
    assert pf_view.is_dir()
    assert not pf_view.is_symlink()


def test_generate_py_is_a_real_copy_not_a_symlink(tmp_path, fake_repo):
    call_workdir = tmp_path / "scratch"
    call_workdir.mkdir()

    wrapper._build_repo_view(call_workdir)

    generate_py = call_workdir / "proteinfoundation" / "generate.py"
    assert generate_py.is_file()
    assert not generate_py.is_symlink()
    assert generate_py.read_text() == "# real generate.py\n"


def test_other_proteinfoundation_entries_are_symlinked(tmp_path, fake_repo):
    call_workdir = tmp_path / "scratch"
    call_workdir.mkdir()

    wrapper._build_repo_view(call_workdir)

    proteina_view = call_workdir / "proteinfoundation" / "proteina.py"
    assert proteina_view.is_symlink()
    datasets_view = call_workdir / "proteinfoundation" / "datasets"
    assert datasets_view.is_symlink()
    assert (datasets_view / "gen_dataset.py").read_text() == "# real gen_dataset.py\n"


def test_hydra_style_realpath_resolution_lands_under_scratch_not_the_real_repo(
    tmp_path, fake_repo
):
    """The actual regression this whole fix is for: hydra's own
    ``compute_search_path_dir`` computes
    ``realpath(dirname(calling_file))`` from generate.py's own ``__file__``
    -- reproduced here exactly, against the mirrored generate.py, to prove
    it resolves to the WRITABLE scratch workdir and never back to the real
    (read-only, in the container) checkout. Before this fix, a plain
    symlinked mirror made this assertion fail: realpath() follows a
    symlink straight through to its target.
    """
    call_workdir = tmp_path / "scratch"
    call_workdir.mkdir()

    wrapper._build_repo_view(call_workdir)

    calling_file = str(call_workdir / "proteinfoundation" / "generate.py")
    abs_base_dir = os.path.realpath(os.path.dirname(calling_file))

    assert abs_base_dir == str((call_workdir / "proteinfoundation").resolve())
    assert abs_base_dir != str((fake_repo / "proteinfoundation").resolve())

    # And the exact join generate.py performs (config_path=
    # "../configs/<subdir>") must land under the scratch workdir's own
    # (real, writable) configs/ directory.
    resolved_config_dir = os.path.normpath(
        os.path.join(abs_base_dir, "../configs/_mcp_token")
    )
    assert resolved_config_dir == str(call_workdir / "configs" / "_mcp_token")
