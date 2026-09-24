"""Every live-proof case's arguments must satisfy its tool's schema.

`test_every_registered_tool_has_a_live_case` checks that a case *exists*. It
says nothing about whether the case is still *correct*, and those drift apart:
`run_boltzgen_filter`'s case was written in commit 8d83387 passing `design_dir`,
and commit f35079d later redesigned that tool's inputs to `generated_files` /
`metrics_files` / `refold_structures`. Nothing noticed. The case only failed
when the whole proof was finally executed inside the container, near the end of
a long piece of work — the most expensive moment to find it.

This check is static: it validates arguments against the manifest without
starting an engine, so drift surfaces in a normal test run rather than after a
GPU queue.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
LIVE_PROOF = REPO_ROOT / "scripts" / "live_proof.py"


def _load_live_proof():
    spec = importlib.util.spec_from_file_location("live_proof", LIVE_PROOF)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # Registered BEFORE exec: a module that defines a @dataclass has to be in
    # sys.modules while it runs, because dataclasses resolves annotations
    # through `sys.modules[cls.__module__].__dict__` and gets None otherwise --
    # "AttributeError: 'NoneType' object has no attribute '__dict__'", which
    # names neither the module nor the dataclass.
    sys.modules.setdefault("live_proof", module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cases():
    return _load_live_proof().CASES


@pytest.fixture(scope="module")
def manifests_by_name():
    from protein_design_mcp.app import build_registry
    from protein_design_mcp.meta_tools import DESCRIBE_TOOL_MANIFEST

    by_name = {}
    # Union of both devices: a case declares the device it needs, and a
    # GPU-only tool is absent from the cpu registry.
    for device in ("cpu", "cuda"):
        registry = build_registry(device=device)
        for manifest in registry.tools_as_manifests() if hasattr(registry, "tools_as_manifests") else []:
            by_name[manifest.name] = manifest
        for name in getattr(registry, "_available", {}):
            by_name[name] = registry._available[name]
    by_name[DESCRIBE_TOOL_MANIFEST.name] = DESCRIBE_TOOL_MANIFEST
    return by_name


def test_no_case_passes_a_parameter_its_tool_does_not_accept(cases, manifests_by_name):
    """The exact failure that reached the in-container run."""
    problems = []
    for case in cases:
        manifest = manifests_by_name.get(case["tool"])
        if manifest is None:
            # Meta-tools without a manifest (get_job_status) are covered by the
            # coverage test; nothing to validate here.
            continue
        accepted = set(manifest.schema)
        sent = set(case.get("arguments", {}))
        extra = sorted(sent - accepted)
        if extra:
            problems.append(
                f"{case['tool']}: case passes {extra}, "
                f"which the manifest does not accept (accepts: {sorted(accepted)})"
            )
    assert not problems, "live-proof cases have drifted from their manifests:\n" + "\n".join(problems)


def test_every_required_parameter_is_supplied(cases, manifests_by_name):
    """The other half of drift: a parameter becoming required after the case was
    written. Fails at dispatch, but only once the case actually runs."""
    problems = []
    for case in cases:
        manifest = manifests_by_name.get(case["tool"])
        if manifest is None:
            continue
        sent = set(case.get("arguments", {}))
        required = {
            name
            for name, spec in manifest.schema.items()
            if isinstance(spec, dict) and spec.get("required") is True
        }
        missing = sorted(required - sent)
        if missing:
            problems.append(f"{case['tool']}: case omits required {missing}")
    assert not problems, "live-proof cases omit required parameters:\n" + "\n".join(problems)


def test_the_validation_can_actually_see_schemas(cases, manifests_by_name):
    """Guards the fixture, not the cases. If manifest resolution silently
    returned nothing, both tests above would pass vacuously — a green result
    proving nothing, which this project has hit before."""
    matched = [c for c in cases if c["tool"] in manifests_by_name]
    assert len(matched) > 20, (
        f"only {len(matched)} of {len(cases)} cases resolved to a manifest; "
        "schema lookup is probably broken"
    )
