"""A manifest's own parameter is not a cross-tool reference.

The doc cross-reference rule exists so a tool's documentation cannot send a
caller to something that does not exist. It matches any ``run_*`` token in
``summary`` or ``doc``, which collides with a perfectly ordinary parameter
naming convention: BoltzGen has ``run_clustering``, AlphaFold 3 has
``run_data_pipeline`` and ``run_inference``.

Two waves hit this and worked around it by rephrasing correct prose. That is the
wrong repair — the checker was wrong, not the prose.

The exemption is deliberately narrow: only the manifest's **own** schema keys.
A sibling tool's parameter name mentioned here would still read ambiguously to a
caller, so it stays flagged.
"""

from __future__ import annotations

import pytest

from protein_design_mcp.manifest.loader import load_manifests
from protein_design_mcp.manifest.schema import ManifestError

BASE = {
    "name": "run_example",
    "category": "scoring",
    "summary": "Score something, briefly and clearly, for the caller.",
    "engine": {"repo": "example", "env": "scoring", "entry": ["python", "x.py"]},
    "schema": {},
    "doc": "## What it does\n\nIt scores a thing.\n",
}


def _write(tmp_path, **overrides):
    import yaml

    data = {**BASE, **overrides}
    (tmp_path / f"{data['name']}.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    return tmp_path


def test_a_run_prefixed_parameter_may_be_named_in_the_doc(tmp_path):
    """The case that forced the workarounds."""
    _write(
        tmp_path,
        schema={
            "run_clustering": {
                "type": "boolean",
                "default": False,
                "description": "Cluster the designs by structural similarity.",
            }
        },
        doc="## What it does\n\nTurning on run_clustering adds a cluster column.\n",
    )
    names = [m.name for m in load_manifests(tmp_path)]
    assert names == ["run_example"]


def test_an_unknown_run_token_that_is_not_a_parameter_is_still_rejected(tmp_path):
    """The exemption must not become a hole. This is the behaviour the rule
    exists for: a doc pointing at a tool that is not there."""
    _write(tmp_path, doc="## What it does\n\nAfterwards call run_nonexistent_tool.\n")
    with pytest.raises(ManifestError, match="run_nonexistent_tool"):
        load_manifests(tmp_path)


def test_the_exemption_does_not_extend_to_another_manifests_parameter(tmp_path):
    """Scoped to the manifest's OWN schema. A name that is a parameter somewhere
    else is still ambiguous to whoever reads this tool's documentation."""
    import yaml

    other = {
        **BASE,
        "name": "run_other",
        "schema": {
            "run_clustering": {
                "type": "boolean",
                "default": False,
                "description": "Cluster the designs by structural similarity.",
            }
        },
    }
    # Two tools share a category here, so both docs need the sibling-selection
    # heading — otherwise that rule fires first and this test would pass for
    # the wrong reason.
    sibling = "## What it does\n\nIt scores.\n\n## When to use this instead of the alternatives\n\nUse the other one for clustering.\n"
    other["doc"] = sibling
    (tmp_path / "run_other.yaml").write_text(yaml.safe_dump(other, sort_keys=False))
    _write(
        tmp_path,
        doc=(
            "## What it does\n\nIt scores.\n\n"
            "## When to use this instead of the alternatives\n\n"
            "See run_clustering for details.\n"
        ),
    )
    with pytest.raises(ManifestError, match="run_clustering"):
        load_manifests(tmp_path)
