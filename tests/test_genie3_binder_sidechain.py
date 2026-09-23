"""`run_genie3_binder` must not offer a side-chain stage it cannot reach.

Genie 3's side-chain pass is the second of two stages, and `workflow.py:202`
guards it with:

    assert config.dataset.source == "unconditional"

Binder generation runs with `source == "target"` (`runner.py:128`,
`postprocess.py:38`), so the assertion can never hold for this tool. Worse, it
sits *after* `[1 / 2] Main stage completed!` — the engine does the whole
generation, then throws the result away.

Observed live (proteinmem-mcp run llm_probe_02, round 2): the model passed
`predict_sidechain: true` because this manifest's own documentation said it
would get side-chain atoms, and lost 52 s of GPU to an AssertionError.

The manifest already has the right pattern for this: `predict_sequence` is not
exposed, and the doc says why under "What is NOT exposed, and why".
"""

from __future__ import annotations

import pytest

from protein_design_mcp.app import manifest_dir
from protein_design_mcp.manifest.loader import load_manifests


@pytest.fixture(scope="module")
def manifests():
    return {m.name: m for m in load_manifests(manifest_dir())}


def test_the_binder_tool_does_not_expose_predict_sidechain(manifests):
    """Offering it is offering a call that always fails, expensively."""
    schema = manifests["run_genie3_binder"].schema
    assert "predict_sidechain" not in schema, (
        "run_genie3_binder exposes predict_sidechain, but Genie 3 asserts "
        "dataset.source == 'unconditional' before the side-chain stage and "
        "binder generation runs with source == 'target'"
    )


def test_the_binder_docs_do_not_promise_side_chain_atoms(manifests):
    """The model followed the documentation into the failure, so the
    documentation is part of the defect."""
    m = manifests["run_genie3_binder"]
    text = f"{m.summary}\n{m.doc}"
    offending = [
        line.strip()
        for line in text.splitlines()
        if "predict_sidechain" in line and "not exposed" not in line.lower()
        and "NOT exposed" not in line
    ]
    assert not offending, (
        "run_genie3_binder still advertises predict_sidechain: " + " | ".join(offending)
    )


def test_the_binder_docs_say_why_it_is_absent(manifests):
    """A silently missing knob sends the next reader to add it back."""
    doc = manifests["run_genie3_binder"].doc
    assert "predict_sidechain" in doc, "the omission must be explained, not silent"
    assert "unconditional" in doc, "the doc must name the upstream constraint"


def test_the_scaffold_tool_keeps_it(manifests):
    """The constraint is specific to conditional generation. run_genie3_scaffold
    IS the unconditional path, so the side-chain stage is reachable there and
    removing the knob would lose a real capability."""
    assert "predict_sidechain" in manifests["run_genie3_scaffold"].schema
