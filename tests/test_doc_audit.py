"""The prose every manifest shows a model agrees with the surface it describes.

Each class below was a real defect found by a run, and each was decidable from
the manifests alone:

    26 "(not yet implemented)" notes about tools that exist
    a `pae_json` left behind when the parameter became `pae_file`
    `run_boltzgen_design_fold` named after it was merged away
    /home/jk661/... in text a model is invited to copy into an argument
    design_spec required, produced by nothing, and saying so nowhere

The reader is a model choosing between tools and filling their arguments, so a
name that no longer exists is worse than an omission: it sends the model to ask
for something and spend a step on the refusal.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import doc_audit  # noqa: E402

from protein_design_mcp.app import manifest_dir  # noqa: E402
from protein_design_mcp.manifest.loader import load_manifests  # noqa: E402


@pytest.fixture(scope="module")
def manifests():
    return {m.name: m for m in load_manifests(manifest_dir())}


def test_no_manifest_names_a_tool_that_does_not_exist(manifests):
    assert doc_audit.check_names(manifests) == []


def test_no_manifest_claims_a_parameter_it_does_not_have(manifests):
    assert doc_audit.check_parameters(manifests) == []


def test_no_model_facing_text_carries_a_machine_specific_path(manifests):
    """engine.mounts legitimately holds real absolute paths -- the dispatcher
    needs them. What must not appear is a path in a DESCRIPTION, because the
    model will copy it into an argument and the file exists on one host."""
    assert doc_audit.check_host_paths(manifests) == []


def test_every_required_path_says_where_a_caller_gets_one(manifests):
    """design_spec was required by six tools, produced by none, and said so
    nowhere -- a round planned one of them and failed on a file no earlier
    step could have written."""
    assert doc_audit.check_provenance(manifests) == []


def test_the_audit_reads_the_text_a_model_actually_sees(manifests):
    """A checker looking at the wrong field passes while the model is misled.
    summary reaches it through tools/list, doc through describe_tool, and both
    parameter and output descriptions through the schema."""
    m = manifests["run_ipsae"]
    text = doc_audit.manifest_text(m)
    assert m.summary.split()[0] in text, "the summary must be included"
    # Parameter DESCRIPTIONS are included; the parameter NAMES are not, which
    # is right -- the names come from the schema and are checked against it.
    assert "Distance cutoff" in text, "parameter descriptions must be included"
    assert "ipSAE's per-chain-pair results" in text, "output descriptions too"


def test_the_checks_are_not_vacuous(manifests):
    """Each returns [] here; that has to mean "nothing wrong", not "nothing
    examined". Feed each one text that should trip it."""
    assert doc_audit.HOST_PATH.search("see /home/someone/weights")
    assert doc_audit.TOOL_NAME.findall("call run_nonexistent_tool next")
    assert doc_audit.SELF_SUPPLIED.search("your own structure file")
    assert not doc_audit.SELF_SUPPLIED.search("a path a previous step returned")


def test_every_parameter_and_output_has_a_usable_description(manifests):
    """This is the text the model reads to fill an argument: it arrives as the
    inputSchema description over MCP, and proteinmem-mcp's validator reads the
    same field before a call. A parameter without one gives a name, a type, and
    nothing about what moving it does.

    Weak on purpose -- it checks that text exists and is more than a
    restatement of the name. Whether a description says what changes when the
    parameter moves, and over what range, stays a reader's judgement; a missing
    one, or "Random seed.", is decidable here."""
    assert doc_audit.check_descriptions(manifests) == []


def test_the_description_check_would_catch_a_stub(manifests):
    """It returns [] above; that has to mean "all present", not "not looking".
    Three real stubs tripped it on the first run -- two `Random seed.` and one
    `Minimisation step limit.`"""
    from dataclasses import replace

    stubbed = replace(manifests["run_openmm_minimize"],
                      schema={"max_iterations": {"type": "integer",
                                                 "description": "Step limit."}})
    assert doc_audit.check_descriptions({"x": stubbed})


def test_the_description_check_accepts_a_real_one(manifests):
    """And it must not simply fail everything: the rewritten ones pass."""
    real = manifests["run_openmm_minimize"].schema["max_iterations"]["description"]
    assert len(real.split()) > 10
