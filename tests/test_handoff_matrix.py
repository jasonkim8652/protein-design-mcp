"""The whole tool surface has no declared handoff that its schemas forbid.

Three broken handoffs were found one at a time, each costing a GPU run and each
only reachable because some model happened to plan that pair. All three were
decidable from the manifests alone -- a declared output's extension against the
declared pattern of the parameter it is meant to feed -- so this decides them
all at once, and fails the suite rather than a run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import handoff_matrix as matrix  # noqa: E402

from protein_design_mcp.app import manifest_dir  # noqa: E402
from protein_design_mcp.manifest.loader import load_manifests  # noqa: E402


@pytest.fixture(scope="module")
def manifests():
    return {m.name: m for m in load_manifests(manifest_dir())}


def test_no_declared_handoff_is_forbidden_by_the_consumer_s_own_schema(capsys):
    assert matrix.main([]) == 0, capsys.readouterr().out


def test_the_check_actually_covers_something():
    """A checker that examines nothing passes trivially. The surface has
    handoffs described in prose; this asserts they are being read."""
    matrix.main([])
    # main() prints the count; re-derive it here rather than parsing stdout.
    manifests = {m.name: m for m in load_manifests(manifest_dir())}
    pairs = 0
    for producer in manifests.values():
        for output in (producer.outputs or ()):
            if not matrix.output_suffixes(output.pattern):
                continue
            for name in matrix._TOOL_MENTION.findall(output.description or ""):
                if name in manifests and name != producer.name:
                    pairs += 1
    assert pairs >= 10, f"only {pairs} handoff mentions found; the check is near-vacuous"


@pytest.mark.parametrize("produced,old_pattern", [
    (".npz", r"\.json$"),          # run_boltz.pae_npz -> run_ipsae, before the fix
    (".cif.gz", r"\.(pdb|cif)$"),  # run_rfdiffusion3_binder -> run_mpnn, before the fix
])
def test_the_defects_this_was_written_for_would_have_been_caught(produced, old_pattern):
    """Both were found by running a workflow and watching it die. The point of
    a static check is that neither would have needed a GPU."""
    assert not ({produced} & matrix.suffixes_for(old_pattern, "p"))


def test_a_mention_that_disclaims_a_handoff_is_not_treated_as_one():
    """run_rfdiffusion2.metadata_trb names run_rfdiffusion_binder to say it has
    "the same role" and is "Not parsed by this tool". Reading that as a handoff
    reported a mismatch that does not exist."""
    assert matrix._NOT_A_HANDOFF.search("same role as run_rfdiffusion_binder's output")
    assert matrix._NOT_A_HANDOFF.search("Not parsed by this tool")
    assert not matrix._NOT_A_HANDOFF.search("this is what run_boltzgen_filter ranks on")


def test_array_valued_path_parameters_are_seen(manifests):
    """run_boltzgen_filter.metrics_files is an array of paths. A checker that
    looked only at scalars reported its producer as having nowhere to go."""
    params = matrix.path_parameters(manifests["run_boltzgen_filter"])
    assert "metrics_files" in params


def test_no_required_path_is_unobtainable_from_a_workflow(capsys):
    """A required path that no declared output can satisfy cannot be filled from
    a plan at all. `design_spec` is required by all six BoltzGen tools and
    produced by none of them -- the caller authors it -- and a model that
    planned run_boltzgen_fold after RFdiffusion3 failed on exactly that, with no
    way to know the parameter was not something an earlier step provides.

    Covered by the same run as the format check; this names the case so a new
    tool with an unobtainable prerequisite fails here rather than in a round.
    """
    assert matrix.main([]) == 0, capsys.readouterr().out


def test_a_caller_authored_parameter_is_exempt_once_it_says_so(manifests):
    """The exemption is the documentation, not a list of names: a parameter is
    allowed to be unobtainable precisely when it tells the caller to write it."""
    description = manifests["run_boltzgen_fold"].schema["design_spec"]["description"].lower()
    assert "you write this file" in description
    assert "no tool on this server produces" in description


def test_a_sentence_about_chain_order_is_not_read_as_a_handoff():
    """run_mpnn's designs_fasta names five generators to say they disagree
    about which chain holds the design. That is where a design LANDS, not
    where this file goes, and reading it as five handoffs reported four
    mismatches that do not exist."""
    sentence = ("run_genie3_binder and run_rfdiffusion3_binder write their "
                "design into chain A, run_rfdiffusion2 into chain B")
    assert matrix._NOT_A_HANDOFF.search(sentence)
    assert not matrix._NOT_A_HANDOFF.search(
        "Feed this straight into run_boltzgen_filter's metrics_files")
