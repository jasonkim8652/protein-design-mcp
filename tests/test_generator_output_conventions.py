"""Every binder generator must say which chain is the design, and expose a
sequence when it generated one.

Both were found by reading real outputs, not source. Four generators run
against the same target produced four different arrangements:

    run_genie3_binder        binder = chain A   UNK, CA-only          no sequence
    run_rfdiffusion2         binder = chain B   poly-alanine          no sequence
    run_rfdiffusion3_binder  binder = chain A   18 distinct residues  SEQUENCE
    run_protpardelle         binder = chain B   poly-glycine          no sequence
    run_proteina_complexa    binder = chain B   17 distinct, all-atom SEQUENCE

A, B, A, B, B -- and the two that DO co-generate a sequence disagree with each
other about which chain it is in — and only run_genie3_binder documented it. A caller that learns
one tool's convention gets the next one exactly backwards, which is the same
class of mistake that made v1's `design_binder` return the target as its own
design. There the cause was a composite's hidden assumption; here it is an
undocumented output.

The sequence case matters for a second reason. RFdiffusion3 co-generates a
real sequence and buries it in a gzipped CIF that `outputs` does not mention,
so a caller cannot see that it exists. The predictable next move is to run
`run_mpnn` to get one -- discarding a sequence the model already produced.
"""

from __future__ import annotations

import pytest

from protein_design_mcp.app import manifest_dir
from protein_design_mcp.manifest.loader import load_manifests

#: Confirmed by reading each tool's real output for the same target, 2026-09-23.
#: `design_chain` is where the generated chain lands; `co_generates_sequence`
#: is whether that chain carries real residue identities rather than a
#: placeholder (UNK / poly-Ala / poly-Gly).
OBSERVED = {
    "run_genie3_binder": {"design_chain": "A", "co_generates_sequence": False},
    "run_rfdiffusion2": {"design_chain": "B", "co_generates_sequence": False},
    "run_rfdiffusion3_binder": {"design_chain": "A", "co_generates_sequence": True},
    "run_protpardelle": {"design_chain": "B", "co_generates_sequence": False},
    "run_proteina_complexa_generate": {"design_chain": "B", "co_generates_sequence": True},
}


@pytest.fixture(scope="module")
def manifests():
    return {m.name: m for m in load_manifests(manifest_dir())}


@pytest.mark.parametrize("name", sorted(OBSERVED))
def test_the_manifest_states_which_chain_holds_the_design(manifests, name):
    """Without this the caller guesses, and the guess is wrong half the time."""
    m = manifests[name]
    text = f"{m.summary}\n{m.doc}\n" + "\n".join(
        f"{o.name} {o.description}" for o in (m.outputs or ())
    )
    chain = OBSERVED[name]["design_chain"]
    assert "chain" in text.lower(), f"{name} never mentions chains at all"
    assert f"chain {chain}" in text, (
        f"{name} writes its design into chain {chain}, but says so nowhere a "
        "caller reads"
    )


@pytest.mark.parametrize("name", sorted(OBSERVED))
def test_a_generator_that_co_generates_a_sequence_says_so_in_its_docs(manifests, name):
    if not OBSERVED[name]["co_generates_sequence"]:
        pytest.skip(f"{name} emits a placeholder chain, so there is no sequence to expose")
    m = manifests[name]
    text = (f"{m.summary}\n{m.doc}\n" + "\n".join(
        o.description for o in (m.outputs or ()))).lower()
    assert "sequence" in text, f"{name} co-generates a sequence but never mentions one"


@pytest.mark.parametrize("name", sorted(OBSERVED))
def test_a_placeholder_generator_says_the_chain_carries_no_sequence(manifests, name):
    """The other half of the same question. A caller who assumes a design has a
    usable sequence will feed a poly-glycine trace to a folding tool and read
    the result as a prediction about their binder."""
    if OBSERVED[name]["co_generates_sequence"]:
        pytest.skip(f"{name} does co-generate a sequence")
    m = manifests[name]
    text = (f"{m.summary}\n{m.doc}\n" + "\n".join(
        f"{o.description}" for o in (m.outputs or ()))).lower()
    assert any(k in text for k in ("backbone-only", "backbone only", "no sequence",
                                   "poly-alanine", "poly-glycine", "placeholder")), (
        f"{name} emits a placeholder chain but never says so"
    )


def test_multiflow_still_exposes_its_codesign_sequences(manifests):
    """The one tool that already got this right; a regression here would be a
    capability silently lost."""
    outputs = {o.name for o in (manifests["run_multiflow"].outputs or ())}
    assert "codesign_sequences" in outputs
