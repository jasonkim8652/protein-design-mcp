"""`run_mpnn`'s ':' separator must be explained where the caller reads it.

Observed live: a round planned run_rfdiffusion3_binder -> run_mpnn -> run_boltz.
run_mpnn returned a two-chain design joined as `<design>:<target>`, and the
model handed it to run_boltz as ONE chain of 574 residues -- 70 designed plus
the 504-residue target, colon removed. Boltz then folded a fusion protein
rather than a complex, so there was no interface, and the run_ipsae step after
it would have had no chain pair to score.

Nothing refused it. ':' is not an amino acid the validator rejects, and it is
not a chain break a folding tool honours; the sequence is simply wrong in a way
that produces a plausible-looking structure. That is the shape of defect this
server documents its way out of, because no schema can catch it.
"""

from __future__ import annotations

import pytest

from protein_design_mcp.app import manifest_dir
from protein_design_mcp.manifest.loader import load_manifests


@pytest.fixture(scope="module")
def manifest():
    return next(m for m in load_manifests(manifest_dir()) if m.name == "run_mpnn")


def _text(manifest) -> str:
    return (f"{manifest.summary}\n{manifest.doc}\n"
            + "\n".join(o.description or "" for o in (manifest.outputs or ()))
            + "\n".join(str(s.get("description", "")) for s in manifest.schema.values()))


def test_the_separator_is_named(manifest):
    """A caller who does not know what ':' means will either keep it or strip
    it, and both are wrong."""
    text = _text(manifest)
    assert ":" in text
    assert "separat" in text.lower() or "chain break" in text.lower()


def test_the_docs_say_to_split_before_folding(manifest):
    """The fix is one sentence: each chain becomes its own entry in the folding
    tool's `chains` list."""
    text = _text(manifest).lower()
    assert "split" in text, "the caller must be told to split it"
    for hint in ("chains", "entry", "separate"):
        assert hint in text, hint


def test_the_docs_say_what_happens_if_you_do_not(manifest):
    """Telling someone to split without saying why leaves them free to decide
    it does not matter -- and the failure is silent."""
    text = _text(manifest).lower()
    assert "fus" in text or "one chain" in text or "single chain" in text


def test_the_docs_name_the_order_of_the_parts(manifest):
    """Which side of the ':' is the design depends on the generator's chain
    order, which differs across them -- A, B, A, B, B."""
    text = _text(manifest).lower()
    assert "order" in text or "chain a" in text
