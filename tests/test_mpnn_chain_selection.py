"""`run_mpnn` must be able to design one chain and leave the rest alone.

Every binder generator here writes a two-chain complex: the target and the
design. Without a way to say which chain to design, ProteinMPNN redesigns
everything in the file. Observed live (proteinmem-mcp run llm_probe_02): given
run_rfdiffusion2's output -- target 504 residues in chain A, an 80-residue
poly-alanine design in chain B -- `run_mpnn` returned a **585-residue**
sequence. That is the target redesigned along with the binder, and it is
neither one.

The harness's target-copy guard does not catch it either: the guard asks
whether the design equals or contains the target, and a *redesigned* target
does neither.

LigandMPNN exposes exactly the knobs needed (`--chains_to_design`,
`--fixed_residues`, `--redesigned_residues`), so this was a gap in what the
manifest chose to expose, not a limitation of the engine -- and it broke the
primary chain this whole server exists to support.
"""

from __future__ import annotations

import pytest

from protein_design_mcp.adapters.mpnn import build_args
from protein_design_mcp.app import manifest_dir
from protein_design_mcp.manifest.loader import load_manifests


@pytest.fixture(scope="module")
def manifest():
    return next(m for m in load_manifests(manifest_dir()) if m.name == "run_mpnn")


def _args(manifest, **overrides):
    params = {
        "backbone_pdb": "/tmp/x.pdb", "model_type": "soluble",
        "num_sequences": 2, "sampling_temp": 0.1, "seed": 1,
        "chains_to_design": None, "fixed_residues": None, "redesigned_residues": None,
    }
    params.update(overrides)
    return build_args(manifest, params)


def test_the_schema_offers_chain_selection(manifest):
    assert "chains_to_design" in manifest.schema


def test_the_schema_offers_residue_level_control(manifest):
    """A binder generator may hand back a complex where the interface residues
    matter more than the chain split."""
    assert "fixed_residues" in manifest.schema
    assert "redesigned_residues" in manifest.schema


def test_chain_selection_reaches_the_engine(manifest):
    args = _args(manifest, chains_to_design="B")
    assert args[args.index("--chains_to_design") + 1] == "B"


def test_several_chains_can_be_selected(manifest):
    args = _args(manifest, chains_to_design="A B")
    assert args[args.index("--chains_to_design") + 1] == "A B"


def test_fixed_residues_reach_the_engine(manifest):
    args = _args(manifest, fixed_residues="A12 A13 B2")
    assert args[args.index("--fixed_residues") + 1] == "A12 A13 B2"


def test_redesigned_residues_reach_the_engine(manifest):
    args = _args(manifest, redesigned_residues="B5 B6")
    assert args[args.index("--redesigned_residues") + 1] == "B5 B6"


@pytest.mark.parametrize("field", ["chains_to_design", "fixed_residues", "redesigned_residues"])
def test_an_unset_selector_is_not_passed_at_all(manifest, field):
    """Passing an empty string is not the same as omitting it: LigandMPNN reads
    an empty --chains_to_design as "design nothing named", and the run either
    errors or silently changes meaning. Omission must stay omission."""
    flag = f"--{field}"
    assert flag not in _args(manifest)


def test_the_docs_say_why_this_matters(manifest):
    """The parameter is useless if the caller does not know the default
    redesigns the target too."""
    text = f"{manifest.summary}\n{manifest.doc}\n" + "\n".join(
        v.get("description", "") for v in manifest.schema.values()
    )
    assert "chains_to_design" in text
    assert "every chain" in text.lower() or "all chains" in text.lower(), (
        "the docs must say what happens when no chain is named"
    )
