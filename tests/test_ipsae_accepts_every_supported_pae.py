"""`run_ipsae` must accept every PAE format ipSAE itself supports.

The parameter was named `pae_json` and carried `pattern: \\.json$`, so a call
with Boltz's PAE was refused before the engine saw it:

    run_ipsae.pae_json = '.../pae_job_model_0.npz' does not match the
    required format \\.json$

ipSAE's own CLI help lists that exact combination as supported:

    ipsae alphafold2_multimer.json  alphafold2_model.pdb  15 15
    ipsae full_data_0.json          model_0.cif           10 10
    ipsae pae_model_0.npz           Boltz_model.cif       10 10

and `core.py` branches on `.pdb`, `.cif`+`.json` and `.cif`+`.npz`. The
restriction was ours, not the engine's -- and `run_boltz`'s own output
description says its `pae_npz` is "for run_ipsae", so the manifests documented
a chain the schema forbade.

A parameter named `pae_json` that accepts `.npz` would be a worse fix than the
bug: the name is what a caller reads first. It is `pae_file` now.
"""

from __future__ import annotations

import pytest

from protein_design_mcp.app import manifest_dir
from protein_design_mcp.manifest.loader import load_manifests
from protein_design_mcp.validation import validate_and_fill


@pytest.fixture(scope="module")
def manifest():
    return next(m for m in load_manifests(manifest_dir()) if m.name == "run_ipsae")


def _call(manifest, pae, structure="/tmp/model.cif"):
    return validate_and_fill(manifest, {"pae_file": pae, "structure": structure})


def test_the_parameter_is_no_longer_called_pae_json(manifest):
    assert "pae_json" not in manifest.schema, "the name promised JSON only"
    assert "pae_file" in manifest.schema


@pytest.mark.parametrize("pae", [
    "/tmp/alphafold2_multimer.json",   # AlphaFold2-Multimer
    "/tmp/full_data_0.json",           # Chai-1
    "/tmp/pae_job_model_0.npz",        # Boltz
])
def test_every_format_ipsae_documents_is_accepted(manifest, pae):
    assert _call(manifest, pae)["pae_file"] == pae


def test_an_unsupported_extension_is_still_refused(manifest):
    """Widened, not removed: a .txt here reaches the engine and fails deep
    inside it with 'Unsupported file combination'."""
    with pytest.raises(Exception):
        _call(manifest, "/tmp/pae.txt")


def test_the_docs_name_which_predictor_produces_which_format(manifest):
    """The formats differ per predictor, so a caller who knows only "a PAE"
    cannot tell which file to pass."""
    text = f"{manifest.doc}\n{manifest.schema['pae_file'].get('description', '')}"
    assert ".npz" in text and ".json" in text
    for engine in ("run_boltz", "run_alphafold2_multimer", "run_alphafold3"):
        assert engine in text, f"{engine} is not named as a source of a PAE file"


def test_the_docs_name_the_predictors_that_do_not_work(manifest):
    """run_protenix and run_chai1 both produce a PAE, so a caller has every
    reason to expect them to work here. They do not: ipSAE reads `pae` and
    `atom_plddts` from a .cif+.json pair, Protenix writes `pae`/`plddt` and
    Chai-1 writes `token_pair_pae`/`atom_plddt`, and the mismatch surfaces as
    a KeyError inside the engine rather than a refusal."""
    text = f"{manifest.doc}\n{manifest.schema['pae_file'].get('description', '')}"
    for engine in ("run_protenix", "run_chai1"):
        assert engine in text, f"{engine} produces a PAE this tool cannot read; say so"


def test_the_docs_warn_that_an_npz_needs_its_confidence_file_beside_it(manifest):
    """ipSAE derives the summary path from the PAE path -- it replaces "pae"
    with "confidence" and ".npz" with ".json" -- so a PAE moved away from its
    sibling fails on a file the caller never named."""
    text = f"{manifest.doc}\n{manifest.schema['pae_file'].get('description', '')}"
    assert "confidence" in text.lower()
