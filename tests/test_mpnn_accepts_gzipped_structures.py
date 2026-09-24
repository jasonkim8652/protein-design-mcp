"""`run_mpnn` must accept the structure files its upstream tools actually write.

`run_rfdiffusion3_binder` writes a GZIPPED CIF -- that is what its
`structure_cif` output is -- and run_mpnn's pattern was `\\.(pdb|cif)$`, so a
planned `run_rfdiffusion3_binder -> run_mpnn` workflow died on the handoff:

    run_mpnn.backbone_pdb = '.../input_job_0_model_0.cif.gz' does not match
    the required format \\.(pdb|cif)$

The restriction was ours. ProDy, which LigandMPNN parses with, reads the file
directly -- confirmed in the mpnn environment on that exact file: 4234 atoms
parsed, all 4234 selected as protein.

Found by executing a strategy a round had actually planned, which is the only
place a handoff between two working tools can fail.
"""

from __future__ import annotations

import pytest

from protein_design_mcp.app import manifest_dir
from protein_design_mcp.manifest.loader import load_manifests
from protein_design_mcp.validation import validate_and_fill


@pytest.fixture(scope="module")
def manifest():
    return next(m for m in load_manifests(manifest_dir()) if m.name == "run_mpnn")


@pytest.mark.parametrize("path", [
    "/tmp/backbone.pdb",
    "/tmp/model.cif",
    "/tmp/input_job_0_model_0.cif.gz",   # run_rfdiffusion3_binder
    "/tmp/design.pdb.gz",
])
def test_the_structure_formats_upstream_tools_write_are_accepted(manifest, path):
    assert validate_and_fill(manifest, {"backbone_pdb": path})["backbone_pdb"] == path


def test_an_unrelated_extension_is_still_refused(manifest):
    """Widened, not removed: a .txt reaches ProDy and fails inside it."""
    with pytest.raises(Exception):
        validate_and_fill(manifest, {"backbone_pdb": "/tmp/notes.txt"})


def test_a_bare_gz_is_refused(manifest):
    """`.gz` alone says nothing about what is inside it."""
    with pytest.raises(Exception):
        validate_and_fill(manifest, {"backbone_pdb": "/tmp/archive.gz"})


def test_the_docs_say_a_compressed_structure_is_fine(manifest):
    """A caller holding run_rfdiffusion3_binder's output needs to know it can
    be passed as-is rather than decompressed first."""
    text = f"{manifest.doc}\n{manifest.schema['backbone_pdb'].get('description', '')}"
    assert ".gz" in text
