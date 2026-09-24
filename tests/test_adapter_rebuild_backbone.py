"""run_rebuild_backbone: make a CA trace designable.

Genie 3 represents every residue it GENERATES as a single CA token --
``feat_utils.py`` takes ``residue.gt_atom_mask[1:2]`` (atom37 index 1 is CA)
for a non-atomized residue, and only a conditioned, atomized residue gets its
full atom set. The "all-atom" in the model's name is about what it conditions
on, not what it generates, so no flag changes this.

Two things then stop ``run_mpnn`` consuming that output, and both had to be
fixed for the handoff to work at all (verified live, 2026-09-24):

1. ProteinMPNN needs N, CA, C and O to build each residue's frame. PULCHRA
   reconstructs them from the CA trace.
2. LigandMPNN parses with ProDy and calls ``atoms.select("protein")``, which
   is matched by RESIDUE NAME. ``UNK`` is not in that set, so the selection
   returns None and the run dies on ``'NoneType' object has no attribute
   'select'``. PULCHRA keeps the ``UNK`` names, so they have to be renamed.

Renaming to GLY loses nothing: the chain is about to be redesigned, and
ProteinMPNN derives a virtual CB from N/CA/C rather than reading the residue
it was given. GLY is the conventional name for a backbone with no side chain.

A survey of every structure this server has written on this host (143 result
directories) found CA-only chains from exactly two tools -- run_genie3_binder
and run_genie3_scaffold. Every other generator already emits a full backbone.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from protein_design_mcp.adapters.rebuild_backbone import (
    build_args, normalise_designable_residues, parse_output)
from protein_design_mcp.validation import ToolInputError


def _ca(serial, resname, chain, resseq, x=0.0):
    return (f"ATOM  {serial:5d}  CA  {resname} {chain}{resseq:4d}    "
            f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           C")


@pytest.fixture
def ca_trace(tmp_path):
    """Four UNK residues, CA only -- the shape Genie 3 emits."""
    path = tmp_path / "trace.pdb"
    path.write_text("\n".join(
        _ca(i, "UNK", "A", i, x=3.8 * i) for i in range(1, 5)) + "\n")
    return path


# --- the rename ---------------------------------------------------------------


def test_unk_becomes_gly_so_prody_sees_a_protein():
    line = _ca(1, "UNK", "A", 1)
    assert normalise_designable_residues([line])[0][17:20] == "GLY"


def test_a_real_residue_is_left_alone():
    """The target chain travels in the same file and must not be rewritten --
    its identity is what `run_mpnn` holds fixed."""
    line = _ca(1, "GLU", "B", 1)
    assert normalise_designable_residues([line])[0][17:20] == "GLU"


def test_only_the_residue_name_columns_change():
    line = _ca(7, "UNK", "A", 3, x=1.25)
    out = normalise_designable_residues([line])[0]
    assert out[:17] == line[:17]
    assert out[20:] == line[20:]


def test_a_non_atom_line_passes_through_untouched():
    assert normalise_designable_residues(["REMARK UNK anything"]) == [
        "REMARK UNK anything"]


def test_an_empty_file_is_not_an_error():
    assert normalise_designable_residues([]) == []


# --- the precondition ---------------------------------------------------------


def test_a_structure_that_already_has_a_backbone_is_refused(tmp_path):
    """Running PULCHRA over a real backbone would replace measured atoms with
    reconstructed ones -- silently worse input, and no error to notice. This
    tool is for CA traces."""
    path = tmp_path / "full.pdb"
    path.write_text(textwrap.dedent("""\
        ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00  0.00           N
        ATOM      2  CA  GLY A   1       1.000   0.000   0.000  1.00  0.00           C
        ATOM      3  C   GLY A   1       2.000   0.000   0.000  1.00  0.00           C
        ATOM      4  O   GLY A   1       3.000   0.000   0.000  1.00  0.00           O
        """))
    with pytest.raises(ToolInputError) as excinfo:
        build_args(None, {"structure": str(path)})
    assert "already" in str(excinfo.value).lower()


def test_a_ca_trace_is_accepted(ca_trace):
    args = build_args(None, {"structure": str(ca_trace)})
    assert str(ca_trace) in args


def test_a_mixed_file_with_one_ca_only_chain_is_accepted(tmp_path):
    """The real case: Genie 3 writes its CA-only design in chain A beside the
    supplied target, which has side chains, in chain B."""
    path = tmp_path / "mixed.pdb"
    path.write_text(
        _ca(1, "UNK", "A", 1) + "\n"
        + "ATOM      2  N   GLU B   1       0.000   0.000   0.000  1.00  0.00           N\n"
        + "ATOM      3  CA  GLU B   1       1.000   0.000   0.000  1.00  0.00           C\n"
        + "ATOM      4  C   GLU B   1       2.000   0.000   0.000  1.00  0.00           C\n"
        + "ATOM      5  O   GLU B   1       3.000   0.000   0.000  1.00  0.00           O\n")
    assert str(path) in build_args(None, {"structure": str(path)})


def test_a_missing_structure_is_refused_clearly(tmp_path):
    with pytest.raises(ToolInputError):
        build_args(None, {"structure": str(tmp_path / "nope.pdb")})


# --- the result ----------------------------------------------------------------


class _Run:
    def __init__(self, workdir):
        self.workdir = Path(workdir)
        self.stdout = ""
        self.stderr = ""
        self.outputs = {}


def test_the_result_reports_what_was_rebuilt(tmp_path):
    out = tmp_path / "rebuilt.pdb"
    out.write_text(textwrap.dedent("""\
        ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00  0.00           N
        ATOM      2  CA  GLY A   1       1.000   0.000   0.000  1.00  0.00           C
        ATOM      3  C   GLY A   1       2.000   0.000   0.000  1.00  0.00           C
        ATOM      4  O   GLY A   1       3.000   0.000   0.000  1.00  0.00           O
        """))
    run = _Run(tmp_path)
    run.outputs = {"structure_pdb": [str(out)]}
    result = parse_output(None, run)
    assert result["chains"] == ["A"]
    assert result["residues_rebuilt"] == 1
    assert result["backbone_complete"] is True


def test_an_output_still_missing_backbone_atoms_is_reported_not_hidden(tmp_path):
    """If PULCHRA silently produced a partial backbone, the next tool would
    die on it instead. Say so here, where the cause is still visible."""
    out = tmp_path / "partial.pdb"
    out.write_text(_ca(1, "GLY", "A", 1) + "\n")
    run = _Run(tmp_path)
    run.outputs = {"structure_pdb": [str(out)]}
    result = parse_output(None, run)
    assert result["backbone_complete"] is False


# --- the wrapper's copy must not drift from the adapter's -------------------


def _wrapper_module():
    """Load the engine wrapper without importing the server package.

    It deliberately carries its own copy of the rename: it runs in the `mpnn`
    environment, which has PULCHRA and LigandMPNN but not `mcp`, so importing
    `protein_design_mcp.adapters.rebuild_backbone` killed the engine on the
    import line before PULCHRA ran (seen live through the real MCP path).
    Two copies of eight lines is the cheap half of that trade; this test is
    the other half.
    """
    import importlib.util

    path = Path(__file__).parent.parent / "scripts" / "engines" / "rebuild_backbone.py"
    spec = importlib.util.spec_from_file_location("rebuild_backbone_engine", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("resname", ["UNK", "GLX", "XAA", "GLY", "GLU", "TRP"])
def test_both_copies_of_the_rename_agree(resname):
    wrapper = _wrapper_module()
    line = _ca(1, resname, "A", 1)
    assert wrapper.normalise_designable_residues([line]) == \
        normalise_designable_residues([line])


def test_both_copies_share_the_same_placeholder_set():
    from protein_design_mcp.adapters import rebuild_backbone as adapter

    assert _wrapper_module().PLACEHOLDER_RESNAMES == adapter.PLACEHOLDER_RESNAMES


def test_the_wrapper_does_not_import_the_server_package():
    """The regression itself: any `protein_design_mcp` import here is fatal in
    the environment this file actually runs in."""
    path = Path(__file__).parent.parent / "scripts" / "engines" / "rebuild_backbone.py"
    source = path.read_text()
    assert "import protein_design_mcp" not in source
    assert "from protein_design_mcp" not in source


def test_a_single_output_arrives_as_a_string_not_a_list(tmp_path):
    """`outputs` gives a list only for `multiple: true`. This manifest
    declares one file, so the dispatcher hands over a plain string --
    indexing it with [0] took the character '/' and the adapter failed with
    `Is a directory: '/'` (seen live through the real MCP path)."""
    out = tmp_path / "rebuilt.pdb"
    out.write_text(textwrap.dedent("""\
        ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00  0.00           N
        ATOM      2  CA  GLY A   1       1.000   0.000   0.000  1.00  0.00           C
        ATOM      3  C   GLY A   1       2.000   0.000   0.000  1.00  0.00           C
        ATOM      4  O   GLY A   1       3.000   0.000   0.000  1.00  0.00           O
        """))
    run = _Run(tmp_path)
    run.outputs = {"structure_pdb": str(out)}
    assert parse_output(None, run)["structure_pdb"] == str(out)


def test_no_output_at_all_is_refused_clearly(tmp_path):
    run = _Run(tmp_path)
    run.outputs = {}
    with pytest.raises(ToolInputError):
        parse_output(None, run)
