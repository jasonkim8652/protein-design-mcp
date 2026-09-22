import json
from pathlib import Path

import pytest

from protein_design_mcp.adapters.mpnn import build_args, parse_output
from protein_design_mcp.app import manifest_dir
from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.loader import load_manifests
from protein_design_mcp.validation import ToolInputError, validate_and_fill

# Verified against dauparas/LigandMPNN's ligandmpnn/run.py (PyPI ligandmpnn==0.1.2,
# ligandmpnn.egg-info/entry_points.txt confirms console_scripts map to
# ligandmpnn.run:main and the module itself has `if __name__ == "__main__": main()`,
# so `python -m ligandmpnn.run` is a real, working invocation).
#
# The FIRST record run.py writes to seqs/*.fa (the native/input sequence) is NOT
# a design and carries no `id=` field — its real header shape is
# ">{name}, T=..., seed=..., num_res=..., num_ligand_res=..., use_ligand_context=...,
# ligand_cutoff_distance=..., batch_size=..., number_of_batches=..., model_path=...".
# Every subsequent record IS a design and does carry `id=`, with header shape
# ">{name}, id=..., T=..., seed=..., overall_confidence=..., ligand_confidence=...,
# seq_rec=...". Both shapes are reproduced below; this adapter must drop the first
# (no `id=`) and keep the rest.
SAMPLE_FASTA = """\
>input, T=0.1, seed=37, num_res=34, num_ligand_res=0, use_ligand_context=False, ligand_cutoff_distance=8.0, batch_size=2, number_of_batches=1, model_path=/models/proteinmpnn_v_48_020.pt
MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ
>input, id=1, T=0.1, seed=37, overall_confidence=0.5310, ligand_confidence=0.0, seq_rec=0.8824
MKTAYIAKQRQLSFVKSHFSRQLEERLGLIEVQ
>input, id=2, T=0.1, seed=37, overall_confidence=0.5502, ligand_confidence=0.0, seq_rec=0.9118
MKTAYIARQRQLSFVKSHFSRQLEERLGLIEVQ
"""


def _manifest():
    return next(m for m in load_manifests(manifest_dir()) if m.name == "run_mpnn")


def test_manifest_is_sequence_design_and_declares_its_output():
    m = _manifest()
    assert m.category == "sequence_design"
    (out,) = m.outputs
    assert out.name == "designs_fasta"


def test_manifest_output_allows_multiple_fasta_files():
    """collect_outputs raises AmbiguousOutputError on >1 match unless
    multiple=True; an inverse-folding run can plausibly write more than one
    seqs/*.fa file, so this output must opt in."""
    (out,) = _manifest().outputs
    assert out.multiple is True


def test_model_type_enum_covers_the_three_variants():
    enum = _manifest().schema["model_type"]["enum"]
    assert set(enum) == {"protein", "soluble", "ligand"}


def test_build_args_maps_model_type_to_the_checkpoint_flag():
    args = build_args(
        _manifest(),
        {"backbone_pdb": "/tmp/bb.pdb", "model_type": "soluble",
         "num_sequences": 8, "sampling_temp": 0.1, "seed": 37},
    )
    assert "soluble_mpnn" in args
    assert "/tmp/bb.pdb" in args
    assert "8" in args


def test_build_args_always_passes_a_seed_for_reproducibility():
    args = build_args(
        _manifest(),
        {"backbone_pdb": "/tmp/bb.pdb", "model_type": "protein",
         "num_sequences": 4, "sampling_temp": 0.1, "seed": 99},
    )
    assert "--seed" in args
    assert "99" in args


def _completed_run_with_fasta(tmp_path: Path, text: str) -> CompletedRun:
    """Build a CompletedRun whose 'designs_fasta' output points at a real
    file on disk, the way the dispatcher actually populates run.outputs for
    a multiple=true output spec (see results.collect_outputs): a LIST of
    collected file paths, never stdout. Settled live in Task 7 — see the
    module docstring in adapters/mpnn.py for the evidence."""
    fasta_path = tmp_path / "seqs" / "backbone.fa"
    fasta_path.parent.mkdir(parents=True, exist_ok=True)
    fasta_path.write_text(text)
    return CompletedRun(
        returncode=0,
        stdout="CUDA not available... using CPU\nDesigning protein from this path: backbone.pdb\n",
        stderr="",
        workdir=tmp_path,
        outputs={"designs_fasta": [str(fasta_path)]},
    )


def test_parse_output_drops_the_native_input_sequence(tmp_path):
    """The first FASTA record is the input, not a design."""
    result = parse_output(_manifest(), _completed_run_with_fasta(tmp_path, SAMPLE_FASTA))
    assert result["num_designs"] == 2
    assert all(d["id"] is not None for d in result["designs"])
    assert "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ" not in [
        d["sequence"] for d in result["designs"]
    ]


def test_parse_output_reports_confidence_per_design(tmp_path):
    result = parse_output(_manifest(), _completed_run_with_fasta(tmp_path, SAMPLE_FASTA))
    assert result["designs"][0]["overall_confidence"] == pytest.approx(0.5310)


def test_parse_output_raises_when_only_the_input_record_is_present(tmp_path):
    only_input = SAMPLE_FASTA.split(">input, id=1")[0]
    with pytest.raises(ValueError, match="no designs"):
        parse_output(_manifest(), _completed_run_with_fasta(tmp_path, only_input))


def test_parse_output_reads_every_file_when_multiple_are_collected(tmp_path):
    """multiple: true means run.outputs['designs_fasta'] can hold more than
    one path; every file's records must be included."""
    first = tmp_path / "seqs" / "a.fa"
    second = tmp_path / "seqs" / "b.fa"
    first.parent.mkdir(parents=True, exist_ok=True)
    first.write_text(SAMPLE_FASTA)
    second.write_text(SAMPLE_FASTA.replace("id=1", "id=3").replace("id=2", "id=4"))
    run = CompletedRun(
        returncode=0, stdout="", stderr="", workdir=tmp_path,
        outputs={"designs_fasta": [str(first), str(second)]},
    )
    result = parse_output(_manifest(), run)
    assert result["num_designs"] == 4
    assert {d["id"] for d in result["designs"]} == {1, 2, 3, 4}


def test_parse_output_raises_when_designs_fasta_output_is_missing(tmp_path):
    """A dispatcher/manifest mismatch (e.g. collection silently producing an
    empty result) must fail loudly rather than returning an empty design
    list to the caller."""
    run = CompletedRun(
        returncode=0, stdout=SAMPLE_FASTA, stderr="", workdir=tmp_path, outputs={},
    )
    with pytest.raises(ValueError, match="designs_fasta"):
        parse_output(_manifest(), run)


def test_validation_rejects_an_unknown_model_type():
    with pytest.raises(ToolInputError, match="model_type"):
        validate_and_fill(_manifest(), {"backbone_pdb": "bb.pdb",
                                        "model_type": "rna"})


def test_validation_rejects_a_temperature_above_the_maximum():
    with pytest.raises(ToolInputError, match="sampling_temp"):
        validate_and_fill(_manifest(), {"backbone_pdb": "bb.pdb",
                                        "sampling_temp": 5.0})
