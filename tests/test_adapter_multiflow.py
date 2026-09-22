from pathlib import Path

import pytest

from protein_design_mcp.adapters.multiflow import build_args, parse_output
from protein_design_mcp.app import manifest_dir
from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.loader import load_manifests
from protein_design_mcp.validation import ToolInputError, validate_and_fill

MINIMAL_PDB = (
    "ATOM      1  N   GLY A   1      11.104  13.207   2.145  1.00  0.00           N\n"
    "ATOM      2  CA  GLY A   1      12.560  13.207   2.145  1.00  0.00           C\n"
    "ATOM      3  C   GLY A   1      13.090  14.630   2.145  1.00  0.00           C\n"
    "TER\n"
)

DEFAULT_PARAMS = {
    "min_length": 70,
    "max_length": 70,
    "length_step": 10,
    "samples_per_length": 1,
    "num_timesteps": 500,
    "do_sde": False,
    "min_t": 0.01,
    "self_condition": True,
    "trans_sample_temp": 1.0,
    "aatypes_temp": 0.1,
    "aatypes_noise": 20.0,
    "aatypes_do_purity": True,
    "seed": 123,
}


def _manifest():
    return next(m for m in load_manifests(manifest_dir()) if m.name == "run_multiflow")


def test_manifest_loads_and_requires_gpu():
    m = _manifest()
    assert m.category == "monomer_generation"
    assert m.requires.gpu is True


def test_build_args_nulls_length_subset():
    args = build_args(_manifest(), DEFAULT_PARAMS)
    assert "inference.samples.length_subset=null" in args


def test_build_args_fixes_checkpoint_to_gpu0_variant():
    args = build_args(_manifest(), DEFAULT_PARAMS)
    ckpt_arg = next(a for a in args if a.startswith("inference.unconditional_ckpt_path="))
    assert ckpt_arg.endswith("last_gpu0.ckpt")


def test_build_args_forces_also_fold_pmpnn_seq_and_trajectories_off():
    args = build_args(_manifest(), DEFAULT_PARAMS)
    assert "inference.also_fold_pmpnn_seq=false" in args
    assert "inference.write_sample_trajectories=false" in args


def test_build_args_includes_aatypes_knobs():
    args = build_args(_manifest(), DEFAULT_PARAMS)
    assert "inference.interpolant.aatypes.temp=0.1" in args
    assert "inference.interpolant.aatypes.noise=20.0" in args
    assert "inference.interpolant.aatypes.do_purity=true" in args


def test_parse_output_pairs_backbone_with_codesign_sequence(tmp_path: Path):
    pdb_path = tmp_path / "backbones" / "predict_out" / "ckpt" / "unconditional" / "run_x" / "length_70" / "sample_0" / "sample.pdb"
    pdb_path.parent.mkdir(parents=True)
    pdb_path.write_text(MINIMAL_PDB)

    fasta_path = (
        tmp_path / "codesign_sequences" / "predict_out" / "ckpt" / "unconditional" / "run_x"
        / "length_70" / "sample_0" / "self_consistency" / "codesign_seqs" / "codesign.fa"
    )
    fasta_path.parent.mkdir(parents=True)
    fasta_path.write_text(">codesign\nMKTAYIAK\n")

    run = CompletedRun(
        returncode=0, stdout="", stderr="", workdir=tmp_path,
        outputs={"backbones": [str(pdb_path)], "codesign_sequences": [str(fasta_path)]},
    )
    result = parse_output(_manifest(), run)
    assert result["num_samples"] == 1
    sample = result["samples"][0]
    assert sample["length"] == 1
    assert sample["codesign_sequence"] == "MKTAYIAK"
    assert sample["id"] == "length_70/sample_0"


def test_parse_output_handles_missing_codesign_sequence_gracefully(tmp_path: Path):
    pdb_path = tmp_path / "backbones" / "predict_out" / "length_70" / "sample_0" / "sample.pdb"
    pdb_path.parent.mkdir(parents=True)
    pdb_path.write_text(MINIMAL_PDB)

    run = CompletedRun(
        returncode=0, stdout="", stderr="", workdir=tmp_path,
        outputs={"backbones": [str(pdb_path)], "codesign_sequences": []},
    )
    result = parse_output(_manifest(), run)
    assert result["samples"][0]["codesign_sequence"] is None


def test_parse_output_raises_when_no_backbones_collected():
    run = CompletedRun(returncode=0, stdout="", stderr="", workdir=Path("/tmp"), outputs={})
    with pytest.raises(ValueError, match="backbones"):
        parse_output(_manifest(), run)


def test_validation_requires_min_and_max_length():
    with pytest.raises(ToolInputError, match="min_length"):
        validate_and_fill(_manifest(), {"max_length": 70})


def test_validation_fills_defaults():
    params = validate_and_fill(_manifest(), {"min_length": 70, "max_length": 70})
    assert params["num_timesteps"] == 500
    assert params["aatypes_temp"] == pytest.approx(0.1)
    assert params["do_sde"] is False


def test_validation_allows_do_sde_true_not_treated_as_default():
    params = validate_and_fill(
        _manifest(), {"min_length": 70, "max_length": 70, "do_sde": True}
    )
    assert params["do_sde"] is True
