import json
from pathlib import Path

import pytest

from protein_design_mcp.adapters.boltz import build_args, parse_output
from protein_design_mcp.app import manifest_dir
from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.loader import load_manifests
from protein_design_mcp.validation import ToolInputError, validate_and_fill

MANIFEST_DIR = manifest_dir()

SEQ_A = "QLEDSEVEAVAKGLEEMYANGVTEDNFKNYVKNNFAQQEISSVEEELNVNISDSCVANKIKDEFFAMISISAIVKAAQKKAWKELAVTVLRFAKANGLKTNAIIVAGQLALWAVQCG"


def _manifest():
    return next(m for m in load_manifests(MANIFEST_DIR) if m.name == "run_boltz")


def _base_params(**overrides):
    params = validate_and_fill(
        _manifest(),
        {"chains": [{"sequence": SEQ_A, "msa": None}], **overrides},
    )
    return params


def test_manifest_loads_and_is_gpu_structure_prediction():
    m = _manifest()
    assert m.category == "structure_prediction"
    assert m.composite is False
    assert m.requires.gpu is True


def test_manifest_documents_the_fork_and_affinity_exclusion():
    doc = _manifest().doc
    assert "fork" in doc.lower()
    assert "affinity head is not exposed" in doc


def test_manifest_uses_prefix_not_env():
    engine = _manifest().engine
    assert engine.prefix == "/home/jk661/.conda/envs/boltz"
    assert engine.env is None
    assert "/home/jk661/projects/lightning-boltz-dev/src" in engine.mounts


# --- validation (corner cases: 0, empty, single item, boundary) ---


def test_validation_rejects_empty_chains():
    with pytest.raises(ToolInputError, match="chains"):
        validate_and_fill(_manifest(), {"chains": []})


def test_validation_rejects_missing_chains():
    with pytest.raises(ToolInputError, match="chains"):
        validate_and_fill(_manifest(), {})


def test_validation_fills_defaults_for_single_chain():
    params = _base_params()
    assert params["recycling_steps"] == 3
    assert params["sampling_steps"] == 200
    assert params["diffusion_samples"] == 1
    assert params["subsample_msa"] is False


def test_validation_accepts_boundary_diffusion_samples():
    params = _base_params(diffusion_samples=25)
    assert params["diffusion_samples"] == 25
    with pytest.raises(ToolInputError):
        validate_and_fill(
            _manifest(),
            {"chains": [{"sequence": SEQ_A, "msa": None}], "diffusion_samples": 26},
        )


# --- build_args / chain normalization ---


def test_build_args_rejects_chain_missing_msa_key():
    params = {"chains": [{"sequence": SEQ_A}], "recycling_steps": 3,
              "sampling_steps": 200, "diffusion_samples": 1, "step_scale": 1.5,
              "use_potentials": False, "output_format": "mmcif",
              "max_msa_seqs": 8192, "subsample_msa": False,
              "num_subsampled_msa": 1024, "seed": 42}
    with pytest.raises(ValueError, match="msa"):
        build_args(_manifest(), params)


def test_build_args_rejects_bad_sequence_characters():
    params = _base_params()
    params["chains"] = [{"sequence": "MKT123", "msa": None}]
    with pytest.raises(ValueError, match="sequence"):
        build_args(_manifest(), params)


def test_build_args_rejects_zero_copies():
    params = _base_params()
    params["chains"] = [{"sequence": SEQ_A, "msa": None, "copies": 0}]
    with pytest.raises(ValueError, match="copies"):
        build_args(_manifest(), params)


def test_build_args_accepts_single_chain_msa_free():
    params = _base_params()
    args = build_args(_manifest(), params)
    assert len(args) == 1
    job = json.loads(args[0])
    assert job["chains"] == [{"sequence": SEQ_A, "msa": None, "copies": 1}]
    assert job["seed"] == 42


def test_build_args_resolves_relative_msa_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "seq.a3m").write_text(">query\n" + SEQ_A + "\n")
    params = _base_params()
    params["chains"] = [{"sequence": SEQ_A, "msa": "seq.a3m"}]
    args = build_args(_manifest(), params)
    job = json.loads(args[0])
    assert job["chains"][0]["msa"] == str(tmp_path / "seq.a3m")


def test_build_args_serializes_multi_chain_with_copies():
    params = _base_params()
    params["chains"] = [
        {"sequence": SEQ_A, "msa": None, "copies": 2},
        {"sequence": "MK", "msa": None},
    ]
    args = build_args(_manifest(), params)
    job = json.loads(args[0])
    assert job["chains"][0]["copies"] == 2
    assert job["chains"][1]["copies"] == 1


# --- parse_output ---

CONFIDENCE_MODEL_0 = {
    "confidence_score": 0.335, "ptm": 0.229, "iptm": 0.0,
    "protein_iptm": 0.0, "complex_plddt": 0.362, "complex_iplddt": 0.362,
    "complex_pde": 3.06, "complex_ipde": 0.0,
}


def _run_with_outputs(tmp_path: Path, confidence_count: int) -> CompletedRun:
    structures = []
    confidences = []
    for i in range(confidence_count):
        struct = tmp_path / f"job_model_{i}.cif"
        struct.write_text("data_job\n")
        structures.append(str(struct))
        conf = tmp_path / f"confidence_job_model_{i}.json"
        conf.write_text(json.dumps({**CONFIDENCE_MODEL_0, "ptm": 0.1 * (i + 1)}))
        confidences.append(str(conf))
    return CompletedRun(
        returncode=0, stdout="", stderr="", workdir=tmp_path,
        outputs={"structures": structures, "confidence_json": confidences},
    )


def test_parse_output_reads_rank0_confidence(tmp_path):
    run = _run_with_outputs(tmp_path, confidence_count=1)
    result = parse_output(_manifest(), run)
    assert result["ptm"] == pytest.approx(0.1)
    assert result["num_structures"] == 1
    assert "fork" in result["fork_notice"].lower()


def test_parse_output_picks_rank0_among_many_by_numeric_suffix(tmp_path):
    # 11 models so lexicographic sort ("_model_10" < "_model_2") would pick
    # the wrong one if the adapter didn't parse the numeric suffix.
    run = _run_with_outputs(tmp_path, confidence_count=11)
    result = parse_output(_manifest(), run)
    assert result["ptm"] == pytest.approx(0.1)  # model_0 -> 0.1 * (0+1)


def test_parse_output_raises_when_confidence_json_missing(tmp_path):
    run = CompletedRun(returncode=0, stdout="", stderr="", workdir=tmp_path, outputs={})
    with pytest.raises(ValueError, match="confidence_json"):
        parse_output(_manifest(), run)
