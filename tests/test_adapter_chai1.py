import json
from pathlib import Path

import numpy as np
import pytest

from protein_design_mcp.adapters.chai1 import build_args, parse_output
from protein_design_mcp.app import manifest_dir
from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.loader import load_manifests
from protein_design_mcp.validation import ToolInputError, validate_and_fill

MANIFEST_DIR = manifest_dir()

SEQ_A = "QLEDSEVEAVAKGLEEMYANGVTEDNFKNYVKNNFAQQEISSVEEELNVNISDSCVANKIKDEFFAMISISAIVKAAQKKAWKELAVTVLRFAKANGLKTNAIIVAGQLALWAVQCG"


def _manifest():
    return next(m for m in load_manifests(MANIFEST_DIR) if m.name == "run_chai1")


def _base_params(**overrides):
    return validate_and_fill(
        _manifest(), {"chains": [{"sequence": SEQ_A, "msa": None}], **overrides}
    )


def test_manifest_loads_and_is_gpu_structure_prediction():
    m = _manifest()
    assert m.category == "structure_prediction"
    assert m.composite is False
    assert m.requires.gpu is True


def test_manifest_discloses_no_pae_output():
    doc = _manifest().doc
    assert "run_ipsae" in doc
    assert "PAE" in doc


def test_manifest_uses_prefix_with_no_extra_mounts():
    engine = _manifest().engine
    assert engine.prefix == "/home/jk661/.conda/envs/chai1"
    assert engine.env is None
    assert engine.mounts == ()


def test_validation_rejects_empty_chains():
    with pytest.raises(ToolInputError, match="chains"):
        validate_and_fill(_manifest(), {"chains": []})


def test_validation_fills_defaults():
    params = _base_params()
    assert params["num_trunk_recycles"] == 3
    assert params["num_diffn_samples"] == 5
    assert params["use_esm_embeddings"] is True
    assert params["low_memory"] is True


def test_build_args_rejects_chain_missing_msa_key():
    params = _base_params()
    del params["chains"]
    params["chains"] = [{"sequence": SEQ_A}]
    with pytest.raises(ValueError, match="msa"):
        build_args(_manifest(), params)


def test_build_args_single_chain_msa_free():
    params = _base_params()
    args = build_args(_manifest(), params)
    job = json.loads(args[0])
    assert job["chains"] == [{"sequence": SEQ_A, "msa": None, "copies": 1}]


def test_build_args_resolves_relative_msa_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "seq.a3m").write_text(">query\n" + SEQ_A + "\n")
    params = _base_params()
    params["chains"] = [{"sequence": SEQ_A, "msa": "seq.a3m"}]
    args = build_args(_manifest(), params)
    job = json.loads(args[0])
    assert job["chains"][0]["msa"] == str(tmp_path / "seq.a3m")


def test_build_args_rejects_too_many_copies():
    params = _base_params()
    params["chains"] = [{"sequence": SEQ_A, "msa": None, "copies": 21}]
    with pytest.raises(ValueError, match="copies"):
        build_args(_manifest(), params)


# --- parse_output ---

SCORE_FIELDS_MODEL_0 = dict(
    aggregate_score=np.array(0.5),
    ptm=np.array(0.4),
    iptm=np.array(0.1),
    per_chain_ptm=np.array([0.4]),
    per_chain_pair_iptm=np.array([[0.1]]),
    has_inter_chain_clashes=np.array(False),
    chain_chain_clashes=np.array([[0]]),
)


def _run_with_outputs(tmp_path: Path, n: int) -> CompletedRun:
    structures, scores = [], []
    for i in range(n):
        struct = tmp_path / f"pred.model_idx_{i}.cif"
        struct.write_text("data_pred\n")
        structures.append(str(struct))
        score_path = tmp_path / f"scores.model_idx_{i}.npz"
        fields = dict(SCORE_FIELDS_MODEL_0)
        fields["ptm"] = np.array(0.1 * (i + 1))
        np.savez(score_path, **fields)
        scores.append(str(score_path))
    return CompletedRun(
        returncode=0, stdout="", stderr="", workdir=tmp_path,
        outputs={"structures": structures, "scores": scores},
    )


def test_parse_output_reads_rank0_scores(tmp_path):
    result = parse_output(_manifest(), _run_with_outputs(tmp_path, 1))
    assert result["ptm"] == pytest.approx(0.1)
    assert result["num_structures"] == 1
    assert "run_ipsae" in result["caveat"]


def test_parse_output_picks_rank0_among_many_by_numeric_suffix(tmp_path):
    result = parse_output(_manifest(), _run_with_outputs(tmp_path, 11))
    assert result["ptm"] == pytest.approx(0.1)


def test_parse_output_raises_when_scores_missing(tmp_path):
    run = CompletedRun(returncode=0, stdout="", stderr="", workdir=tmp_path, outputs={})
    with pytest.raises(ValueError, match="scores"):
        parse_output(_manifest(), run)
