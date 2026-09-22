import json
from pathlib import Path

import pytest

from protein_design_mcp.adapters.openfold3 import build_args, parse_output
from protein_design_mcp.app import manifest_dir
from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.loader import load_manifests
from protein_design_mcp.validation import ToolInputError, validate_and_fill

MANIFEST_DIR = manifest_dir()

SEQ = "NLYIQWLKDGGPSSGRPPPS"


def _manifest():
    return next(m for m in load_manifests(MANIFEST_DIR) if m.name == "run_openfold3")


def _base_params(**overrides):
    return validate_and_fill(
        _manifest(), {"chains": [{"sequence": SEQ, "msa": None}], **overrides}
    )


def test_manifest_loads_and_is_gpu_structure_prediction():
    m = _manifest()
    assert m.category == "structure_prediction"
    assert m.composite is False
    assert m.requires.gpu is True


def test_manifest_caps_diffusion_samples_at_5():
    schema = _manifest().schema
    assert schema["num_diffusion_samples"]["maximum"] == 5
    assert schema["num_diffusion_samples"]["default"] == 5


def test_manifest_documents_the_cap_reason():
    doc = _manifest().doc
    assert "OOM" in doc or "75.94" in doc or "out of memory" in doc.lower()


def test_manifest_uses_prefix_with_no_extra_mounts():
    engine = _manifest().engine
    assert engine.prefix == "/home/jk661/.conda/envs/openfold3"
    assert engine.mounts == ()


def test_validation_rejects_diffusion_samples_above_cap():
    with pytest.raises(ToolInputError, match="num_diffusion_samples"):
        validate_and_fill(
            _manifest(),
            {"chains": [{"sequence": SEQ, "msa": None}], "num_diffusion_samples": 6},
        )


def test_validation_rejects_empty_chains():
    with pytest.raises(ToolInputError, match="chains"):
        validate_and_fill(_manifest(), {"chains": []})


def test_validation_fills_defaults():
    params = _base_params()
    assert params["num_diffusion_samples"] == 5
    assert params["seeds"] == [42]


def test_build_args_rejects_chain_missing_msa_key():
    params = _base_params()
    params["chains"] = [{"sequence": SEQ}]
    with pytest.raises(ValueError, match="msa"):
        build_args(_manifest(), params)


def test_build_args_single_chain_msa_free():
    params = _base_params()
    args = build_args(_manifest(), params)
    job = json.loads(args[0])
    assert job["chains"] == [{"sequence": SEQ, "msa": None, "copies": 1}]
    assert job["seeds"] == [42]


def test_build_args_resolves_relative_msa_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "seq.a3m").write_text(">query\n" + SEQ + "\n")
    params = _base_params()
    params["chains"] = [{"sequence": SEQ, "msa": "seq.a3m"}]
    args = build_args(_manifest(), params)
    job = json.loads(args[0])
    assert job["chains"][0]["msa"] == str(tmp_path / "seq.a3m")


# --- parse_output ---

AGG = {
    "avg_plddt": 85.1, "gpde": 0.52, "iptm": 0.0, "ptm": 0.30,
    "has_clash": 0.0, "sample_ranking_score": 0.13,
    "chain_ptm": {"A": 0.30}, "chain_pair_iptm": {},
}


def _run_with_outputs(tmp_path: Path, seeds: list[int], samples: int) -> CompletedRun:
    structures, aggregated = [], []
    for seed in seeds:
        for i in range(1, samples + 1):
            struct = tmp_path / f"job_seed_{seed}_sample_{i}_model.cif"
            struct.write_text("data_job\n")
            structures.append(str(struct))
            agg = tmp_path / f"job_seed_{seed}_sample_{i}_confidences_aggregated.json"
            data = dict(AGG, ptm=0.1 * i + seed)
            agg.write_text(json.dumps(data))
            aggregated.append(str(agg))
    return CompletedRun(
        returncode=0, stdout="", stderr="", workdir=tmp_path,
        outputs={"structures": structures, "confidence_aggregated_json": aggregated},
    )


def test_parse_output_reads_lowest_seed_sample(tmp_path):
    run = _run_with_outputs(tmp_path, seeds=[42], samples=1)
    result = parse_output(_manifest(), run)
    assert result["ptm"] == pytest.approx(42.1)
    assert result["num_structures"] == 1
    assert result["num_diffusion_samples_cap"] == 5


def test_parse_output_picks_lowest_seed_among_many(tmp_path):
    run = _run_with_outputs(tmp_path, seeds=[42, 2], samples=1)
    result = parse_output(_manifest(), run)
    assert result["ptm"] == pytest.approx(2.1)


def test_parse_output_raises_when_confidence_missing(tmp_path):
    run = CompletedRun(returncode=0, stdout="", stderr="", workdir=tmp_path, outputs={})
    with pytest.raises(ValueError, match="confidence_aggregated_json"):
        parse_output(_manifest(), run)
