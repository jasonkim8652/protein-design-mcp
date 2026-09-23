import json
from pathlib import Path

import pytest

from protein_design_mcp.adapters.protenix import build_args, parse_output
from protein_design_mcp.app import manifest_dir
from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.loader import load_manifests
from protein_design_mcp.validation import ToolInputError, validate_and_fill

MANIFEST_DIR = manifest_dir()

SEQ = "NLYIQWLKDGGPSSGRPPPS"


def _manifest():
    return next(m for m in load_manifests(MANIFEST_DIR) if m.name == "run_protenix")


def _base_params(**overrides):
    return validate_and_fill(
        _manifest(), {"chains": [{"sequence": SEQ, "msa": None}], **overrides}
    )


def test_manifest_loads_and_is_gpu_structure_prediction():
    m = _manifest()
    assert m.category == "structure_prediction"
    assert m.composite is False
    assert m.requires.gpu is True


def test_manifest_pins_the_v1_model_and_warns_about_v2():
    doc = _manifest().doc
    assert "protenix_base_default_v1.0.0" in doc
    assert "v2" in doc


def test_manifest_documents_the_remote_msa_service_workaround():
    doc = _manifest().doc
    assert "protenix-server.com" in doc


def test_manifest_uses_prefix_with_no_import_time_extra_mounts():
    """No module discover_mounts can see needs an extra mount (confirmed
    live via `python -m protein_design_mcp.mounts .../protenix protenix`) --
    but a RUNTIME JIT compile does (task-13-report.md: `CUDA_HOME
    environment variable is not set` in-container). That mount and its
    CUDA_HOME env_vars are invisible to discover_mounts by construction
    (it only walks Python's import machinery, see run_rfdiffusion2.yaml's
    identical case) and are hand-declared with a comment saying so -- this
    asserts on that mount rather than requiring none at all."""
    engine = _manifest().engine
    assert engine.prefix == "/home/jk661/.conda/envs/protenix"
    assert engine.mounts == ("/usr/local/cuda-12.6",)
    assert engine.env_vars["CUDA_HOME"] == "/usr/local/cuda-12.6"


def test_validation_rejects_empty_chains():
    with pytest.raises(ToolInputError, match="chains"):
        validate_and_fill(_manifest(), {"chains": []})


def test_validation_fills_defaults():
    params = _base_params()
    assert params["cycle"] == 10
    assert params["step"] == 200
    assert params["sample"] == 5
    assert params["seeds"] == [101]
    assert params["dtype"] == "bf16"
    assert params["need_atom_confidence"] is True


def test_build_args_rejects_chain_missing_msa_key():
    params = _base_params()
    params["chains"] = [{"sequence": SEQ}]
    with pytest.raises(ValueError, match="msa"):
        build_args(_manifest(), params)


def test_build_args_rejects_non_integer_seed():
    params = _base_params()
    params["seeds"] = [101, "bad"]
    with pytest.raises(ValueError, match="seeds"):
        build_args(_manifest(), params)


def test_build_args_single_chain_msa_free():
    params = _base_params()
    args = build_args(_manifest(), params)
    job = json.loads(args[0])
    assert job["chains"] == [{"sequence": SEQ, "msa": None, "copies": 1}]
    assert job["seeds"] == [101]


def test_build_args_resolves_relative_msa_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "seq.a3m").write_text(">query\n" + SEQ + "\n")
    params = _base_params()
    params["chains"] = [{"sequence": SEQ, "msa": "seq.a3m"}]
    args = build_args(_manifest(), params)
    job = json.loads(args[0])
    assert job["chains"][0]["msa"] == str(tmp_path / "seq.a3m")


# --- parse_output ---

SUMMARY = {
    "plddt": 93.9, "gpde": 0.42, "ptm": 0.46, "iptm": 0.0,
    "chain_ptm": [0.46], "chain_iptm": [0.0], "has_clash": False,
    "ranking_score": 0.09,
}


def _run_with_outputs(tmp_path: Path, seeds: list[int], samples: int) -> CompletedRun:
    structures, confidences = [], []
    for seed in seeds:
        for i in range(samples):
            struct = tmp_path / f"job_sample_{i}_seed{seed}.cif"
            struct.write_text("data_job\n")
            structures.append(str(struct))
            conf = tmp_path / f"seed_{seed}_job_summary_confidence_sample_{i}.json"
            data = dict(SUMMARY, ptm=0.1 * (i + 1) + seed)
            conf.write_text(json.dumps(data))
            confidences.append(str(conf))
    return CompletedRun(
        returncode=0, stdout="", stderr="", workdir=tmp_path,
        outputs={"structures": structures, "confidence_json": confidences},
    )


def test_parse_output_reads_lowest_seed_sample(tmp_path):
    run = _run_with_outputs(tmp_path, seeds=[101], samples=1)
    result = parse_output(_manifest(), run)
    assert result["ptm"] == pytest.approx(101.1)
    assert result["model_pinned"] == "protenix_base_default_v1.0.0"
    assert result["num_structures"] == 1


def test_parse_output_picks_lowest_seed_among_many(tmp_path):
    run = _run_with_outputs(tmp_path, seeds=[101, 2], samples=1)
    result = parse_output(_manifest(), run)
    # seed 2 sorts lower than seed 101 numerically (not lexicographically)
    assert result["ptm"] == pytest.approx(2.1)


def test_parse_output_raises_when_confidence_json_missing(tmp_path):
    run = CompletedRun(returncode=0, stdout="", stderr="", workdir=tmp_path, outputs={})
    with pytest.raises(ValueError, match="confidence_json"):
        parse_output(_manifest(), run)
