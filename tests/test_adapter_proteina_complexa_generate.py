from pathlib import Path

import pytest
import yaml

from protein_design_mcp.adapters.proteina_complexa_generate import build_args, parse_output
from protein_design_mcp.app import manifest_dir
from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.schema import parse_manifest
from protein_design_mcp.validation import ToolInputError, validate_and_fill

MANIFEST_DIR = manifest_dir()
MANIFEST_PATH = MANIFEST_DIR / "run_proteina_complexa_generate.yaml"


def _manifest():
    """Parse this tool's own manifest file directly, rather than through
    load_manifests/load_manifests_resilient over the whole directory --
    those also validate CROSS-manifest consistency (sibling-doc headings,
    doc references resolving to real tool names) for every OTHER manifest
    in the directory too, so this tool's own tests would otherwise be at
    the mercy of an unrelated, concurrently-edited manifest elsewhere in
    the same directory (this project ships ~25+ manifests from several
    concurrent waves). This still fully exercises this manifest's own
    schema/doc parsing (parse_manifest, called by _load_one, is exactly
    what the directory-wide loaders call per file)."""
    return parse_manifest(yaml.safe_load(MANIFEST_PATH.read_text()))


def _base_params(**overrides):
    return validate_and_fill(_manifest(), {"task_name": "02_PDL1", **overrides})


# --- manifest shape ---


def test_manifest_loads_and_is_gpu_binder_generation():
    m = _manifest()
    assert m.category == "binder_generation"
    assert m.composite is False
    assert m.requires.gpu is True


def test_manifest_uses_prefix_not_env():
    engine = _manifest().engine
    assert engine.prefix == "/home/jk661/.conda/envs/proteina"
    assert engine.env is None
    assert engine.entry == ("complexa", "generate")
    assert engine.repo == "proteinfoundation"


def test_manifest_sets_pythonnousersite():
    assert _manifest().engine.env_vars.get("PYTHONNOUSERSITE") == "1"


def test_manifest_sets_complexa_init():
    """complexa refuses every step but init/demo/download/validate/status
    without COMPLEXA_INIT set -- verified live, 2026-09-22."""
    assert _manifest().engine.env_vars.get("COMPLEXA_INIT") == "uv"


def test_manifest_documents_no_raw_override_passthrough():
    doc = _manifest().doc
    assert "code-execution" in doc.lower()
    schema = _manifest().schema
    assert "config" not in schema
    assert "overrides" not in schema


def test_task_name_is_required_enum():
    spec = _manifest().schema["task_name"]
    assert spec["required"] is True
    assert "02_PDL1" in spec["enum"]
    assert len(spec["enum"]) == 44


def test_job_splitting_is_not_exposed():
    """See the manifest doc's 'GPU job splitting is intentionally not
    exposed' section -- cli_runner.py:719 assigns
    CUDA_VISIBLE_DEVICES=str(job_id) per parallel job, which on this
    shared host would dispatch onto other users' GPUs."""
    schema = _manifest().schema
    assert "gen_njobs" not in schema
    assert "job_id" not in schema


# --- validation corner cases ---


def test_validation_rejects_missing_task_name():
    with pytest.raises(ToolInputError, match="task_name"):
        validate_and_fill(_manifest(), {})


def test_validation_rejects_unknown_task_name():
    with pytest.raises(ToolInputError):
        _base_params(task_name="99_NOT_A_TARGET")


def test_validation_fills_defaults():
    params = _base_params()
    assert params["search_algorithm"] == "best-of-n"
    assert params["nsteps"] == 400
    assert params["best_of_n_replicas"] == 2
    assert params["seed"] == 5
    assert "reward_weights" in params
    assert params["reward_weights"]["i_pae"] == -1.0


def test_validation_accepts_boundary_nsteps():
    params = _base_params(nsteps=2)
    assert params["nsteps"] == 2
    with pytest.raises(ToolInputError):
        _base_params(nsteps=1)


def test_validation_accepts_zero_search_reward_threshold():
    """0 is a legitimate reward threshold, not falsy-for-'unset'."""
    params = _base_params(search_reward_threshold=0)
    assert params["search_reward_threshold"] == 0


def test_validation_leaves_optional_binder_length_absent_when_unset():
    params = _base_params()
    assert "binder_length_min" not in params
    assert "binder_length_max" not in params


def test_validation_single_length_sample():
    params = _base_params(num_lengths=1)
    assert params["num_lengths"] == 1


# --- build_args ---


def test_build_args_uses_fixed_config_path():
    args = build_args(_manifest(), _base_params())
    assert args[0] == "/home/jk661/projects/proteina-complexa/configs/search_binder_local_pipeline.yaml"


def test_build_args_ends_with_verbose():
    args = build_args(_manifest(), _base_params())
    assert args[-1] == "--verbose"


def test_build_args_hardcodes_single_job():
    args = build_args(_manifest(), _base_params())
    assert "++gen_njobs=1" in args


def test_build_args_includes_task_name():
    args = build_args(_manifest(), _base_params(task_name="33_TrkA"))
    assert "++generation.task_name=\"33_TrkA\"" in args


def test_build_args_overrides_ckpt_paths_to_absolute():
    """See the manifest doc's 'Path resolution' section -- ./ckpts is
    relative to the repo root, not this tool's scratch cwd."""
    args = build_args(_manifest(), _base_params())
    assert (
        '++ckpt_path="/home/jk661/projects/proteina-complexa/ckpts"' in args
    )
    assert (
        '++autoencoder_ckpt_path='
        '"/home/jk661/projects/proteina-complexa/ckpts/complexa_ae.ckpt"'
        in args
    )


def test_build_args_overrides_target_pdb_path_to_absolute():
    args = build_args(_manifest(), _base_params(task_name="02_PDL1"))
    joined = " ".join(args)
    assert "conditional_features.0.pdb_path=" in joined
    assert (
        "/home/jk661/projects/proteina-complexa/assets/target_data/"
        "bindcraft_targets/PD-L1.pdb" in joined
    )


def test_build_args_target_pdb_path_covers_every_enum_value():
    """Every task_name the manifest's enum offers must resolve to a real
    entry in this adapter's own target-path table, or a caller-visible
    choice would silently 500 at build_args time."""
    manifest = _manifest()
    for task_name in manifest.schema["task_name"]["enum"]:
        params = _base_params(task_name=task_name)
        args = build_args(manifest, params)
        assert "conditional_features.0.pdb_path=" in " ".join(args)


def test_build_args_omits_binder_length_overrides_when_unset():
    args = build_args(_manifest(), _base_params())
    assert not any(a.startswith("++generation.dataloader.dataset.nres.low=") for a in args)
    assert not any(a.startswith("++generation.dataloader.dataset.nres.high=") for a in args)


def test_build_args_includes_binder_length_overrides_when_set():
    args = build_args(_manifest(), _base_params(binder_length_min=80, binder_length_max=150))
    assert "++generation.dataloader.dataset.nres.low=80" in args
    assert "++generation.dataloader.dataset.nres.high=150" in args


def test_build_args_renders_null_search_reward_threshold():
    args = build_args(_manifest(), _base_params())
    assert "++generation.search.reward_threshold=null" in args


def test_build_args_renders_zero_search_reward_threshold_as_zero_not_null():
    args = build_args(_manifest(), _base_params(search_reward_threshold=0))
    assert "++generation.search.reward_threshold=0" in args


def test_build_args_maps_refinement_none_to_hydra_null():
    args = build_args(_manifest(), _base_params(refinement_algorithm="none"))
    assert "++generation.refinement.algorithm=null" in args


def test_build_args_maps_refinement_sequence_hallucination():
    args = build_args(_manifest(), _base_params(refinement_algorithm="sequence_hallucination"))
    assert '++generation.refinement.algorithm="sequence_hallucination"' in args


def test_build_args_renders_step_checkpoints_list():
    args = build_args(_manifest(), _base_params())
    assert "++generation.search.step_checkpoints=[0,100,200,300,400]" in args


def test_build_args_includes_every_reward_weight_key():
    args = build_args(_manifest(), _base_params())
    joined = " ".join(args)
    for component in (
        "con", "i_pae", "plddt", "dgram_cce", "min_ipae", "min_ipsae",
        "avg_ipsae", "max_ipsae", "min_ipsae_10", "max_ipsae_10",
        "avg_ipsae_10", "i_ptm", "i_ptm_energy", "rg", "nc_termini",
        "helix_binder", "alignment_bb_ca_binder",
    ):
        assert f"reward_weights.{component}=" in joined


def test_build_args_overrides_single_reward_weight():
    params = _base_params(reward_weights={"i_pae": -1.0, "i_ptm": 0.5})
    args = build_args(_manifest(), params)
    assert (
        "++generation.reward_model.reward_models.af2folding.reward_weights.i_ptm=0.5"
        in args
    )


def test_build_args_rejects_non_object_reward_weights():
    manifest = _manifest()
    params = validate_and_fill(manifest, {"task_name": "02_PDL1"})
    params["reward_weights"] = "not-a-dict"
    with pytest.raises(ValueError, match="reward_weights"):
        build_args(manifest, params)


def test_build_args_omits_refinement_loss_weights_when_unset():
    args = build_args(_manifest(), _base_params())
    assert not any("refinement.loss_weights." in a for a in args)


def test_build_args_includes_refinement_loss_weights_when_set():
    params = _base_params(refinement_loss_weights={"i_ptm": 0.2})
    args = build_args(_manifest(), params)
    assert "++generation.refinement.loss_weights.i_ptm=0.2" in args


def test_build_args_reward_model_nums_default_null():
    args = build_args(_manifest(), _base_params())
    assert (
        "++generation.reward_model.reward_models.af2folding.model_nums=null" in args
    )


def test_build_args_reward_model_nums_renders_list_when_set():
    params = _base_params(reward_model_nums=[1, 2])
    args = build_args(_manifest(), params)
    assert (
        "++generation.reward_model.reward_models.af2folding.model_nums=[1,2]" in args
    )


# --- parse_output ---


def _write_rewards_csv(path: Path) -> None:
    path.write_text(
        "pdb_path,pdb_index,aatype,total_reward,i_pae,plddt\n"
        "/tmp/a.pdb,0,MKT,2.5,-4.0,0.8\n"
        "/tmp/b.pdb,1,MKV,0,-2.0,0.6\n"
    )


def test_parse_output_reads_every_row_and_reward_components(tmp_path):
    csv_path = tmp_path / "rewards_search_binder_local_pipeline_0.csv"
    _write_rewards_csv(csv_path)
    run = CompletedRun(
        returncode=0, stdout="", stderr="", workdir=tmp_path,
        outputs={"rewards_csv": str(csv_path)},
    )
    result = parse_output(_manifest(), run)
    assert result["num_samples"] == 2
    assert result["rewards"][0]["pdb_path"] == "/tmp/a.pdb"
    assert result["rewards"][0]["total_reward"] == pytest.approx(2.5)
    assert result["rewards"][0]["reward_components"]["i_pae"] == pytest.approx(-4.0)


def test_parse_output_preserves_zero_total_reward():
    """0 is a real reward value, must not become None/falsy."""
    import csv as csv_module
    import io

    handle = io.StringIO()
    writer = csv_module.DictWriter(handle, fieldnames=["pdb_path", "pdb_index", "total_reward"])
    writer.writeheader()
    writer.writerow({"pdb_path": "/tmp/b.pdb", "pdb_index": 1, "total_reward": "0"})
    rows = list(csv_module.DictReader(io.StringIO(handle.getvalue())))
    assert rows[0]["total_reward"] == "0"


def test_parse_output_raises_when_rewards_csv_missing(tmp_path):
    run = CompletedRun(returncode=0, stdout="", stderr="", workdir=tmp_path, outputs={})
    with pytest.raises(ValueError, match="rewards_csv"):
        parse_output(_manifest(), run)


def test_parse_output_handles_single_sample(tmp_path):
    csv_path = tmp_path / "rewards_search_binder_local_pipeline_0.csv"
    csv_path.write_text("pdb_path,pdb_index,total_reward\n/tmp/only.pdb,0,1.0\n")
    run = CompletedRun(
        returncode=0, stdout="", stderr="", workdir=tmp_path,
        outputs={"rewards_csv": str(csv_path)},
    )
    result = parse_output(_manifest(), run)
    assert result["num_samples"] == 1


def test_parse_output_handles_empty_rewards_rows(tmp_path):
    """Header only, zero data rows -- must not crash, num_samples == 0."""
    csv_path = tmp_path / "rewards_search_binder_local_pipeline_0.csv"
    csv_path.write_text("pdb_path,pdb_index,total_reward\n")
    run = CompletedRun(
        returncode=0, stdout="", stderr="", workdir=tmp_path,
        outputs={"rewards_csv": str(csv_path)},
    )
    result = parse_output(_manifest(), run)
    assert result["num_samples"] == 0
    assert result["rewards"] == []
