from pathlib import Path

import pytest
import yaml

from protein_design_mcp.adapters.proteina_complexa_filter import build_args, parse_output
from protein_design_mcp.app import manifest_dir
from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.schema import parse_manifest
from protein_design_mcp.validation import ToolInputError, validate_and_fill

MANIFEST_DIR = manifest_dir()
MANIFEST_PATH = MANIFEST_DIR / "run_proteina_complexa_filter.yaml"


def _manifest():
    """See test_adapter_proteina_complexa_generate.py's _manifest() docstring
    for why this parses the file directly rather than via the directory-wide
    loaders."""
    return parse_manifest(yaml.safe_load(MANIFEST_PATH.read_text()))


def _base_params(**overrides):
    return validate_and_fill(_manifest(), {"rewards_csv": "rewards.csv", **overrides})


# --- manifest shape ---


def test_manifest_loads_and_is_cpu_run_analysis():
    m = _manifest()
    assert m.category == "run_analysis"
    assert m.composite is False
    assert m.requires.gpu is False


def test_manifest_uses_prefix_not_env():
    engine = _manifest().engine
    assert engine.prefix == "/home/jk661/.conda/envs/proteina_complexa"
    assert engine.env is None
    assert engine.entry == ("complexa", "filter")
    assert engine.repo == "proteinfoundation"


def test_manifest_stages_rewards_csv():
    assert _manifest().engine.stage == ("rewards_csv",)


def test_manifest_sets_complexa_init():
    """complexa refuses every step but init/demo/download/validate/status
    without COMPLEXA_INIT set -- verified live, 2026-09-22 (see
    run_proteina_complexa_generate's identical check)."""
    assert _manifest().engine.env_vars.get("COMPLEXA_INIT") == "uv"


def test_manifest_documents_why_delete_non_top_n_samples_is_not_exposed():
    schema = _manifest().schema
    assert "delete_non_top_n_samples" not in schema
    assert "no knob with no effect" in _manifest().doc.lower()


def test_rewards_csv_is_required():
    assert _manifest().schema["rewards_csv"]["required"] is True


# --- validation corner cases ---


def test_validation_fills_defaults():
    params = _base_params()
    assert params["filter_samples_limit"] == 1000
    assert params["dedup_sequence"] is True
    assert "reward_threshold" not in params


def test_validation_rejects_missing_rewards_csv():
    with pytest.raises(ToolInputError, match="rewards_csv"):
        validate_and_fill(_manifest(), {})


def test_validation_accepts_zero_reward_threshold():
    """0.0 is a legitimate threshold, not falsy-for-'unset'."""
    params = _base_params(reward_threshold=0.0)
    assert params["reward_threshold"] == 0.0


def test_validation_accepts_boundary_filter_samples_limit():
    params = _base_params(filter_samples_limit=1)
    assert params["filter_samples_limit"] == 1
    with pytest.raises(ToolInputError):
        _base_params(filter_samples_limit=0)


def test_validation_dedup_sequence_can_be_disabled():
    params = _base_params(dedup_sequence=False)
    assert params["dedup_sequence"] is False


# --- build_args ---


def test_build_args_uses_fixed_config_path():
    args = build_args(_manifest(), _base_params())
    assert args[0] == "/home/jk661/projects/proteina-complexa/configs/search_binder_local_pipeline.yaml"


def test_build_args_ends_with_verbose():
    args = build_args(_manifest(), _base_params())
    assert args[-1] == "--verbose"


def test_build_args_points_root_path_at_staged_directory():
    args = build_args(_manifest(), _base_params())
    assert "++root_path=rewards_csv" in args


def test_build_args_matches_generate_tools_config_name():
    """filter must resolve the SAME config_name as
    run_proteina_complexa_generate, or the staged rewards CSV's basename
    (unchanged from generate's own naming) will not match filter.py's own
    `rewards_{config_name}_*.csv` file-discovery pattern."""
    args = build_args(_manifest(), _base_params())
    assert "++base_config_name=search_binder_local_pipeline" in args


def test_build_args_renders_null_reward_threshold():
    args = build_args(_manifest(), _base_params())
    assert "++generation.filter.reward_threshold=null" in args


def test_build_args_renders_zero_reward_threshold_as_zero_not_null():
    args = build_args(_manifest(), _base_params(reward_threshold=0.0))
    assert "++generation.filter.reward_threshold=0.0" in args


def test_build_args_includes_filter_samples_limit_and_dedup():
    args = build_args(_manifest(), _base_params(filter_samples_limit=5, dedup_sequence=False))
    assert "++generation.filter.filter_samples_limit=5" in args
    assert "++generation.filter.dedup_sequence=false" in args


# --- parse_output ---


def test_parse_output_reads_selected_designs(tmp_path):
    top_csv = tmp_path / "top_samples_search_binder_local_pipeline.csv"
    top_csv.write_text(
        "pdb_path,total_reward,aatype\n/tmp/a.pdb,2.5,MKT\n/tmp/b.pdb,1.1,MKV\n"
    )
    all_csv = tmp_path / "all_rewards_search_binder_local_pipeline.csv"
    all_csv.write_text("pdb_path,total_reward,aatype\n/tmp/a.pdb,2.5,MKT\n/tmp/b.pdb,1.1,MKV\n/tmp/c.pdb,-1.0,MKX\n")
    run = CompletedRun(
        returncode=0, stdout="", stderr="", workdir=tmp_path,
        outputs={"top_samples_csv": str(top_csv), "all_rewards_csv": str(all_csv)},
    )
    result = parse_output(_manifest(), run)
    assert result["num_selected"] == 2
    assert result["num_total_designs"] == 3
    assert result["selected_designs"][0]["pdb_path"] == "/tmp/a.pdb"
    assert result["selected_designs"][0]["total_reward"] == pytest.approx(2.5)


def test_parse_output_preserves_zero_total_reward_row(tmp_path):
    top_csv = tmp_path / "top_samples_search_binder_local_pipeline.csv"
    top_csv.write_text("pdb_path,total_reward\n/tmp/a.pdb,0\n")
    run = CompletedRun(
        returncode=0, stdout="", stderr="", workdir=tmp_path,
        outputs={"top_samples_csv": str(top_csv)},
    )
    result = parse_output(_manifest(), run)
    assert result["selected_designs"][0]["total_reward"] == 0.0


def test_parse_output_handles_single_selected_design(tmp_path):
    top_csv = tmp_path / "top_samples_search_binder_local_pipeline.csv"
    top_csv.write_text("pdb_path,total_reward\n/tmp/only.pdb,9.0\n")
    run = CompletedRun(
        returncode=0, stdout="", stderr="", workdir=tmp_path,
        outputs={"top_samples_csv": str(top_csv)},
    )
    result = parse_output(_manifest(), run)
    assert result["num_selected"] == 1


def test_parse_output_handles_empty_selection(tmp_path):
    """No design passed the threshold -- header only, zero rows."""
    top_csv = tmp_path / "top_samples_search_binder_local_pipeline.csv"
    top_csv.write_text("pdb_path,total_reward\n")
    run = CompletedRun(
        returncode=0, stdout="", stderr="", workdir=tmp_path,
        outputs={"top_samples_csv": str(top_csv)},
    )
    result = parse_output(_manifest(), run)
    assert result["num_selected"] == 0
    assert result["selected_designs"] == []


def test_parse_output_raises_when_top_samples_csv_missing(tmp_path):
    run = CompletedRun(returncode=0, stdout="", stderr="", workdir=tmp_path, outputs={})
    with pytest.raises(ValueError, match="top_samples_csv"):
        parse_output(_manifest(), run)


def test_parse_output_falls_back_to_stdout_when_all_rewards_csv_absent(tmp_path):
    top_csv = tmp_path / "top_samples_search_binder_local_pipeline.csv"
    top_csv.write_text("pdb_path,total_reward\n/tmp/a.pdb,1.0\n")
    run = CompletedRun(
        returncode=0,
        stdout="Final top samples: 7\n",
        stderr="",
        workdir=tmp_path,
        outputs={"top_samples_csv": str(top_csv)},
    )
    result = parse_output(_manifest(), run)
    assert result["num_total_designs"] == 7
