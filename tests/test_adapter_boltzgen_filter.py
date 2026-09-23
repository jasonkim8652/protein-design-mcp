import csv
from pathlib import Path

import pytest

from protein_design_mcp.adapters.boltzgen_filter import build_args, parse_output
from protein_design_mcp.app import manifest_dir
from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.loader import load_manifests
from protein_design_mcp.validation import ToolInputError, validate_and_fill

MANIFEST_DIR = manifest_dir()


def _manifest():
    return next(m for m in load_manifests(MANIFEST_DIR) if m.name == "run_boltzgen_filter")


def _base_params(**overrides):
    return validate_and_fill(
        _manifest(),
        {
            "design_spec": "design.yaml",
            "generated_files": ["design_0.cif", "design_0.npz"],
            "metrics_files": ["metrics.csv", "seqs.pkl.gz"],
            "refold_structures": ["design_0.cif"],
            **overrides,
        },
    )


# --- manifest shape ---


def test_manifest_loads_and_is_run_analysis_no_gpu():
    m = _manifest()
    assert m.category == "run_analysis"
    assert m.composite is False
    assert m.requires.gpu is False


def test_manifest_uses_prefix_not_env():
    engine = _manifest().engine
    assert engine.prefix == "/home/jk661/miniforge3/envs/boltzgen"
    assert engine.env is None
    assert engine.entry == ("boltzgen", "run")


def test_manifest_documents_that_it_runs_no_model():
    text = (_manifest().summary + _manifest().doc).lower()
    assert "runs no model" in text


def test_design_spec_and_file_lists_are_required():
    schema = _manifest().schema
    assert schema["design_spec"]["required"] is True
    assert schema["generated_files"]["required"] is True
    assert schema["metrics_files"]["required"] is True
    assert schema["refold_structures"]["required"] is True


def test_manifest_stages_all_three_file_lists_into_one_shared_tree():
    engine = _manifest().engine
    assert set(engine.stage) == {"generated_files", "metrics_files", "refold_structures"}
    assert engine.stage_subdir == {
        "generated_files": "design_dir",
        "metrics_files": "design_dir",
        "refold_structures": "design_dir/refold_cif",
    }


def test_every_threshold_and_ranking_knob_is_a_parameter():
    schema = _manifest().schema
    for key in (
        "budget",
        "top_budget",
        "alpha",
        "refolding_rmsd_threshold",
        "filter_biased",
        "filter_cysteine",
        "filter_designfolding",
        "from_inverse_folded",
        "filter_bindingsite",
        "filter_target_aligned",
        "metrics_override",
        "additional_filters",
        "size_buckets",
        "use_affinity",
    ):
        assert key in schema, f"{key} is not exposed as a schema parameter"
        assert schema[key]["description"], f"{key} has no description"


def test_min_interaction_pae_direction_is_documented():
    """WAVE-COMMON: a caller can't rank on min_interaction_pae correctly
    without knowing lower is better -- must be stated somewhere in the doc."""
    doc = _manifest().doc.lower()
    assert "lower is better" in doc or "lower-is-better" in doc


# --- validation corner cases ---


def test_validation_rejects_missing_generated_files():
    with pytest.raises(ToolInputError, match="generated_files"):
        validate_and_fill(
            _manifest(),
            {
                "design_spec": "design.yaml",
                "metrics_files": ["a.csv", "b.pkl.gz"],
                "refold_structures": ["a.cif"],
            },
        )


def test_validation_rejects_missing_metrics_files():
    with pytest.raises(ToolInputError, match="metrics_files"):
        validate_and_fill(
            _manifest(),
            {
                "design_spec": "design.yaml",
                "generated_files": ["a.cif", "a.npz"],
                "refold_structures": ["a.cif"],
            },
        )


def test_validation_rejects_missing_refold_structures():
    with pytest.raises(ToolInputError, match="refold_structures"):
        validate_and_fill(
            _manifest(),
            {
                "design_spec": "design.yaml",
                "generated_files": ["a.cif", "a.npz"],
                "metrics_files": ["a.csv", "b.pkl.gz"],
            },
        )


def test_validation_fills_defaults():
    params = _base_params()
    assert params["budget"] == 30
    assert params["alpha"] == 0.001
    assert params["filter_biased"] is True
    assert params["use_affinity"] is False


def test_validation_accepts_boundary_alpha():
    params = _base_params(alpha=1.0)
    assert params["alpha"] == 1.0
    with pytest.raises(ToolInputError):
        _base_params(alpha=1.1)


def test_validation_accepts_alpha_zero():
    """0 is a real, meaningful value (pure quality ranking) -- must not be
    treated as falsy/missing."""
    params = _base_params(alpha=0.0)
    assert params["alpha"] == 0.0


# --- build_args ---


def test_build_args_wraps_boltzgen_run_with_filtering_step_only():
    params = _base_params()
    args = build_args(_manifest(), params)
    assert args[0] == str(Path("design.yaml"))
    assert "--steps" in args
    assert args[args.index("--steps") + 1] == "filtering"
    assert "--config" in args
    assert args[args.index("--config") + 1] == "filtering"


def test_build_args_derives_design_dir_from_metrics_files_parent():
    params = _base_params(
        metrics_files=["/scratch/design_dir/metrics.csv", "/scratch/design_dir/seqs.pkl.gz"],
        refold_structures=["/scratch/design_dir/refold_cif/design_0.cif"],
    )
    args = build_args(_manifest(), params)
    joined = " ".join(args)
    assert "design_dir=/scratch/design_dir" in joined
    assert "outdir=." in joined


def test_build_args_renders_booleans_lowercase_for_omegaconf():
    params = _base_params(filter_biased=False)
    args = build_args(_manifest(), params)
    joined = " ".join(args)
    assert "filter_biased=false" in joined
    assert "filter_biased=False" not in joined


def test_build_args_omits_metrics_override_when_absent():
    params = _base_params()
    args = build_args(_manifest(), params)
    assert not any(a.startswith("metrics_override=") for a in args)


def test_build_args_renders_metrics_override_with_yaml_null():
    params = _base_params(metrics_override={"design_ptm": 2, "plip_hbonds_refolded": None})
    args = build_args(_manifest(), params)
    joined = " ".join(args)
    assert "design_ptm: 2" in joined
    assert "plip_hbonds_refolded: null" in joined


def test_build_args_renders_additional_filters_list():
    params = _base_params(
        additional_filters=[{"feature": "design_ptm", "threshold": 0.7, "lower_is_better": False}]
    )
    args = build_args(_manifest(), params)
    joined = " ".join(args)
    assert "feature: \"design_ptm\"" in joined
    assert "threshold: 0.7" in joined
    assert "lower_is_better: false" in joined


def test_build_args_rejects_additional_filter_missing_feature():
    params = _base_params(additional_filters=[{"threshold": 0.7, "lower_is_better": False}])
    with pytest.raises(ValueError, match="feature"):
        build_args(_manifest(), params)


def test_build_args_omits_empty_additional_filters_and_size_buckets():
    params = _base_params(additional_filters=[], size_buckets=[])
    args = build_args(_manifest(), params)
    joined = " ".join(args)
    assert "additional_filters=" not in joined
    assert "size_buckets=" not in joined


# --- parse_output ---


def _write_ranked_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "id",
        "final_rank",
        "designed_sequence",
        "designed_chain_sequence",
        "num_design",
        "design_to_target_iptm",
        "design_ptm",
        "min_design_to_target_pae",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_parse_output_reads_selected_designs(tmp_path):
    csv_path = tmp_path / "final_designs_metrics_1.csv"
    _write_ranked_csv(
        csv_path,
        [
            {
                "id": "1g13prot_small",
                "final_rank": 1,
                "designed_sequence": "AALVLALLVLLIELLNK",
                "designed_chain_sequence": "AALVLALLVLLIELLNK",
                "num_design": 17,
                "design_to_target_iptm": 0.60088,
                "design_ptm": 0.9398,
                "min_design_to_target_pae": 4.27042,
            }
        ],
    )
    run = CompletedRun(
        returncode=0,
        stdout="Total number of designs:     1\nRemaining designs: 1\n",
        stderr="",
        workdir=tmp_path,
        outputs={"ranked_metrics_csv": str(csv_path)},
    )
    result = parse_output(_manifest(), run)
    assert result["num_selected"] == 1
    assert result["num_total_designs"] == 1
    assert result["num_passing_all_filters"] == 1
    design = result["selected_designs"][0]
    assert design["id"] == "1g13prot_small"
    assert design["designed_sequence"] == "AALVLALLVLLIELLNK"
    assert design["design_to_target_iptm"] == pytest.approx(0.60088)


def test_parse_output_handles_zero_designs_selected(tmp_path):
    """Corner case: every design filtered out -- csv has a header but no
    rows. Must not crash, must report num_selected == 0, not None."""
    csv_path = tmp_path / "final_designs_metrics_1.csv"
    _write_ranked_csv(csv_path, [])
    run = CompletedRun(
        returncode=0,
        stdout="Total number of designs:     3\nRemaining designs: 0\n",
        stderr="",
        workdir=tmp_path,
        outputs={"ranked_metrics_csv": str(csv_path)},
    )
    result = parse_output(_manifest(), run)
    assert result["num_selected"] == 0
    assert result["selected_designs"] == []
    assert result["num_total_designs"] == 3
    assert result["num_passing_all_filters"] == 0


def test_parse_output_raises_when_ranked_csv_missing(tmp_path):
    run = CompletedRun(returncode=0, stdout="", stderr="", workdir=tmp_path, outputs={})
    with pytest.raises(ValueError, match="ranked_metrics_csv"):
        parse_output(_manifest(), run)
