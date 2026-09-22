import csv
from pathlib import Path

import pytest

from protein_design_mcp.adapters.boltzgen_analyze import build_args, parse_output
from protein_design_mcp.app import manifest_dir
from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.loader import load_manifests
from protein_design_mcp.validation import ToolInputError, validate_and_fill

MANIFEST_DIR = manifest_dir()


def _manifest():
    return next(m for m in load_manifests(MANIFEST_DIR) if m.name == "run_boltzgen_analyze")


def _base_params(**overrides):
    return validate_and_fill(
        _manifest(),
        {
            "design_spec": "design.yaml",
            "generated_files": ["a.cif", "a.npz"],
            "refold_structures": ["a.cif"],
            "refold_metrics": ["a.npz"],
            **overrides,
        },
    )


# --- manifest shape ---


def test_manifest_loads_and_is_run_analysis_no_gpu():
    m = _manifest()
    assert m.category == "run_analysis"
    assert m.requires.gpu is False


def test_manifest_stages_all_five_groups_with_explicit_subdirs():
    engine = _manifest().engine
    assert set(engine.stage) == {
        "generated_files", "refold_structures", "refold_metrics",
        "design_refold_structures", "design_refold_metrics",
    }
    assert engine.stage_subdir == {
        "generated_files": "design_dir",
        "refold_structures": "design_dir/refold_cif",
        "refold_metrics": "design_dir/fold_out_npz",
        "design_refold_structures": "design_dir/refold_design_cif",
        "design_refold_metrics": "design_dir/fold_out_design_npz",
    }


def test_manifest_documents_that_it_runs_no_model():
    text = (_manifest().summary + _manifest().doc).lower()
    assert "runs no model" in text


def test_manifest_mounts_the_foldseek_binary_with_a_discovery_explanation():
    """foldseek is a standalone binary invoked as a subprocess, not a Python
    import -- python -m protein_design_mcp.mounts (discover_mounts) cannot
    find it (it walks sys.path/.pth files), so it must be hand-declared, and
    the manifest must say so or a future mounts-regeneration pass will
    assume it's spurious and delete it (the exact failure class
    run_mmseqs_search's own mounts comment already guards against)."""
    engine = _manifest().engine
    assert "/home/jk661/.local/bin/foldseek" in engine.mounts


def test_manifest_explains_why_foldseek_is_hand_declared():
    """The explanation belongs in the manifest's own YAML comment (PyYAML
    strips comments from the parsed Manifest, so this reads the raw file,
    same place a human re-deriving mounts would actually look)."""
    source = (MANIFEST_DIR / "run_boltzgen_analyze.yaml").read_text().lower()
    assert "discover_mounts" in source
    assert "hand" in source


def test_run_clustering_description_states_benefit_and_cost_not_unverified():
    """Coordinator follow-up: foldseek is confirmed present on this host, so
    'unverified' is no longer the right framing -- the description must say
    what turning this on gets you and what it costs instead."""
    desc = _manifest().schema["run_clustering"]["description"].lower()
    assert "unverified" not in desc
    assert "cluster" in desc


def test_designfolding_metrics_path_records_its_unexercised_status_in_the_manifest():
    """Coordinator follow-up: the designfolding_metrics -> run_boltzgen_design_fold
    path was verified unit-test-only, not with a live GPU run -- that caveat
    must live in the manifest/doc (git-tracked), not only in the
    (gitignored) wave report, or it disappears."""
    text = (_manifest().doc + _manifest().schema["designfolding_metrics"]["description"]).lower()
    assert "not yet" in text or "not been" in text or "no live" in text or "unverified" in text


def test_required_and_optional_params():
    schema = _manifest().schema
    assert schema["generated_files"]["required"] is True
    assert schema["refold_structures"]["required"] is True
    assert schema["refold_metrics"]["required"] is True
    assert schema["design_refold_structures"]["required"] is False
    assert schema["design_refold_metrics"]["required"] is False


# --- validation ---


def test_validation_fills_defaults():
    params = _base_params()
    assert params["backbone_fold_metrics"] is True
    assert params["designfolding_metrics"] is False
    assert params["num_processes"] == 32
    assert params["num_workers"] == 4
    assert params["foldseek_binary"] == "/home/jk661/.local/bin/foldseek"


def test_validation_rejects_missing_refold_metrics():
    with pytest.raises(ToolInputError, match="refold_metrics"):
        validate_and_fill(
            _manifest(),
            {
                "design_spec": "d.yaml",
                "generated_files": ["a.cif", "a.npz"],
                "refold_structures": ["a.cif"],
            },
        )


def test_design_refold_params_absent_by_default():
    params = _base_params()
    assert "design_refold_structures" not in params
    assert "design_refold_metrics" not in params


# --- build_args ---


def test_build_args_wraps_boltzgen_run_with_analysis_step_only():
    params = _base_params(generated_files=[
        "/scratch/design_dir/a.cif", "/scratch/design_dir/a.npz",
    ])
    args = build_args(_manifest(), params)
    assert args[0] == str(Path("design.yaml"))
    assert args[args.index("--steps") + 1] == "analysis"
    assert "--protocol" not in args


def test_build_args_derives_design_dir_from_staged_files_parent():
    params = _base_params(generated_files=[
        "/scratch/design_dir/a.cif", "/scratch/design_dir/a.npz",
    ])
    args = build_args(_manifest(), params)
    joined = " ".join(args)
    assert "design_dir=/scratch/design_dir" in joined


def test_build_args_renders_booleans_lowercase():
    params = _base_params(largest_hydrophobic=True, run_clustering=False)
    args = build_args(_manifest(), params)
    joined = " ".join(args)
    assert "largest_hydrophobic=true" in joined
    assert "run_clustering=false" in joined


def test_build_args_includes_liability_and_process_knobs():
    params = _base_params(liability_modality="antibody", num_processes=8)
    args = build_args(_manifest(), params)
    joined = " ".join(args)
    assert "liability_modality=antibody" in joined
    assert "num_processes=8" in joined


def test_build_args_includes_foldseek_binary_path():
    params = _base_params(run_clustering=True)
    args = build_args(_manifest(), params)
    joined = " ".join(args)
    assert "run_clustering=true" in joined
    assert "foldseek_binary=/home/jk661/.local/bin/foldseek" in joined


# --- parse_output ---


def _write_metrics_csv(path: Path, num_rows: int) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["id", "design_ptm"])
        writer.writeheader()
        for i in range(num_rows):
            writer.writerow({"id": f"design_{i}", "design_ptm": 0.9})


def test_parse_output_counts_rows(tmp_path):
    csv_path = tmp_path / "aggregate_metrics_analyze.csv"
    _write_metrics_csv(csv_path, 3)
    run = CompletedRun(
        returncode=0, stdout="", stderr="", workdir=tmp_path,
        outputs={"aggregate_metrics_csv": str(csv_path)},
    )
    result = parse_output(_manifest(), run)
    assert result["num_designs_analyzed"] == 3


def test_parse_output_handles_zero_rows(tmp_path):
    """Corner case: header-only CSV (e.g. every design failed upstream)."""
    csv_path = tmp_path / "aggregate_metrics_analyze.csv"
    _write_metrics_csv(csv_path, 0)
    run = CompletedRun(
        returncode=0, stdout="", stderr="", workdir=tmp_path,
        outputs={"aggregate_metrics_csv": str(csv_path)},
    )
    result = parse_output(_manifest(), run)
    assert result["num_designs_analyzed"] == 0


def test_parse_output_raises_when_csv_missing(tmp_path):
    run = CompletedRun(returncode=0, stdout="", stderr="", workdir=tmp_path, outputs={})
    with pytest.raises(ValueError, match="aggregate_metrics_csv"):
        parse_output(_manifest(), run)
