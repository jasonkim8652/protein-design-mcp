import csv
import shutil
from pathlib import Path

import numpy as np
import pytest

from protein_design_mcp.adapters.boltzgen_analyze import build_args, parse_output
from protein_design_mcp.app import manifest_dir
from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.loader import load_manifests
from protein_design_mcp.validation import ToolInputError, validate_and_fill

MANIFEST_DIR = manifest_dir()
FIXTURES = Path(__file__).parent / "fixtures" / "boltzgen"


def _manifest():
    return next(m for m in load_manifests(MANIFEST_DIR) if m.name == "run_boltzgen_analyze")


def _base_params(**overrides):
    return validate_and_fill(
        _manifest(),
        {
            "design_spec": "design.yaml",
            "generated_files": [str(FIXTURES / "generated_designs" / f"design_spec.{ext}") for ext in ("cif", "npz")],
            "refold_structures": [str(FIXTURES / "refold/design_spec.cif")],
            "refold_metrics": [str(FIXTURES / "refold/design_spec.npz")],
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


def test_manifest_uses_bundled_foldseek_without_runtime_mounts():
    manifest = _manifest()
    assert manifest.engine.mounts == ()
    assert manifest.schema["foldseek_binary"]["default"] == "/usr/local/bin/foldseek"


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
    assert params["foldseek_binary"] == "/usr/local/bin/foldseek"


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
    params = _base_params()
    args = build_args(_manifest(), params)
    assert args[0] == str(Path("design.yaml"))
    assert args[args.index("--steps") + 1] == "analysis"
    assert "--protocol" not in args


def test_build_args_derives_design_dir_from_staged_files_parent(tmp_path):
    staged = tmp_path / "design_dir"
    shutil.copytree(FIXTURES / "generated_designs", staged)
    params = _base_params(generated_files=[str(staged / f"design_spec.{ext}") for ext in ("cif", "npz")])
    args = build_args(_manifest(), params)
    joined = " ".join(args)
    assert f"design_dir={staged}" in joined


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
    assert "foldseek_binary=/usr/local/bin/foldseek" in joined


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


# The content checks must catch monomer handoffs even with matching filenames.
def _tiny_handoff(tmp_path, *, refold_chains=None, refold_tokens=None):
    original = [("A", ["ALA", "ARG"]), ("B", ["GLY"])]
    def cif(path, chains):
        lines = ["data_test", "loop_", "_entity_poly.entity_id", "_entity_poly.type"]
        lines += [f"{i} polypeptide(L)" for i, _ in enumerate(chains, 1)]
        lines += ["loop_", "_struct_asym.id", "_struct_asym.entity_id"]
        lines += [f"{chain} {i}" for i, (chain, _) in enumerate(chains, 1)]
        lines += ["loop_", "_entity_poly_seq.entity_id", "_entity_poly_seq.num", "_entity_poly_seq.mon_id"]
        lines += [f"{i} {j} {res}" for i, (_, seq) in enumerate(chains, 1) for j, res in enumerate(seq, 1)]
        path.write_text("\n".join(lines) + "\n")
    generated, refold = tmp_path / "generated", tmp_path / "refold"
    generated.mkdir()
    refold.mkdir()
    cif(generated / "sample.cif", original)
    cif(refold / "sample.cif", original if refold_chains is None else refold_chains)
    np.savez(generated / "sample.npz", mol_type=np.zeros(3, dtype=int), design_mask=[1, 1, 0])
    tokens = [2, 3, 9] if refold_tokens is None else refold_tokens
    np.savez(refold / "sample.npz", mol_type=np.zeros((1, len(tokens)), dtype=int), res_type=np.eye(33, dtype=int)[tokens][None, :])
    return _base_params(generated_files=[str(generated / f"sample.{ext}") for ext in ("cif", "npz")], refold_structures=[str(refold / "sample.cif")], refold_metrics=[str(refold / "sample.npz")])


def test_preflight_accepts_complete_complex(tmp_path):
    assert "analysis" in build_args(_manifest(), _tiny_handoff(tmp_path))


def test_preflight_rejects_monomer_in_required_complex_inputs(tmp_path):
    params = _tiny_handoff(tmp_path, refold_chains=[("A", ["ALA", "ARG"])], refold_tokens=[2, 3])
    with pytest.raises(ToolInputError, match="complete.*complex") as exc:
        build_args(_manifest(), params)
    assert "design_refold_structures" in str(exc.value)


@pytest.mark.parametrize("chains", [[("A", ["ALA", "ARG"]), ("B", ["ALA"])], [("B", ["GLY"]), ("A", ["ALA", "ARG"])], [("A", ["ALA", "ARG"]), ("C", ["GLY"])]])
def test_preflight_rejects_changed_target_sequence_chain_order_or_ids(tmp_path, chains):
    with pytest.raises(ToolInputError, match="complete.*complex"):
        build_args(_manifest(), _tiny_handoff(tmp_path, refold_chains=chains))


@pytest.mark.parametrize("tokens", [[2, 3], [2, 3, 2]])
def test_preflight_rejects_incompatible_metrics_with_valid_complex_cif(tmp_path, tokens):
    with pytest.raises(ToolInputError, match="refold_metrics"):
        build_args(_manifest(), _tiny_handoff(tmp_path, refold_tokens=tokens))


@pytest.mark.parametrize("field", ["generated_files", "refold_structures", "refold_metrics"])
def test_preflight_rejects_mismatched_design_ids(tmp_path, field):
    params = _tiny_handoff(tmp_path)
    source = Path(params[field][0])
    renamed = source.with_name("other" + source.suffix)
    source.rename(renamed)
    params[field][0] = str(renamed)
    with pytest.raises(ToolInputError, match="design IDs"):
        build_args(_manifest(), params)


def test_preflight_rejects_duplicate_design_ids(tmp_path):
    params = _tiny_handoff(tmp_path)
    params["refold_structures"] *= 2
    with pytest.raises(ToolInputError, match="duplicate"):
        build_args(_manifest(), params)


@pytest.mark.parametrize("extra", [{"designfolding_metrics": True}, {"design_refold_structures": ["unused.cif"]}, {"design_refold_metrics": ["unused.npz"]}])
def test_preflight_requires_paired_design_refold_outputs(tmp_path, extra):
    params = _tiny_handoff(tmp_path)
    params.update(extra)
    with pytest.raises(ToolInputError, match="design_refold_structures.*design_refold_metrics"):
        build_args(_manifest(), params)


def test_preflight_accepts_optional_monomer_with_required_complex(tmp_path):
    params = _tiny_handoff(tmp_path)
    monomer = tmp_path / "monomer"
    monomer.mkdir()
    mono = _tiny_handoff(monomer, refold_chains=[("A", ["ALA", "ARG"])], refold_tokens=[2, 3])
    params.update(designfolding_metrics=True, design_refold_structures=mono["refold_structures"], design_refold_metrics=mono["refold_metrics"])
    assert "designfolding_metrics=true" in build_args(_manifest(), params)


def test_preflight_rejects_unreadable_cif_as_input_error(tmp_path):
    params = _tiny_handoff(tmp_path)
    Path(params["refold_structures"][0]).write_text("not a cif")
    with pytest.raises(ToolInputError, match="refold_structures"):
        build_args(_manifest(), params)
