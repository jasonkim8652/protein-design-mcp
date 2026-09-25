import json
from pathlib import Path

import pytest

from protein_design_mcp.adapters.protpardelle import build_args, parse_output
from protein_design_mcp.app import manifest_dir
from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.loader import load_manifests
from protein_design_mcp.validation import ToolInputError, validate_and_fill

TWO_CHAIN_PDB = (
    "ATOM      1  N   GLY A   1      11.104  13.207   2.145  1.00  0.00           N\n"
    "ATOM      2  CA  GLY A   1      12.560  13.207   2.145  1.00  0.00           C\n"
    "ATOM      3  C   GLY A   1      13.090  14.630   2.145  1.00  0.00           C\n"
    "ATOM      4  O   GLY A   1      12.400  15.630   2.145  1.00  0.00           O\n"
    "TER\n"
    "ATOM      5  N   GLY B   1      21.104  13.207   2.145  1.00  0.00           N\n"
    "ATOM      6  CA  GLY B   1      22.560  13.207   2.145  1.00  0.00           C\n"
    "ATOM      7  C   GLY B   1      23.090  14.630   2.145  1.00  0.00           C\n"
    "ATOM      8  O   GLY B   1      22.400  15.630   2.145  1.00  0.00           O\n"
    "ATOM      9  N   GLY B   2      24.400  14.700   2.200  1.00  0.00           N\n"
    "ATOM     10  CA  GLY B   2      25.100  16.000   2.300  1.00  0.00           C\n"
    "ATOM     11  C   GLY B   2      26.100  16.500   2.300  1.00  0.00           C\n"
    "ATOM     12  O   GLY B   2      27.100  16.000   2.300  1.00  0.00           O\n"
    "TER\n"
)

DEFAULT_PARAMS = {
    "target_pdb": "target.pdb",
    "contig": "A1-128;/;120-120",
    "total_lengths": [[128, 128], [120, 120]],
    "hotspots": ["A19", "A76"],
    "model": "cc83",
    "step_scale": 1.2,
    "schurn": 200.0,
    "crop_cond_start": 0.0,
    "translation": [0.0, 0.0, 0.0],
    "num_samples": 2,
    "batch_size": 8,
}


def _manifest():
    return next(m for m in load_manifests(manifest_dir()) if m.name == "run_protpardelle")


def test_manifest_loads_and_requires_gpu():
    m = _manifest()
    assert m.category == "binder_generation"
    assert m.requires.gpu is True


def test_manifest_validates_contig_pattern():
    import re
    pattern = re.compile(_manifest().schema["contig"]["pattern"])
    assert pattern.match("A1-128;/;120-120")
    assert pattern.match("0-20;A1-50;0-20;/;100-100")
    assert pattern.match("A1-79;/;B1-141;/;70-150")
    assert not pattern.match("A1-128 / 120-120")  # space-separated, not ';/;'
    assert not pattern.match("")


def test_build_args_serializes_total_lengths_and_hotspots():
    args = build_args(_manifest(), DEFAULT_PARAMS)
    assert json.loads(args[args.index("--total-lengths") + 1]) == [[128, 128], [120, 120]]
    assert json.loads(args[args.index("--hotspots") + 1]) == ["A19", "A76"]


def test_build_args_passes_null_hotspots_through():
    args = build_args(_manifest(), {**DEFAULT_PARAMS, "hotspots": None})
    assert json.loads(args[args.index("--hotspots") + 1]) is None


def test_build_args_omits_seed_when_not_given():
    args = build_args(_manifest(), DEFAULT_PARAMS)
    assert "--seed" not in args


def test_build_args_includes_seed_when_given():
    args = build_args(_manifest(), {**DEFAULT_PARAMS, "seed": 37})
    assert args[args.index("--seed") + 1] == "37"


def test_build_args_rejects_malformed_hotspot_tag():
    with pytest.raises(ValueError, match="malformed tag"):
        build_args(_manifest(), {**DEFAULT_PARAMS, "hotspots": ["not-a-tag"]})


def test_build_args_rejects_total_lengths_chain_count_mismatch():
    with pytest.raises(ValueError, match="contig"):
        build_args(_manifest(), {**DEFAULT_PARAMS, "total_lengths": [[128, 128]]})


def test_build_args_rejects_total_lengths_min_greater_than_max():
    with pytest.raises(ValueError, match="min <= max"):
        build_args(
            _manifest(),
            {**DEFAULT_PARAMS, "total_lengths": [[128, 128], [200, 100]]},
        )


def test_parse_output_reports_every_chain(tmp_path: Path):
    pdb_path = tmp_path / "sample_0.pdb"
    pdb_path.write_text(TWO_CHAIN_PDB)
    run = CompletedRun(
        returncode=0, stdout="", stderr="", workdir=tmp_path,
        outputs={"samples": [str(pdb_path)]},
    )
    result = parse_output(_manifest(), run)
    assert result["num_samples"] == 1
    chains = {c["chain_id"]: c["length"] for c in result["samples"][0]["chains"]}
    assert chains == {"A": 1, "B": 2}


def test_parse_output_raises_when_no_samples_collected():
    run = CompletedRun(returncode=0, stdout="", stderr="", workdir=Path("/tmp"), outputs={})
    with pytest.raises(ValueError, match="samples"):
        parse_output(_manifest(), run)


def test_validation_requires_target_pdb_contig_total_lengths_hotspots():
    with pytest.raises(ToolInputError):
        validate_and_fill(_manifest(), {})


def test_validation_rejects_malformed_contig_at_schema_level():
    with pytest.raises(ToolInputError, match="contig"):
        validate_and_fill(
            _manifest(),
            {
                "target_pdb": "t.pdb",
                "contig": "A1-128 / 120-120",
                "total_lengths": [[128, 128], [120, 120]],
                "hotspots": None,
            },
        )


def test_validation_fills_defaults():
    params = validate_and_fill(
        _manifest(),
        {
            "target_pdb": "t.pdb",
            "contig": "A1-128;/;120-120",
            "total_lengths": [[128, 128], [120, 120]],
            "hotspots": None,
        },
    )
    assert params["model"] == "cc83"
    assert params["step_scale"] == pytest.approx(1.2)
    assert params["schurn"] == pytest.approx(200.0)
    assert params.get("seed") is None
    assert params["translation"] == [0.0, 0.0, 0.0]


def test_validation_accepts_explicit_null_hotspots():
    # Corner case: None is a legitimate, required-to-be-stated value here,
    # not "missing".
    params = validate_and_fill(
        _manifest(),
        {
            "target_pdb": "t.pdb",
            "contig": "A1-128;/;120-120",
            "total_lengths": [[128, 128], [120, 120]],
            "hotspots": None,
        },
    )
    assert params["hotspots"] is None


# --- an all-atom model needs weights this host may not have -----------------


def test_an_all_atom_model_without_its_mpnn_weights_is_refused(tmp_path, monkeypatch):
    """`cc94` is the one checkpoint in the enum that the doc calls all-atom:
    it produces side chains, which means it runs ProteinMPNN, which loads
    `model_params/ProteinMPNN/vanilla_model_weights/v_48_020.pt`. That file is
    not installed here, so a live round sampled 500 backbone steps (~35s of
    GPU) and then died with a raw

        FileNotFoundError: [Errno 2] No such file or directory:
        '.../ProteinMPNN/vanilla_model_weights/v_48_020.pt'

    naming a path no caller chose and no message explained. The other three
    checkpoints are backbone-only and unaffected -- the same call with the
    default cc83 succeeds.
    """
    from protein_design_mcp.adapters import protpardelle
    from protein_design_mcp.validation import ToolInputError

    monkeypatch.setattr(protpardelle, "MPNN_WEIGHTS", tmp_path / "absent.pt")
    target = tmp_path / "t.pdb"
    target.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n")
    with pytest.raises(ToolInputError) as excinfo:
        protpardelle.build_args(None, {
            "target_pdb": str(target), "contig": "A1-1;/;5-5",
            "total_lengths": [[1, 1], [5, 5]], "hotspots": None, "model": "cc94",
        })
    message = str(excinfo.value)
    assert "cc94" in message
    assert "v_48_020" in message


def test_a_backbone_only_model_is_unaffected_by_the_missing_weights(tmp_path, monkeypatch):
    from protein_design_mcp.adapters import protpardelle

    monkeypatch.setattr(protpardelle, "MPNN_WEIGHTS", tmp_path / "absent.pt")
    target = tmp_path / "t.pdb"
    target.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n")
    args = protpardelle.build_args(None, {
        "target_pdb": str(target), "contig": "A1-1;/;5-5",
        "total_lengths": [[1, 1], [5, 5]], "hotspots": None, "model": "cc83",
        "step_scale": 1.2, "schurn": 0.0, "crop_cond_start": 0.0,
        "translation": [0.0, 0.0, 0.0], "num_samples": 1, "batch_size": 1,
    })
    assert "cc83" in args


def test_the_all_atom_model_runs_when_the_weights_are_present(tmp_path, monkeypatch):
    from protein_design_mcp.adapters import protpardelle

    weights = tmp_path / "v_48_020.pt"
    weights.write_bytes(b"not really weights")
    monkeypatch.setattr(protpardelle, "MPNN_WEIGHTS", weights)
    target = tmp_path / "t.pdb"
    target.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n")
    args = protpardelle.build_args(None, {
        "target_pdb": str(target), "contig": "A1-1;/;5-5",
        "total_lengths": [[1, 1], [5, 5]], "hotspots": None, "model": "cc94",
        "step_scale": 1.2, "schurn": 0.0, "crop_cond_start": 0.0,
        "translation": [0.0, 0.0, 0.0], "num_samples": 1, "batch_size": 1,
    })
    assert "cc94" in args


def test_only_cc94_is_treated_as_all_atom():
    """Evidence-based, not guessed: the manifest doc calls cc83 and cc95
    backbone-only, cc78 experimental, and cc94 all-atom, and cc94 is the one
    observed to load ProteinMPNN."""
    from protein_design_mcp.adapters import protpardelle

    assert protpardelle.ALL_ATOM_MODELS == {"cc94"}


# --- a dangling chain break is not a chain ----------------------------------


@pytest.mark.parametrize("contig,chains", [
    ("B2-505;/;80-120", 2),   # the documented shape: a break BETWEEN segments
    ("B2-505;/", 1),          # dangling break -- nothing follows it
    ("/;80-120", 1),          # leading break -- nothing precedes it
    ("B2-505", 1),            # no break at all
    ("A1-50;/;20-30;/;40-60", 3),
])
def test_chain_count_matches_protpardelles_own(contig, chains):
    """Protpardelle counts SEGMENTS, not separators. Our validator counted
    `contig.count("/") + 1`, so `B2-505;/` read as two chains and a
    `total_lengths` of two entries was accepted -- then Protpardelle refused it
    from inside its own sampler:

        AssertionError: Contig B2-505;/ has 1 chains but length ranges
        specify 2 chains.

    A live round sent exactly that. The count has to agree with the engine's,
    or our validation passes malformed input through to an assertion.
    """
    from protein_design_mcp.adapters.protpardelle import contig_chain_count

    assert contig_chain_count(contig) == chains


def test_a_dangling_break_with_mismatched_lengths_is_refused(tmp_path):
    from protein_design_mcp.adapters import protpardelle
    from protein_design_mcp.validation import ToolInputError

    target = tmp_path / "t.pdb"
    target.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n")
    with pytest.raises((ToolInputError, ValueError)) as excinfo:
        protpardelle.build_args(None, {
            "target_pdb": str(target), "contig": "B2-505;/",
            "total_lengths": [[504, 504], [80, 120]], "hotspots": None,
            "model": "cc83", "step_scale": 1.2, "schurn": 0.0,
            "crop_cond_start": 0.0, "translation": [0.0, 0.0, 0.0],
            "num_samples": 1, "batch_size": 1,
        })
    assert "1" in str(excinfo.value) and "2" in str(excinfo.value)


# --- a chain break needs a segment on both sides -----------------------------
#
# Settled by a live matrix against the engine (2026-09-25), not by reading one
# of its parsers:
#
#   contig            total_lengths          result
#   B2-505;80-120     [[584,624]]            PASS -- ONE fused 613-mer
#   B2-505;/;80-120   [[504,504],[80,120]]   PASS -- two chains, A:504 B:108
#   B2-505;/          [[504,504]]            int('/') ValueError
#
# So "/" IS the chain separator and is REQUIRED for a binder: without it the
# design comes back fused to the target, exactly like an RFdiffusion3 contig
# missing its "/0". `contig_to_motif_placement`'s docstring shows a grammar
# with no "/" because it is one of several paths, not the whole grammar --
# reading it as the whole grammar is what made the previous analysis wrong.
#
# The one rule still missing: a "/" with nothing on one side of it reaches the
# engine's scaffold branch, which calls int("/").


@pytest.mark.parametrize("contig", ["B2-505;/", "/;B2-505", "B2-505;/;", "/"])
def test_a_chain_break_without_a_segment_on_both_sides_is_refused(contig, tmp_path):
    from protein_design_mcp.adapters import protpardelle
    from protein_design_mcp.validation import ToolInputError

    target = tmp_path / "t.pdb"
    target.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n")
    with pytest.raises((ToolInputError, ValueError)) as excinfo:
        protpardelle.build_args(None, {
            "target_pdb": str(target), "contig": contig,
            "total_lengths": [[80, 120]], "hotspots": None, "model": "cc83",
            "step_scale": 1.2, "schurn": 0.0, "crop_cond_start": 0.0,
            "translation": [0.0, 0.0, 0.0], "num_samples": 1, "batch_size": 1,
        })
    assert "/" in str(excinfo.value)


def test_a_break_between_two_segments_is_accepted(tmp_path):
    """The shape that produced two chains live."""
    from protein_design_mcp.adapters import protpardelle

    target = tmp_path / "t.pdb"
    target.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n")
    args = protpardelle.build_args(None, {
        "target_pdb": str(target), "contig": "B2-505;/;80-120",
        "total_lengths": [[504, 504], [80, 120]], "hotspots": None,
        "model": "cc83", "step_scale": 1.2, "schurn": 0.0,
        "crop_cond_start": 0.0, "translation": [0.0, 0.0, 0.0],
        "num_samples": 1, "batch_size": 1,
    })
    assert "B2-505;/;80-120" in args


def test_a_contig_with_no_break_at_all_is_still_accepted(tmp_path):
    """It runs -- it just fuses the chains, which is the caller's call to make
    and is what the parameter description now warns about."""
    from protein_design_mcp.adapters import protpardelle

    target = tmp_path / "t.pdb"
    target.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n")
    args = protpardelle.build_args(None, {
        "target_pdb": str(target), "contig": "B2-505;80-120",
        "total_lengths": [[584, 624]], "hotspots": None, "model": "cc83",
        "step_scale": 1.2, "schurn": 0.0, "crop_cond_start": 0.0,
        "translation": [0.0, 0.0, 0.0], "num_samples": 1, "batch_size": 1,
    })
    assert "B2-505;80-120" in args
