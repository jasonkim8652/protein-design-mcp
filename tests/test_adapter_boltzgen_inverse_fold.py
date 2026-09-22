from pathlib import Path

import pytest

from protein_design_mcp.adapters.boltzgen_inverse_fold import build_args, parse_output
from protein_design_mcp.app import manifest_dir
from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.loader import load_manifests
from protein_design_mcp.validation import ToolInputError, validate_and_fill

MANIFEST_DIR = manifest_dir()


def _manifest():
    return next(
        m for m in load_manifests(MANIFEST_DIR) if m.name == "run_boltzgen_inverse_fold"
    )


def _base_params(**overrides):
    return validate_and_fill(_manifest(), {"design_spec": "design.yaml", **overrides})


# --- manifest shape ---


def test_manifest_loads_and_is_gpu_sequence_design():
    m = _manifest()
    assert m.category == "sequence_design"
    assert m.composite is False
    assert m.requires.gpu is True


def test_manifest_uses_prefix_not_env():
    engine = _manifest().engine
    assert engine.prefix == "/home/jk661/miniforge3/envs/boltzgen"
    assert engine.env is None
    assert engine.entry == ("boltzgen", "run")


def test_manifest_names_the_engine_not_proteinmpnn():
    """WAVE-COMMON's own M3 example: a variant enum with no engine name is
    unfindable. This tool must say BoltzGen's own head, not ProteinMPNN."""
    text = (_manifest().summary + _manifest().doc).lower()
    assert "not proteinmpnn" in text or "not protein" in text
    assert "boltzgen1_ifold" in text or "own inverse-folding head" in text


def test_design_spec_is_required():
    assert _manifest().schema["design_spec"]["required"] is True


def test_avoid_residues_defaults_empty_not_tied_to_protocol():
    spec = _manifest().schema["avoid_residues"]
    assert spec["default"] == ""


# --- validation corner cases ---


def test_validation_fills_defaults():
    params = _base_params()
    assert params["inverse_fold_num_sequences"] == 1
    assert params["avoid_residues"] == ""
    assert params["use_kernels"] == "auto"


def test_validation_rejects_missing_design_spec():
    with pytest.raises(ToolInputError, match="design_spec"):
        validate_and_fill(_manifest(), {})


def test_validation_accepts_boundary_num_sequences():
    params = _base_params(inverse_fold_num_sequences=1)
    assert params["inverse_fold_num_sequences"] == 1
    with pytest.raises(ToolInputError):
        _base_params(inverse_fold_num_sequences=0)


def test_validation_rejects_lowercase_avoid_residues():
    with pytest.raises(ToolInputError):
        _base_params(avoid_residues="c")


def test_validation_accepts_multi_residue_avoid_string():
    params = _base_params(avoid_residues="CM")
    assert params["avoid_residues"] == "CM"


# --- build_args ---


def test_build_args_wraps_boltzgen_run_with_only_inverse_fold():
    params = _base_params()
    args = build_args(_manifest(), params)
    assert args[0] == str(Path("design.yaml"))
    assert "--steps" in args
    assert args[args.index("--steps") + 1] == "inverse_folding"
    assert "--only_inverse_fold" in args


def test_build_args_never_passes_protocol_flag():
    params = _base_params()
    args = build_args(_manifest(), params)
    assert "--protocol" not in args


def test_build_args_always_passes_inverse_fold_avoid_explicitly():
    """Even empty -- deterministic regardless of BoltzGen's own
    protocol-based default (see manifest doc)."""
    params = _base_params()
    args = build_args(_manifest(), params)
    assert "--inverse_fold_avoid" in args
    assert args[args.index("--inverse_fold_avoid") + 1] == ""


def test_build_args_passes_num_sequences_and_checkpoint():
    params = _base_params(inverse_fold_num_sequences=4)
    args = build_args(_manifest(), params)
    assert args[args.index("--inverse_fold_num_sequences") + 1] == "4"
    assert "--inverse_fold_checkpoint" in args


# --- parse_output (reuses the same real-mmCIF-writer helper pattern) ---


def _write_cif(path: Path, chains: dict[str, str]) -> None:
    from Bio.PDB import MMCIFIO
    from Bio.PDB.Atom import Atom
    from Bio.PDB.Chain import Chain as BioChain
    from Bio.PDB.Model import Model
    from Bio.PDB.Residue import Residue as BioResidue
    from Bio.PDB.Structure import Structure as BioStructure

    three = {
        "A": "ALA", "G": "GLY", "L": "LEU", "V": "VAL", "I": "ILE",
        "N": "ASN", "K": "LYS", "S": "SER", "E": "GLU", "D": "ASP",
    }
    structure = BioStructure("test")
    model = Model(0)
    structure.add(model)
    for chain_id, seq in chains.items():
        bio_chain = BioChain(chain_id)
        model.add(bio_chain)
        for res_idx, aa in enumerate(seq, start=1):
            resname = three.get(aa, "ALA")
            residue = BioResidue((" ", res_idx, " "), resname, "")
            bio_chain.add(residue)
            atom = Atom(
                name="CA", coord=[float(res_idx), 0.0, 0.0], bfactor=0.0,
                occupancy=1.0, altloc=" ", fullname=" CA ", serial_number=res_idx,
                element="C",
            )
            residue.add(atom)

    io = MMCIFIO()
    io.set_structure(structure)
    io.save(str(path))


def test_parse_output_reports_every_chain_per_sequence(tmp_path):
    cif0 = tmp_path / "spec_0.cif"
    cif1 = tmp_path / "spec_1.cif"
    _write_cif(cif0, {"A": "LALVL", "B": "GGSN"})
    _write_cif(cif1, {"A": "GAVIK", "B": "GGSN"})
    run = CompletedRun(
        returncode=0, stdout="", stderr="", workdir=tmp_path,
        outputs={"inverse_folded_designs": [str(cif0), str(cif1)]},
    )
    result = parse_output(_manifest(), run)
    assert result["num_designs"] == 2
    first = result["designs"][0]
    assert first["id"] == "spec_0"
    chain_a = next(c for c in first["chains"] if c["chain_id"] == "A")
    assert chain_a["sequence"] == "LALVL"
    chain_b = next(c for c in first["chains"] if c["chain_id"] == "B")
    assert chain_b["sequence"] == "GGSN"


def test_parse_output_handles_single_sequence(tmp_path):
    cif = tmp_path / "spec_0.cif"
    _write_cif(cif, {"A": "LALVL"})
    run = CompletedRun(
        returncode=0, stdout="", stderr="", workdir=tmp_path,
        outputs={"inverse_folded_designs": str(cif)},
    )
    result = parse_output(_manifest(), run)
    assert result["num_designs"] == 1


def test_parse_output_raises_when_outputs_missing(tmp_path):
    run = CompletedRun(returncode=0, stdout="", stderr="", workdir=tmp_path, outputs={})
    with pytest.raises(ValueError, match="inverse_folded_designs"):
        parse_output(_manifest(), run)


def test_parse_output_ignores_npz_metadata_mixed_into_outputs(tmp_path):
    """inverse_folded_designs' pattern now collects BOTH .cif and .npz (see
    the manifest) so a downstream run_boltzgen_fold/analyze call can be
    handed the combined list directly."""
    cif = tmp_path / "spec_0.cif"
    npz = tmp_path / "spec_0.npz"
    _write_cif(cif, {"A": "LALVL"})
    npz.write_bytes(b"\x00not a cif")
    run = CompletedRun(
        returncode=0, stdout="", stderr="", workdir=tmp_path,
        outputs={"inverse_folded_designs": [str(cif), str(npz)]},
    )
    result = parse_output(_manifest(), run)
    assert result["num_designs"] == 1


def test_manifest_inverse_folded_designs_pattern_also_collects_npz():
    pattern = _manifest().outputs[0].pattern
    assert pattern.endswith(".[cn][ip][fz]")
