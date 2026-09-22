from pathlib import Path

import pytest

from protein_design_mcp.adapters.boltzgen_design import build_args, parse_output
from protein_design_mcp.app import manifest_dir
from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.loader import load_manifests
from protein_design_mcp.validation import ToolInputError, validate_and_fill

MANIFEST_DIR = manifest_dir()


def _manifest():
    return next(m for m in load_manifests(MANIFEST_DIR) if m.name == "run_boltzgen_design")


def _base_params(**overrides):
    return validate_and_fill(_manifest(), {"design_spec": "design.yaml", **overrides})


# --- manifest shape ---


def test_manifest_loads_and_is_gpu_binder_generation():
    m = _manifest()
    assert m.category == "binder_generation"
    assert m.composite is False
    assert m.requires.gpu is True


def test_manifest_uses_prefix_not_env():
    engine = _manifest().engine
    assert engine.prefix == "/home/jk661/miniforge3/envs/boltzgen"
    assert engine.env is None
    assert engine.entry == ("boltzgen", "run")


def test_manifest_documents_protocol_has_no_effect():
    """Empirically verified (boltzgen configure --steps design under all six
    protocols -> byte-identical design.yaml) -- protocol must not be a
    schema parameter, and the doc must say why."""
    schema = _manifest().schema
    assert "protocol" not in schema
    assert "verified" in _manifest().doc.lower()


def test_design_spec_is_required():
    assert _manifest().schema["design_spec"]["required"] is True


def test_num_designs_defaults_smaller_than_boltzgens_own_cli_default():
    spec = _manifest().schema["num_designs"]
    assert spec["default"] == 10
    assert spec["default"] < 10000


# --- validation corner cases ---


def test_validation_fills_defaults():
    params = _base_params()
    assert params["num_designs"] == 10
    assert params["use_kernels"] == "auto"
    assert params["design_checkpoints"] == [
        "huggingface:boltzgen/boltzgen-1:boltzgen1_diverse.ckpt",
        "huggingface:boltzgen/boltzgen-1:boltzgen1_adherence.ckpt",
    ]


def test_validation_rejects_missing_design_spec():
    with pytest.raises(ToolInputError, match="design_spec"):
        validate_and_fill(_manifest(), {})


def test_validation_accepts_boundary_num_designs():
    params = _base_params(num_designs=1)
    assert params["num_designs"] == 1
    with pytest.raises(ToolInputError):
        _base_params(num_designs=0)


def test_diffusion_batch_size_has_no_static_default():
    """Left unset, BoltzGen's own auto heuristic (1 if num_designs<100 else
    10) applies -- so this key must be genuinely absent, not defaulted to
    something that would override that heuristic."""
    params = _base_params()
    assert "diffusion_batch_size" not in params


# --- build_args ---


def test_build_args_wraps_boltzgen_run_with_design_step_only():
    params = _base_params()
    args = build_args(_manifest(), params)
    assert args[0] == str(Path("design.yaml"))
    assert "--steps" in args
    assert args[args.index("--steps") + 1] == "design"
    assert "--output" in args
    assert args[args.index("--output") + 1] == "."


def test_build_args_never_passes_protocol_flag():
    """No --protocol at all -- this tool hardcodes nothing and exposes
    nothing, since it has no effect on the design step (see manifest doc)."""
    params = _base_params()
    args = build_args(_manifest(), params)
    assert "--protocol" not in args


def test_build_args_omits_diffusion_batch_size_when_unset():
    params = _base_params()
    args = build_args(_manifest(), params)
    assert "--diffusion_batch_size" not in args


def test_build_args_includes_diffusion_batch_size_when_set():
    params = _base_params(diffusion_batch_size=5)
    args = build_args(_manifest(), params)
    assert "--diffusion_batch_size" in args
    assert args[args.index("--diffusion_batch_size") + 1] == "5"


def test_build_args_omits_step_scale_and_noise_scale_when_unset():
    params = _base_params()
    args = build_args(_manifest(), params)
    assert "--step_scale" not in args
    assert "--noise_scale" not in args


def test_build_args_includes_num_designs_and_checkpoints():
    params = _base_params(num_designs=5)
    args = build_args(_manifest(), params)
    assert args[args.index("--num_designs") + 1] == "5"
    assert "--design_checkpoints" in args


# --- parse_output ---


def _write_cif(path: Path, chains: dict[str, str]) -> None:
    """Write a minimal multi-chain mmCIF via Bio.PDB's own writer, so it is
    guaranteed to carry every field Bio.PDB's own MMCIFParser expects back."""
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


def test_parse_output_reports_every_chain_per_design(tmp_path):
    cif0 = tmp_path / "design_spec_0.cif"
    cif1 = tmp_path / "design_spec_1.cif"
    _write_cif(cif0, {"A": "AALVL", "B": "GGSN"})
    _write_cif(cif1, {"A": "AGIVK", "B": "GGSN"})
    run = CompletedRun(
        returncode=0, stdout="", stderr="", workdir=tmp_path,
        outputs={"generated_designs": [str(cif0), str(cif1)]},
    )
    result = parse_output(_manifest(), run)
    assert result["num_designs"] == 2
    assert len(result["designs"]) == 2
    first = result["designs"][0]
    assert first["id"] == "design_spec_0"
    chain_ids = {c["chain_id"] for c in first["chains"]}
    assert chain_ids == {"A", "B"}
    chain_a = next(c for c in first["chains"] if c["chain_id"] == "A")
    assert chain_a["sequence"] == "AALVL"
    assert chain_a["length"] == 5


def test_parse_output_handles_single_design_no_suffix(tmp_path):
    """num_designs=1 -> BoltzGen writes '<stem>.cif', no '_0' suffix."""
    cif = tmp_path / "design_spec.cif"
    _write_cif(cif, {"A": "AALVL"})
    run = CompletedRun(
        returncode=0, stdout="", stderr="", workdir=tmp_path,
        outputs={"generated_designs": str(cif)},
    )
    result = parse_output(_manifest(), run)
    assert result["num_designs"] == 1
    assert result["designs"][0]["id"] == "design_spec"


def test_parse_output_raises_when_generated_designs_missing(tmp_path):
    run = CompletedRun(returncode=0, stdout="", stderr="", workdir=tmp_path, outputs={})
    with pytest.raises(ValueError, match="generated_designs"):
        parse_output(_manifest(), run)
