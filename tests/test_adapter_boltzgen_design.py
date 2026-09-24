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
    return validate_and_fill(_manifest(), {"target_structure": "target.cif", "target_chains": ["A"], **overrides})


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
    # This line used to read ("boltzgen", "run"), which pinned the BUG rather
    # than the requirement: the adapter's argv is written for the wrapper, and
    # sending it to BoltzGen made every call fail with "unrecognized
    # arguments". The wrapper builds the design spec and runs `boltzgen run`.
    assert engine.entry == ("python", "/app/scripts/engines/boltzgen_design.py")


def test_manifest_documents_protocol_has_no_effect():
    """Empirically verified (boltzgen configure --steps design under all six
    protocols -> byte-identical design.yaml) -- protocol must not be a
    schema parameter, and the doc must say why."""
    schema = _manifest().schema
    assert "protocol" not in schema
    assert "verified" in _manifest().doc.lower()


def test_the_tool_builds_its_own_spec_from_parameters():
    """design_spec used to be required and nothing on this server produced
    one, so the run_boltzgen_* family was unreachable from a planned workflow.
    The spec's content is structured parameters, so the tool builds it."""
    schema = _manifest().schema
    assert "design_spec" not in schema, (
        "a raw spec file alongside the parameters would need 'exactly one of' "
        "validation this schema cannot express, and leaves two ways to say one thing"
    )
    for field in ("target_structure", "target_chains",
                  "binder_length_min", "binder_length_max", "binder_chain_id"):
        assert field in schema, field


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


def test_validation_still_demands_something_to_design_against():
    """Making every field optional would let a call through that names neither
    a target nor a spec, and fail inside the engine instead of at the boundary
    where the message can name the missing parameter."""
    with pytest.raises(ToolInputError, match="target_structure"):
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
    # The spec path is no longer argv[0]: the wrapper writes the spec and puts
    # it there itself, and everything after --passthrough is what it hands on.
    assert "--passthrough" in args
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


def test_parse_output_ignores_npz_metadata_mixed_into_generated_designs(tmp_path):
    """generated_designs' pattern now collects BOTH .cif and .npz (see the
    manifest) so a downstream run_boltzgen_fold/analyze call can be handed
    the combined list directly -- parse_output must skip the .npz entries
    when building chain sequences, not try to parse them as structures."""
    cif = tmp_path / "design_spec_0.cif"
    npz = tmp_path / "design_spec_0.npz"
    _write_cif(cif, {"A": "AALVL"})
    npz.write_bytes(b"\x00not a cif")
    run = CompletedRun(
        returncode=0, stdout="", stderr="", workdir=tmp_path,
        outputs={"generated_designs": [str(cif), str(npz)]},
    )
    result = parse_output(_manifest(), run)
    assert result["num_designs"] == 1
    assert result["designs"][0]["id"] == "design_spec_0"


def test_manifest_generated_designs_pattern_also_collects_npz():
    """The metadata .npz beside each .cif is required by BoltzGen's own
    downstream fold/design_fold/analyze steps (data_from_generated.py reads
    both from the SAME design_dir) -- without it, run_boltzgen_fold has no
    path to this tool's output. See wave-E-report.md."""
    generated = next(o for o in _manifest().outputs if o.name == "generated_designs")
    assert generated.pattern.endswith(".[cn][ip][fz]")


# --- the manifest must invoke the wrapper, not boltzgen directly ------------


def test_the_entry_point_is_the_wrapper_that_understands_these_arguments():
    """`build_args` emits `--target-structure ... --passthrough <rest>`, which
    only `scripts/engines/boltzgen_design.py` parses: it builds BoltzGen's
    design-spec YAML from those parameters and then runs `boltzgen run <spec>`
    with whatever followed --passthrough.

    With `engine.entry: ["boltzgen", "run"]` the arguments went straight to
    BoltzGen, which has never heard of them, and every call died with

        boltzgen: error: unrecognized arguments: --target-structure
        --target-chains A --binder-length-min 15 ... --passthrough

    -- for live_proof's own arguments as much as for a model's, so this was
    broken for every caller, not a bad call.
    """
    import yaml
    from pathlib import Path

    manifest = yaml.safe_load(
        Path("src/protein_design_mcp/manifests/run_boltzgen_design.yaml").read_text())
    entry = manifest["engine"]["entry"]
    assert entry[0] == "python"
    assert entry[1].endswith("scripts/engines/boltzgen_design.py"), entry


def test_the_wrapper_accepts_every_flag_the_adapter_emits():
    """The two halves were written apart and drifted apart. Parse the adapter's
    real output with the wrapper's real parser so a rename on either side
    fails here instead of inside a container."""
    import argparse
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "bg_design_engine", Path("scripts/engines/boltzgen_design.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    parser = argparse.ArgumentParser()
    parser.add_argument("--design-spec", default=None)
    parser.add_argument("--target-structure")
    parser.add_argument("--target-chains", default="")
    parser.add_argument("--binder-length-min", type=int)
    parser.add_argument("--binder-length-max", type=int)
    parser.add_argument("--binder-chain-id", default="C")
    parser.add_argument("--passthrough", nargs=argparse.REMAINDER, default=[])

    args = build_args(None, {
        "target_structure": "/t.pdb",
        "target_chains": ["A"],
        "binder_length_min": 15,
        "binder_length_max": 20,
        "binder_chain_id": "C",
        "num_designs": 1,
        "diffusion_batch_size": 1,
        "design_checkpoints": ["ckpt"],
        "step_scale": 1.5,
        "noise_scale": 1.0,
        "use_kernels": False,
        "moldir": "/moldir",
        "num_workers": 0,
    })
    parsed = parser.parse_args(args)
    assert parsed.target_structure == "/t.pdb"
    assert parsed.binder_length_min == 15
    assert parsed.passthrough, "everything after --passthrough goes to boltzgen run"
