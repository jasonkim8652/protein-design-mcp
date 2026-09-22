"""Adapter for MultiFlow's unconditional + codesign sampler (`run_multiflow`).

Builds Hydra command-line overrides directly, forwarded as-is by
``scripts/engines/multiflow.py`` (which also handles the self-consistency
score collection and its crash-tolerance fallback -- see that script's
docstring).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.schema import Manifest

_CKPT_PATH = "/home/jk661/projects/multiflow/weights/last_gpu0.ckpt"
_PREDICT_DIR = "predict_out"  # must match scripts/engines/multiflow.py's _PREDICT_DIR


def _hydra_bool(value: bool) -> str:
    return "true" if value else "false"


def build_args(manifest: Manifest, params: dict[str, Any]) -> list[str]:
    """Translate validated parameters into Hydra overrides."""
    del manifest
    return [
        f"inference.unconditional_ckpt_path={_CKPT_PATH}",
        f"inference.predict_dir={_PREDICT_DIR}",
        # inference_unconditional.yaml's own default `pmpnn_dir: ./ProteinMPNN`
        # is cwd-relative -- confirmed live to fail with
        # `can't open file '.../ProteinMPNN/helper_scripts/
        # parse_multiple_chains.py'` once cwd is a scratch workdir rather
        # than the multiflow checkout itself.
        "inference.pmpnn_dir=/home/jk661/projects/multiflow/ProteinMPNN",
        "inference.folding.pmpnn_path=/home/jk661/projects/multiflow/ProteinMPNN/",
        f"inference.seed={params['seed']}",
        "inference.num_gpus=1",
        "inference.also_fold_pmpnn_seq=false",
        "inference.write_sample_trajectories=false",
        # inference_unconditional.yaml's own default sets
        # inference.samples.length_subset=[70,100,200,300] -- same
        # confirmed-live issue as FrameFlow's identical config field (see
        # run_frameflow's adapter); nulled explicitly so min/max_length
        # actually control the run.
        "inference.samples.length_subset=null",
        f"inference.samples.min_length={params['min_length']}",
        f"inference.samples.max_length={params['max_length']}",
        f"inference.samples.length_step={params['length_step']}",
        f"inference.samples.samples_per_length={params['samples_per_length']}",
        f"inference.interpolant.min_t={params['min_t']}",
        f"inference.interpolant.self_condition={_hydra_bool(params['self_condition'])}",
        f"inference.interpolant.sampling.num_timesteps={params['num_timesteps']}",
        f"inference.interpolant.sampling.do_sde={_hydra_bool(params['do_sde'])}",
        f"inference.interpolant.trans.sample_temp={params['trans_sample_temp']}",
        f"inference.interpolant.aatypes.temp={params['aatypes_temp']}",
        f"inference.interpolant.aatypes.noise={params['aatypes_noise']}",
        f"inference.interpolant.aatypes.do_purity={_hydra_bool(params['aatypes_do_purity'])}",
    ]


def _length_from_pdb(path: str) -> int:
    count = 0
    for line in Path(path).read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")) and line[12:16].strip() == "CA":
            count += 1
    return count


def _read_fasta_sequence(path: str) -> str | None:
    lines = [l.strip() for l in Path(path).read_text().splitlines() if l.strip()]
    seq_lines = [l for l in lines if not l.startswith(">")]
    return "".join(seq_lines) if seq_lines else None


def _sample_key(file_path: Path, parent_calls_to_sample_dir: int) -> tuple[str, str]:
    """(length_dir_name, sample_dir_name) -- identifies one generated sample.

    ``results.collect_outputs`` copies each declared output's matches into
    its OWN ``<spec.name>/`` subdirectory, each preserving its path relative
    to the workdir underneath -- so a backbone's ``sample.pdb`` and its
    codesigned FASTA no longer share a parent directory after collection
    (they live under sibling ``backbones/`` and ``codesign_sequences/``
    trees). Both trees still preserve the SAME
    ``.../length_<L>/sample_<i>/...`` structure relative to their own root,
    so pairing by that two-level name (not by any shared path) is what
    still works post-collection. ``sample_<i>`` alone is not unique --
    confirmed live, every length restarts its own sample numbering at 0.

    ``parent_calls_to_sample_dir`` is how many ``.parent`` calls from
    ``file_path`` itself reach the ``sample_<i>/`` directory: 1 for
    ``sample_<i>/sample.pdb``, 3 for
    ``sample_<i>/self_consistency/codesign_seqs/codesign.fa``.
    """
    sample_dir = file_path
    for _ in range(parent_calls_to_sample_dir):
        sample_dir = sample_dir.parent
    return (sample_dir.parent.name, sample_dir.name)


def parse_output(manifest: Manifest, run: CompletedRun) -> dict[str, Any]:
    """Pair each backbone with its codesigned sequence by (length, sample) key."""
    del manifest
    backbone_paths = run.outputs.get("backbones")
    if not backbone_paths:
        raise ValueError(
            "run_multiflow's declared 'backbones' output was not collected "
            f"-- no sample.pdb was found. run.outputs was: {run.outputs}"
        )
    if not isinstance(backbone_paths, list):
        backbone_paths = [backbone_paths]

    fasta_paths = run.outputs.get("codesign_sequences") or []
    if not isinstance(fasta_paths, list):
        fasta_paths = [fasta_paths]
    fasta_by_key = {
        _sample_key(Path(p), parent_calls_to_sample_dir=3): p for p in fasta_paths
    }

    # Always present (see scripts/engines/multiflow.py: written on every
    # successful call, {} in the defensive fallback) -- keyed by
    # "length_dir/sample_dir", the same string _sample_key's two-tuple
    # joins into for each sample's own "id" below.
    self_consistency_path = run.outputs.get("self_consistency_summary")
    self_consistency_by_id: dict[str, dict] = {}
    if self_consistency_path:
        self_consistency_by_id = json.loads(Path(self_consistency_path).read_text())

    samples = []
    for pdb_path in sorted(backbone_paths):
        key = _sample_key(Path(pdb_path), parent_calls_to_sample_dir=1)
        sample_id = f"{key[0]}/{key[1]}"
        fasta_path = fasta_by_key.get(key)
        sequence = _read_fasta_sequence(fasta_path) if fasta_path else None
        samples.append(
            {
                "id": sample_id,
                "length": _length_from_pdb(pdb_path),
                "codesign_sequence": sequence,
                "self_consistency": self_consistency_by_id.get(sample_id),
            }
        )

    return {"samples": samples, "num_samples": len(samples)}
