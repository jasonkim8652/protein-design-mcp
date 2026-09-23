"""Adapter for Proteina-Complexa's ``generate`` pipeline step
(``run_proteina_complexa_generate``).

Wraps ``complexa generate <config> ++overrides... --verbose``. There is no
caller-supplied config path or raw-override passthrough here -- see the
manifest's doc for why (Hydra ``_target_:`` class instantiation makes an
arbitrary config file or override string a code-execution vector). Every
override below is derived from one specific, individually-validated schema
field.

``--verbose`` matters more here than for a typical CLI wrap: Proteina-
Complexa's own ``cli_runner.run_step`` only streams the underlying
``python -m proteinfoundation.generate`` subprocess's output to OUR
captured stdout/stderr when ``--verbose`` is passed; without it, output goes
to a log file inside the engine's own process that this dispatcher never
sees, and a failure would surface as a bare non-zero exit with no
diagnostic text at all.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.schema import Manifest

# Fixed, vetted config this tool always runs against -- never caller
# supplied (see the manifest doc's "No caller-supplied config file" section).
_CONFIG_PATH = "/home/jk661/projects/proteina-complexa/configs/search_binder_local_pipeline.yaml"

_REPO_ROOT = "/home/jk661/projects/proteina-complexa"

# run_name is pure directory-naming plumbing for this tool (it does not
# change what is generated) -- fixed so this adapter's own output glob
# pattern stays predictable, rather than exposed as a caller parameter.
_RUN_NAME = "mcp_run"

# Checkpoint paths this engine's own config declares as CWD-RELATIVE
# ("./ckpts", "./ckpts/complexa_ae.ckpt") -- correct when a caller runs
# `complexa` from the repo root by hand, but this tool's subprocess runs
# with cwd set to a disposable scratch directory (dispatch.env.EnvDispatcher),
# where the relative form resolves to nothing. Made absolute here rather
# than left for the caller to override, since there is only ever one right
# value on this host.
_CKPT_PATH = f"{_REPO_ROOT}/ckpts"
_AUTOENCODER_CKPT_PATH = f"{_REPO_ROOT}/ckpts/complexa_ae.ckpt"

# Every catalog target's own `target_path` in configs/targets/targets_dict.yaml
# (read live, 2026-09-22) is ALSO cwd-relative for the same reason, and is
# used directly (verified in datasets/gen_dataset.py's TargetFeatures.__init__:
# `os.path.exists(pdb_path)` on the literal string, no repo-root prefixing
# anywhere in that path). Overriding generation.task_name alone is not
# enough to fix this -- the resolved pdb_path has to be overridden too, so
# this tool carries the full catalog's own path mapping (not re-derived at
# call time) and rewrites it to absolute before handing it to complexa.
_TARGET_PDB_PATHS = {
    "01_PD1": "assets/target_data/bindcraft_targets/PD1.pdb",
    "02_PDL1": "assets/target_data/bindcraft_targets/PD-L1.pdb",
    "03_PDL1_AAV": "assets/target_data/bindcraft_targets/PD-L1.pdb",
    "04_IFNAR2": "assets/target_data/bindcraft_targets/IFNAR2.pdb",
    "05_CD45": "assets/target_data/bindcraft_targets/CD45.pdb",
    "06_CD45": "assets/target_data/bindcraft_targets/CD45.pdb",
    "07_CD45": "assets/target_data/bindcraft_targets/CD45.pdb",
    "08_CD45": "assets/target_data/bindcraft_targets/CD45.pdb",
    "09_CD45": "assets/target_data/bindcraft_targets/CD45.pdb",
    "10_CD45": "assets/target_data/bindcraft_targets/CD45.pdb",
    "11_CD45": "assets/target_data/bindcraft_targets/CD45.pdb",
    "12_Claudin1": "assets/target_data/bindcraft_targets/CLDN1.pdb",
    "13_BBF14": "assets/target_data/bindcraft_targets/BBF-14.pdb",
    "14_CrSAS6": "assets/target_data/bindcraft_targets/Sas6.pdb",
    "15_DerF7": "assets/target_data/bindcraft_targets/DerF7.pdb",
    "16_DerF7": "assets/target_data/bindcraft_targets/DerF7.pdb",
    "17_DerF7": "assets/target_data/bindcraft_targets/DerF7.pdb",
    "18_DerF21": "assets/target_data/bindcraft_targets/DerF21.pdb",
    "19_DerF21": "assets/target_data/bindcraft_targets/DerF21.pdb",
    "20_DerF21": "assets/target_data/bindcraft_targets/DerF21.pdb",
    "21_DerF21": "assets/target_data/bindcraft_targets/DerF21.pdb",
    "22_DerF21": "assets/target_data/bindcraft_targets/DerF21.pdb",
    "23_BetV1": "assets/target_data/bindcraft_targets/BetV1.pdb",
    "24_SpCas9": "assets/target_data/bindcraft_targets/sCas9_cropped.pdb",
    "25_CbAgo": "assets/target_data/bindcraft_targets/CbAgo_cropped_NPIWI.pdb",
    "26_CbAgo": "assets/target_data/bindcraft_targets/CbAgo_cropped_PAZ.pdb",
    "27_HER2_AAV": "assets/target_data/bindcraft_targets/HER2_cropped.pdb",
    "28_HER2_AAV": "assets/target_data/bindcraft_targets/HER2_cropped.pdb",
    "29_BHRF1": "assets/target_data/alpha_proteo_targets/2wh6_cropped.pdb",
    "30_SC2RBD": "assets/target_data/alpha_proteo_targets/6m0j_cropped.pdb",
    "31_IL7RA": "assets/target_data/alpha_proteo_targets/3di3_cropped.pdb",
    "31_IL7RA_FIX": "assets/target_data/alpha_proteo_targets/3di3_cropped_fixed.pdb",
    "31_IL7RA_REPACK": "assets/target_data/alpha_proteo_targets/3di3_repacked.pdb",
    "32_PDL1_ALPHA": "assets/target_data/alpha_proteo_targets/5o45_cropped.pdb",
    "32_PDL1_ALPHA_FIX": "assets/target_data/alpha_proteo_targets/5o45_cropped_fixed.pdb",
    "32_PDL1_ALPHA_REPACK": "assets/target_data/alpha_proteo_targets/5o45_repacked.pdb",
    "33_TrkA": "assets/target_data/alpha_proteo_targets/1www_cropped.pdb",
    "34_Insulin": "assets/target_data/alpha_proteo_targets/4zxb_cropped.pdb",
    "35_H1": "assets/target_data/alpha_proteo_targets/5vli_cropped_fixed.pdb",
    "36_VEGFA": "assets/target_data/alpha_proteo_targets/1bj1_cropped.pdb",
    "37_IL17A": "assets/target_data/alpha_proteo_targets/4hsa_cropped.pdb",
    "38_TNFalpha": "assets/target_data/alpha_proteo_targets/1tnf_cropped.pdb",
    "38_TNFalpha_FIX": "assets/target_data/alpha_proteo_targets/1tnf_cropped_fixed.pdb",
    "38_TNFalpha_REPACK": "assets/target_data/alpha_proteo_targets/1tnf_repacked.pdb",
}


def _omega_scalar(value: Any) -> str:
    """Render one Python value as an OmegaConf/Hydra CLI override scalar.

    Same convention ``adapters.boltzgen_filter._omega_scalar`` already
    verified live against OmegaConf's own round-trip: ``null`` for None,
    lowercase ``true``/``false`` for bool, a bare number for int/float, a
    JSON-quoted string otherwise.
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value))


def _omega_list(values: list[Any]) -> str:
    return "[" + ",".join(_omega_scalar(v) for v in values) + "]"


def _omega_mapping(mapping: dict[str, Any]) -> str:
    items = ",".join(f"{key}:{_omega_scalar(val)}" for key, val in mapping.items())
    return "{" + items + "}"


def build_args(manifest: Manifest, params: dict[str, Any]) -> list[str]:
    """Translate validated parameters into ``complexa generate``'s argv.

    ``manifest`` is unused here -- part of every adapter's signature, see
    ``protein_design_mcp.adapters.boltz`` for why.
    """
    del manifest

    task_name = params["task_name"]
    target_pdb_path = _TARGET_PDB_PATHS.get(task_name)
    if target_pdb_path is None:
        raise ValueError(
            f"task_name {task_name!r} has no entry in this adapter's own "
            "_TARGET_PDB_PATHS -- the manifest's enum and this table have "
            "drifted apart; regenerate the table from "
            "configs/targets/targets_dict.yaml."
        )

    overrides = [
        f"++generation.task_name={_omega_scalar(task_name)}",
        f"++run_name={_omega_scalar(_RUN_NAME)}",
        f"++ckpt_path={_omega_scalar(_CKPT_PATH)}",
        f"++autoencoder_ckpt_path={_omega_scalar(_AUTOENCODER_CKPT_PATH)}",
        "++generation.dataloader.dataset.conditional_features.0.pdb_path="
        f"{_omega_scalar(f'{_REPO_ROOT}/{target_pdb_path}')}",
        # Job splitting is never exposed -- see the manifest doc's "GPU job
        # splitting is intentionally not exposed" section. Hardcoding this
        # keeps a future config change from silently reintroducing it.
        "++gen_njobs=1",
        f"++seed={_omega_scalar(params['seed'])}",
        # --- search ---
        f"++generation.search.algorithm={_omega_scalar(params['search_algorithm'])}",
        f"++generation.search.max_batch_size={_omega_scalar(params['max_batch_size'])}",
        f"++generation.search.reward_threshold={_omega_scalar(params.get('search_reward_threshold'))}",
        f"++generation.search.step_checkpoints={_omega_list(params['step_checkpoints'])}",
        f"++generation.search.best_of_n.replicas={_omega_scalar(params['best_of_n_replicas'])}",
        f"++generation.search.beam_search.n_branch={_omega_scalar(params['beam_search_n_branch'])}",
        f"++generation.search.beam_search.beam_width={_omega_scalar(params['beam_search_beam_width'])}",
        "++generation.search.beam_search.keep_lookahead_samples="
        f"{_omega_scalar(params['beam_search_keep_lookahead_samples'])}",
        "++generation.search.beam_search.save_intermediate_states="
        f"{_omega_scalar(params['beam_search_save_intermediate_states'])}",
        f"++generation.search.fk_steering.n_branch={_omega_scalar(params['fk_steering_n_branch'])}",
        f"++generation.search.fk_steering.beam_width={_omega_scalar(params['fk_steering_beam_width'])}",
        f"++generation.search.fk_steering.temperature={_omega_scalar(params['fk_steering_temperature'])}",
        "++generation.search.fk_steering.keep_lookahead_samples="
        f"{_omega_scalar(params['fk_steering_keep_lookahead_samples'])}",
        f"++generation.search.mcts.n_simulations={_omega_scalar(params['mcts_n_simulations'])}",
        f"++generation.search.mcts.exploration_prob={_omega_scalar(params['mcts_exploration_prob'])}",
        f"++generation.search.mcts.exploration_constant={_omega_scalar(params['mcts_exploration_constant'])}",
        f"++generation.search.mcts.keep_lookahead_samples={_omega_scalar(params['mcts_keep_lookahead_samples'])}",
        # --- sampling / diffusion ---
        f"++generation.dataloader.dataset.nres.nsamples={_omega_scalar(params['num_lengths'])}",
        f"++generation.dataloader.dataset.nrepeat_per_sample={_omega_scalar(params['nrepeat_per_sample'])}",
        f"++generation.dataloader.batch_size={_omega_scalar(params['batch_size'])}",
        f"++generation.args.nsteps={_omega_scalar(params['nsteps'])}",
        f"++generation.args.self_cond={_omega_scalar(params['self_cond'])}",
        f"++generation.args.guidance_w={_omega_scalar(params['guidance_w'])}",
        f"++generation.args.ag_ratio={_omega_scalar(params['ag_ratio'])}",
        f"++generation.args.ag_ckpt_path={_omega_scalar(params.get('ag_ckpt_path'))}",
        f"++generation.args.save_trajectory_every={_omega_scalar(params['save_trajectory_every'])}",
        # --- refinement ---
        "++generation.refinement.algorithm="
        f"{_omega_scalar(None if params['refinement_algorithm'] == 'none' else params['refinement_algorithm'])}",
        f"++generation.refinement.refine_targets={_omega_scalar(params['refine_targets'])}",
        f"++generation.refinement.save_pre_refinement={_omega_scalar(params['save_pre_refinement'])}",
        f"++generation.refinement.enable_soft_optimization={_omega_scalar(params['enable_soft_optimization'])}",
        f"++generation.refinement.enable_greedy_optimization={_omega_scalar(params['enable_greedy_optimization'])}",
        f"++generation.refinement.n_temp_iters={_omega_scalar(params['n_temp_iters'])}",
        f"++generation.refinement.n_hard_iters={_omega_scalar(params['n_hard_iters'])}",
        f"++generation.refinement.n_recycles={_omega_scalar(params['n_recycles'])}",
        f"++generation.refinement.n_greedy_iters={_omega_scalar(params['n_greedy_iters'])}",
        f"++generation.refinement.greedy_percentage={_omega_scalar(params['greedy_percentage'])}",
        # --- reward model (af2folding) ---
        "++generation.reward_model.reward_models.af2folding.num_recycles="
        f"{_omega_scalar(params['reward_num_recycles'])}",
        "++generation.reward_model.reward_models.af2folding.use_initial_guess="
        f"{_omega_scalar(params['reward_use_initial_guess'])}",
        "++generation.reward_model.reward_models.af2folding.use_initial_atom_pos="
        f"{_omega_scalar(params['reward_use_initial_atom_pos'])}",
        "++generation.reward_model.reward_models.af2folding.model_nums="
        f"{_omega_list(params['reward_model_nums']) if params.get('reward_model_nums') is not None else 'null'}",
    ]

    binder_length_min = params.get("binder_length_min")
    if binder_length_min is not None:
        overrides.append(f"++generation.dataloader.dataset.nres.low={_omega_scalar(binder_length_min)}")
    binder_length_max = params.get("binder_length_max")
    if binder_length_max is not None:
        overrides.append(f"++generation.dataloader.dataset.nres.high={_omega_scalar(binder_length_max)}")

    refinement_loss_weights = params.get("refinement_loss_weights")
    if refinement_loss_weights:
        if not isinstance(refinement_loss_weights, dict):
            raise ValueError("refinement_loss_weights must be an object mapping term name to a number.")
        for term, weight in refinement_loss_weights.items():
            overrides.append(f"++generation.refinement.loss_weights.{term}={_omega_scalar(weight)}")

    reward_weights = params.get("reward_weights") or {}
    if not isinstance(reward_weights, dict):
        raise ValueError("reward_weights must be an object mapping component name to a number.")
    for component, weight in reward_weights.items():
        overrides.append(
            "++generation.reward_model.reward_models.af2folding.reward_weights."
            f"{component}={_omega_scalar(weight)}"
        )

    return [_CONFIG_PATH, *overrides, "--verbose"]


def _to_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _to_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


_RESERVED_COLUMNS = {"pdb_path", "pdb_index", "aatype", "sample_type", "metadata_tag", "job_id"}


def parse_output(manifest: Manifest, run: CompletedRun) -> dict[str, Any]:
    """Read the rewards CSV ``generate.py`` itself wrote. ``manifest`` is
    unused (see ``build_args``).

    Every row becomes one entry in ``rewards``: the fixed columns
    (``pdb_path``, ``pdb_index``, ``total_reward``) plus every OTHER column
    in the CSV nested under ``reward_components`` -- the reward component
    set is not fixed (it depends on which reward_weights keys were
    supplied and non-default), so this reads the CSV's own header rather
    than assuming a column list.
    """
    del manifest
    csv_paths = run.outputs.get("rewards_csv")
    if not csv_paths:
        raise ValueError(
            "run_proteina_complexa_generate's declared 'rewards_csv' output "
            f"was not collected -- no rewards CSV was found. run.outputs was: {run.outputs}"
        )
    path = csv_paths[0] if isinstance(csv_paths, list) else csv_paths

    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    rewards = []
    for row in rows:
        components = {
            key: _to_float(value)
            for key, value in row.items()
            if key not in _RESERVED_COLUMNS and key != "total_reward"
        }
        rewards.append(
            {
                "pdb_path": row.get("pdb_path"),
                "pdb_index": _to_int(row.get("pdb_index")),
                "total_reward": _to_float(row.get("total_reward")),
                "reward_components": components,
            }
        )

    return {"rewards": rewards, "num_samples": len(rewards)}
