"""
PyRosetta wrapper for energy scoring, relaxation, interface analysis, and design.

PyRosetta provides Rosetta molecular modeling functionality through a
Python interface.  This runner lazily initializes PyRosetta on first use
and exposes four operations:

- **score**: Score a PDB using a Rosetta score function.
- **relax**: FastRelax a PDB to find a low-energy conformation.
- **interface_score**: Compute interface energy metrics for a complex.
- **design**: Fixed-backbone sequence design via PackRotamers.
"""

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from protein_design_mcp.exceptions import PyRosettaError

logger = logging.getLogger(__name__)

_PYROSETTA_INITIALIZED = False


def _ensure_init() -> None:
    """Lazily initialize PyRosetta once per process."""
    global _PYROSETTA_INITIALIZED
    if _PYROSETTA_INITIALIZED:
        return
    try:
        import pyrosetta

        pyrosetta.init(
            "-mute all -ignore_unrecognized_res true -detect_disulf false",
            silent=True,
        )
        _PYROSETTA_INITIALIZED = True
        logger.info("PyRosetta initialized successfully")
    except ImportError:
        raise PyRosettaError(
            "PyRosetta is not installed. Install with: "
            "pip install pyrosetta-installer && python -c "
            "'import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()'"
        )
    except Exception as e:
        raise PyRosettaError(f"Failed to initialize PyRosetta: {e}") from e


@dataclass
class PyRosettaConfig:
    """Configuration for PyRosetta operations."""

    score_function: str = "ref2015"
    relax_max_iter: int = 200
    relax_nstruct: int = 1
    design_repack_radius: float = 8.0


class PyRosettaRunner:
    """Wrapper for PyRosetta molecular modeling operations."""

    def __init__(self, config: PyRosettaConfig | None = None):
        self.config = config or PyRosettaConfig()

    async def score(
        self,
        pdb_path: str,
        score_fn: str | None = None,
    ) -> dict[str, Any]:
        """Score a PDB structure using a Rosetta energy function.

        Args:
            pdb_path: Path to input PDB file.
            score_fn: Score function name (default: ref2015).

        Returns:
            Dict with total_score, per_residue scores, and energy breakdown.
        """
        _ensure_init()
        import pyrosetta

        pdb_path = str(pdb_path)
        if not Path(pdb_path).exists():
            raise PyRosettaError(f"PDB file not found: {pdb_path}")

        sfxn_name = score_fn or self.config.score_function

        try:
            pose = pyrosetta.pose_from_pdb(pdb_path)
            sfxn = pyrosetta.create_score_function(sfxn_name)
            total_score = sfxn(pose)

            # Per-residue energies
            per_residue = []
            for i in range(1, pose.total_residue() + 1):
                res_energy = pose.energies().residue_total_energy(i)
                per_residue.append({
                    "residue": i,
                    "name": pose.residue(i).name3().strip(),
                    "chain": pose.pdb_info().chain(i),
                    "energy": round(float(res_energy), 3),
                })

            # Energy component breakdown
            energies = pose.energies()
            components = {}
            from pyrosetta.rosetta.core.scoring import ScoreType

            for st in sfxn.get_nonzero_weighted_scoretypes():
                name = ScoreType(st).name
                weight = sfxn.get_weight(st)
                value = energies.total_energies()[st]
                if abs(value) > 1e-6:
                    components[name] = {
                        "weighted": round(float(value * weight), 3),
                        "unweighted": round(float(value), 3),
                        "weight": round(float(weight), 3),
                    }

            return {
                "total_score": round(float(total_score), 3),
                "score_function": sfxn_name,
                "num_residues": pose.total_residue(),
                "per_residue": per_residue,
                "energy_components": components,
                "pdb_path": pdb_path,
            }
        except PyRosettaError:
            raise
        except Exception as e:
            raise PyRosettaError(f"Scoring failed: {e}") from e

    async def relax(
        self,
        pdb_path: str,
        nstruct: int | None = None,
        score_fn: str | None = None,
    ) -> dict[str, Any]:
        """Relax a PDB structure using Rosetta FastRelax.

        Args:
            pdb_path: Path to input PDB file.
            nstruct: Number of relaxation trajectories.
            score_fn: Score function name.

        Returns:
            Dict with relaxed_pdb_path, energy_before, energy_after, rmsd.
        """
        _ensure_init()
        import pyrosetta
        from pyrosetta.rosetta.protocols.relax import FastRelax

        pdb_path = str(pdb_path)
        if not Path(pdb_path).exists():
            raise PyRosettaError(f"PDB file not found: {pdb_path}")

        sfxn_name = score_fn or self.config.score_function
        n = nstruct or self.config.relax_nstruct

        try:
            pose = pyrosetta.pose_from_pdb(pdb_path)
            sfxn = pyrosetta.create_score_function(sfxn_name)

            energy_before = sfxn(pose)

            # Save initial coords for RMSD
            initial_pose = pose.clone()

            best_pose = None
            best_energy = float("inf")

            for _ in range(n):
                work_pose = initial_pose.clone()
                relax = FastRelax()
                relax.set_scorefxn(sfxn)
                relax.max_iter(self.config.relax_max_iter)
                relax.apply(work_pose)

                energy = sfxn(work_pose)
                if energy < best_energy:
                    best_energy = energy
                    best_pose = work_pose.clone()

            # Compute RMSD
            from pyrosetta.rosetta.core.scoring import CA_rmsd

            rmsd = CA_rmsd(initial_pose, best_pose)

            # Write output
            out_fd = tempfile.NamedTemporaryFile(
                suffix=".pdb", prefix="relaxed_", delete=False
            )
            out_path = out_fd.name
            out_fd.close()
            best_pose.dump_pdb(out_path)

            return {
                "relaxed_pdb_path": out_path,
                "energy_before": round(float(energy_before), 3),
                "energy_after": round(float(best_energy), 3),
                "energy_change": round(float(best_energy - energy_before), 3),
                "ca_rmsd": round(float(rmsd), 3),
                "nstruct": n,
                "score_function": sfxn_name,
            }
        except PyRosettaError:
            raise
        except Exception as e:
            raise PyRosettaError(f"Relaxation failed: {e}") from e

    async def interface_score(
        self,
        pdb_path: str,
        chains: str = "A_B",
        score_fn: str | None = None,
    ) -> dict[str, Any]:
        """Compute interface energy metrics for a protein complex.

        Uses Rosetta InterfaceAnalyzerMover to compute binding energy
        (dG_separated), buried surface area (dSASA), and interface contacts.

        Args:
            pdb_path: Path to complex PDB file.
            chains: Chain grouping, e.g. "A_B" or "AB_C".
            score_fn: Score function name.

        Returns:
            Dict with dG_separated, dSASA, interface hbonds, contacts, etc.
        """
        _ensure_init()
        import pyrosetta
        from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover

        pdb_path = str(pdb_path)
        if not Path(pdb_path).exists():
            raise PyRosettaError(f"PDB file not found: {pdb_path}")

        sfxn_name = score_fn or self.config.score_function

        try:
            pose = pyrosetta.pose_from_pdb(pdb_path)
            sfxn = pyrosetta.create_score_function(sfxn_name)
            sfxn(pose)

            iam = InterfaceAnalyzerMover(chains)
            iam.set_scorefunction(sfxn)
            iam.set_pack_separated(True)
            iam.set_pack_input(False)
            iam.apply(pose)

            return {
                "dG_separated": round(float(iam.get_separated_interface_energy()), 3),
                "dSASA": round(float(iam.get_interface_delta_sasa()), 1),
                "interface_hbonds": int(iam.get_interface_delta_hbond_unsat()),
                "packstat": round(float(iam.get_interface_packstat()), 3),
                "nres_interface": int(iam.get_num_interface_residues()),
                "chains": chains,
                "score_function": sfxn_name,
                "pdb_path": pdb_path,
            }
        except PyRosettaError:
            raise
        except Exception as e:
            raise PyRosettaError(f"Interface scoring failed: {e}") from e

    async def design(
        self,
        pdb_path: str,
        chains: str = "A_B",
        fixed_positions: list[int] | None = None,
        score_fn: str | None = None,
    ) -> dict[str, Any]:
        """Fixed-backbone sequence design using PackRotamers + MinMover.

        Chains: score → PackRotamers → MinMover → score.
        The composite ``rosetta_design`` tool wraps this.

        Args:
            pdb_path: Path to input PDB file.
            chains: Chain grouping for interface detection.
            fixed_positions: 1-indexed positions to keep fixed.
            score_fn: Score function name.

        Returns:
            Dict with designed sequence, scores, and designed PDB path.
        """
        _ensure_init()
        import pyrosetta
        from pyrosetta.rosetta.core.pack.task import TaskFactory
        from pyrosetta.rosetta.core.pack.task.operation import (
            InitializeFromCommandline,
            RestrictToRepacking,
            PreventRepacking,
        )
        from pyrosetta.rosetta.protocols.minimization_packing import (
            PackRotamersMover,
            MinMover,
        )

        pdb_path = str(pdb_path)
        if not Path(pdb_path).exists():
            raise PyRosettaError(f"PDB file not found: {pdb_path}")

        sfxn_name = score_fn or self.config.score_function
        fixed = set(fixed_positions or [])

        try:
            pose = pyrosetta.pose_from_pdb(pdb_path)
            sfxn = pyrosetta.create_score_function(sfxn_name)

            energy_before = sfxn(pose)
            seq_before = pose.sequence()

            # Set up task factory
            tf = TaskFactory()
            tf.push_back(InitializeFromCommandline())

            if fixed:
                # Prevent repacking at fixed positions
                prevent = PreventRepacking()
                for pos in fixed:
                    if 1 <= pos <= pose.total_residue():
                        prevent.include_residue(pos)
                tf.push_back(prevent)

            # Pack rotamers
            packer = PackRotamersMover(sfxn)
            packer.task_factory(tf)
            packer.apply(pose)

            # Minimize
            movemap = pyrosetta.MoveMap()
            movemap.set_bb(True)
            movemap.set_chi(True)

            min_mover = MinMover()
            min_mover.movemap(movemap)
            min_mover.score_function(sfxn)
            min_mover.min_type("lbfgs_armijo_nonmonotone")
            min_mover.tolerance(0.01)
            min_mover.apply(pose)

            energy_after = sfxn(pose)
            seq_after = pose.sequence()

            # Count mutations
            mutations = []
            for i, (a, b) in enumerate(zip(seq_before, seq_after)):
                if a != b:
                    mutations.append({
                        "position": i + 1,
                        "from": a,
                        "to": b,
                    })

            # Write output
            out_fd = tempfile.NamedTemporaryFile(
                suffix=".pdb", prefix="designed_", delete=False
            )
            out_path = out_fd.name
            out_fd.close()
            pose.dump_pdb(out_path)

            return {
                "designed_pdb_path": out_path,
                "sequence": seq_after,
                "sequence_before": seq_before,
                "energy_before": round(float(energy_before), 3),
                "energy_after": round(float(energy_after), 3),
                "energy_change": round(float(energy_after - energy_before), 3),
                "num_mutations": len(mutations),
                "mutations": mutations,
                "fixed_positions": sorted(fixed) if fixed else [],
                "score_function": sfxn_name,
            }
        except PyRosettaError:
            raise
        except Exception as e:
            raise PyRosettaError(f"Design failed: {e}") from e
