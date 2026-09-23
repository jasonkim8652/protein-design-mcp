"""Wrapper for PyRosetta's InterfaceAnalyzerMover (``run_rosetta_interface``).
Runs inside the manifest's ``engine.prefix`` env, ``~/.conda/envs/BindCraft``
-- reused strictly READ-ONLY for its working ``pyrosetta`` install, never
modified (see the manifest's "Verification status" section for why: the
dedicated ``pyrosetta`` env built from
``/opt/pyrosetta_wheels/pyrosetta-2017-cp312-cp312-linux_x86_64.whl`` is
broken -- that wheel ships zero compiled ``.so`` extensions, so
``import pyrosetta`` can never succeed from it).

CONFIRMED LIVE end to end, 2026-09-22, running this exact script through
``~/.conda/envs/BindCraft``'s interpreter over
``tests/fixtures/test_pdbs/1BRS.pdb`` (barnase-barstar, chains ``A_D``) --
see the manifest's "Verification status" section for the returned values.
``DockingPartners.docking_partners_from_string`` for the "A_B" interface
notation, the 6-positional-argument InterfaceAnalyzerMover constructor,
``set_compute_interface_sc``, and the ``InterfaceData`` struct's field names
(``dG``, ``dSASA``, ``sc_value``, ``interface_hbonds``, ``delta_unsat_hbonds``,
``packstat``) are all exercised for real by this run, not merely read from
source.

Reads one argv: a JSON object (see ``adapters/rosetta_interface.py`` for its
exact shape). Prints one JSON object as the LAST line of stdout; nothing is
written to disk (this tool declares no ``outputs:`` -- there is nothing to
collect).
"""

from __future__ import annotations

import json
import sys


def main() -> None:
    job = json.loads(sys.argv[1])

    import pyrosetta
    from pyrosetta.rosetta.core.pose import DockingPartners
    from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover

    pyrosetta.init(
        "-mute all -ignore_unrecognized_res true -detect_disulf false",
        silent=True,
    )

    pose = pyrosetta.pose_from_pdb(job["complex_pdb"])

    partners = DockingPartners.docking_partners_from_string(job["interface"])
    scorefxn = pyrosetta.create_score_function(job["score_function"])

    iam = InterfaceAnalyzerMover(
        partners,
        False,  # tracer -- never write PyMOL/tracer output to stdout ourselves
        scorefxn,
        job["compute_packstat"],
        job["pack_input"],
        job["pack_separated"],
    )
    iam.set_compute_interface_sc(job["compute_interface_sc"])
    iam.set_pack_rounds(job["pack_rounds"])
    iam.apply(pose)

    data = iam.get_all_data()
    # dG/dSASA are Rosetta vector1<double> objects: [total, side1, side2].
    # Index 1 (Rosetta's own 1-based indexing, exposed as-is through the
    # pybind11 binding) is the total across the whole interface -- the same
    # value get_interface_dG()/get_interface_delta_sasa() would report.
    result = {
        "dG": float(data.dG[1]),
        "dSASA": float(data.dSASA[1]),
        "shape_complementarity": float(data.sc_value) if job["compute_interface_sc"] else None,
        "interface_hbonds": int(data.interface_hbonds),
        "delta_unsat_hbonds": int(data.delta_unsat_hbonds),
        # data.interface_nres is a per-side vector1, not a scalar -- the
        # mover's own get_num_interface_residues() getter is the total
        # count across both sides (verified live, 2026-09-22).
        "num_interface_residues": int(iam.get_num_interface_residues()),
        "packstat": float(data.packstat) if job["compute_packstat"] else None,
        "interface": job["interface"],
        "score_function": job["score_function"],
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
