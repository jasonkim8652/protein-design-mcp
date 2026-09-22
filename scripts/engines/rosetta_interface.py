"""Wrapper for PyRosetta's InterfaceAnalyzerMover (``run_rosetta_interface``).
Runs inside the ``pyrosetta`` environment.

See ``adapters/rosetta_interface.py`` and the manifest's "Verification
status" section for why `import pyrosetta` currently fails deterministically
in THIS environment (the shipped wheel is missing its compiled `.so`
extension -- not something this script can work around). Everything below
is built against the InterfaceAnalyzerMover API this wave verified live, end
to end, in a separate, genuinely working PyRosetta install on this same host
(see the wave report) -- ``DockingPartners.docking_partners_from_string``
for the "A_B" interface notation, the 6-positional-argument
InterfaceAnalyzerMover constructor, ``set_compute_interface_sc``, and the
``InterfaceData`` struct's field names (``dG``, ``dSASA``, ``sc_value``,
``interface_hbonds``, ``delta_unsat_hbonds``, ``packstat``).

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
