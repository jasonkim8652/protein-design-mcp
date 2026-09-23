"""Measure per-residue interface contacts and buried surface area (BSA)
between two (groups of) chains in an existing complex.

Runs inside the ``protein-design-mcp`` environment itself -- this is not an
external engine, it is this server's own already-tested geometry code
(``protein_design_mcp.utils.pdb.get_interface_residues`` for contacts,
``protein_design_mcp.utils.sasa.calculate_sasa`` for SASA) exposed as a
tool. No new science is implemented here: this script only wires validated
CLI arguments to those two functions and shapes the result.

Writes ``results.json`` into the working directory (an ``outputs:`` entry
in the manifest); the adapter reads it back rather than parsing stdout,
since the payload is nested (a per-residue table), not a handful of scalars.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from protein_design_mcp.exceptions import InvalidPDBError
from protein_design_mcp.utils.pdb import Structure, get_interface_residues, parse_pdb, write_pdb
from protein_design_mcp.utils.sasa import calculate_sasa


def _residue_key(chain_id: str, residue_number: int) -> str:
    """"{chain_id}{residue_number}" -- the exact key utils.sasa.calculate_sasa
    uses for SASAResult.per_residue, and (not a coincidence -- see the
    manifest doc) exactly the hotspot-tag format run_rfdiffusion_binder,
    run_genie3_binder, run_protpardelle and run_rfdiffusion3_binder expect.
    """
    return f"{chain_id}{residue_number}"


def _write_subset(structure: Structure, chain_ids: set[str], out_path: Path) -> None:
    subset = Structure(
        name=out_path.stem,
        chains=[c for c in structure.chains if c.chain_id in chain_ids],
    )
    write_pdb(subset, out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--complex-pdb", required=True)
    parser.add_argument("--target-chain", required=True)
    parser.add_argument("--binder-chains", required=True, help="JSON list of chain IDs")
    parser.add_argument("--contact-cutoff", type=float, required=True)
    args = parser.parse_args()

    binder_chains: list[str] = json.loads(args.binder_chains)

    structure = parse_pdb(args.complex_pdb)
    present = {c.chain_id for c in structure.chains}
    if args.target_chain not in present:
        raise InvalidPDBError(
            f"target_chain {args.target_chain!r} not found in {args.complex_pdb}; "
            f"chains present: {sorted(present)}"
        )
    missing_binders = sorted(set(binder_chains) - present)
    if missing_binders:
        raise InvalidPDBError(
            f"binder_chains {missing_binders} not found in {args.complex_pdb}; "
            f"chains present: {sorted(present)}"
        )

    # --- contacts: CA-CA distance cutoff, exactly get_interface_residues'
    # own (already tested) definition. One call per binder chain; a target
    # residue counts as a contact if it is within cutoff of ANY of them.
    contact_target_resnums: set[int] = set()
    partner_contacts: dict[str, list[str]] = {}
    for binder_chain in binder_chains:
        target_hits, partner_hits = get_interface_residues(
            args.complex_pdb, args.target_chain, binder_chain, args.contact_cutoff
        )
        contact_target_resnums.update(int(r) for r in target_hits)
        partner_contacts[binder_chain] = partner_hits

    # --- buried surface area: isolated vs bound-subset SASA, per residue.
    # "Bound" is deliberately the target+binder_chains SUBSET, not
    # necessarily the whole complex_pdb file, so a crystal's unrelated extra
    # chains don't dilute the specific pairwise interface being measured.
    workdir = Path.cwd()
    isolated_target_path = workdir / "_isolated_target.pdb"
    isolated_binder_path = workdir / "_isolated_binder.pdb"
    bound_path = workdir / "_bound_subset.pdb"

    _write_subset(structure, {args.target_chain}, isolated_target_path)
    _write_subset(structure, set(binder_chains), isolated_binder_path)
    _write_subset(structure, {args.target_chain, *binder_chains}, bound_path)

    isolated_target_sasa = calculate_sasa(str(isolated_target_path)).per_residue
    isolated_binder_sasa = calculate_sasa(str(isolated_binder_path)).per_residue
    bound_sasa = calculate_sasa(str(bound_path)).per_residue

    target_chain_obj = next(c for c in structure.chains if c.chain_id == args.target_chain)
    residues = []
    for residue in target_chain_obj.residues:
        key = _residue_key(args.target_chain, residue.residue_number)
        sasa_unbound = isolated_target_sasa.get(key, 0.0)
        sasa_bound = bound_sasa.get(key, 0.0)
        # Floating-point SASA noise between two independent Shrake-Rupley
        # runs can make bound - unbound very slightly positive; a truly
        # buried residue is never MORE exposed once bound, so clamp at 0
        # rather than report a physically meaningless negative BSA.
        buried = max(0.0, sasa_unbound - sasa_bound)
        residues.append(
            {
                "chain": args.target_chain,
                "residue_number": residue.residue_number,
                "residue_name": residue.residue_name,
                "is_contact": residue.residue_number in contact_target_resnums,
                "buried_sasa_a2": round(buried, 3),
                "sasa_unbound_a2": round(sasa_unbound, 3),
                "sasa_bound_a2": round(sasa_bound, 3),
                "hotspot_tag": key,
            }
        )

    hotspot_tags = [
        r["hotspot_tag"]
        for r in sorted(
            (r for r in residues if r["is_contact"]),
            key=lambda r: (-r["buried_sasa_a2"], r["residue_number"]),
        )
    ]

    partner_residues = []
    for binder_chain in binder_chains:
        binder_chain_obj = next(c for c in structure.chains if c.chain_id == binder_chain)
        contact_partner_resnums = {int(r) for r in partner_contacts[binder_chain]}
        for residue in binder_chain_obj.residues:
            key = _residue_key(binder_chain, residue.residue_number)
            sasa_unbound = isolated_binder_sasa.get(key, 0.0)
            sasa_bound = bound_sasa.get(key, 0.0)
            buried = max(0.0, sasa_unbound - sasa_bound)
            partner_residues.append(
                {
                    "chain": binder_chain,
                    "residue_number": residue.residue_number,
                    "residue_name": residue.residue_name,
                    "is_contact": residue.residue_number in contact_partner_resnums,
                    "buried_sasa_a2": round(buried, 3),
                    "sasa_unbound_a2": round(sasa_unbound, 3),
                    "sasa_bound_a2": round(sasa_bound, 3),
                    "hotspot_tag": key,
                }
            )

    result = {
        "target_chain": args.target_chain,
        "binder_chains": binder_chains,
        "contact_cutoff_angstrom": args.contact_cutoff,
        "residues": residues,
        "partner_residues": partner_residues,
        "hotspot_tags": hotspot_tags,
        "n_interface_residues": len(hotspot_tags),
        "total_buried_sasa_a2": round(
            sum(r["buried_sasa_a2"] for r in residues if r["is_contact"]), 3
        ),
    }
    (workdir / "results.json").write_text(json.dumps(result, indent=2))
    print(f"n_interface_residues: {result['n_interface_residues']}")


if __name__ == "__main__":
    main()
