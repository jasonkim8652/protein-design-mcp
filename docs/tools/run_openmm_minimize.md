# run_openmm_minimize

**Category:** preparation  
**Engine:** `openmm_minimize`  
**Environment:** `md`  
**GPU required:** no

> This file is generated from `src/protein_design_mcp/manifests/run_openmm_minimize.yaml`. Edit the manifest, then run `python scripts/generate_tool_docs.py`.

## Summary

Relax a structure with OpenMM molecular mechanics, removing the clashes and strained geometry in predicted structures. Potential energy is sensitive to clashes and geometry. Returns the energy before and after, and writes the relaxed structure.

## What this is
Gradient-based energy minimisation under an Amber or CHARMM force field,
using OpenMM. Missing terminal heavy atoms (such as OXT, which AF2
predictions omit) are added with PDBFixer before hydrogens are added.
Missing internal residues or side chains are not reconstructed. Hydrogens
use OpenMM's simplified geometry preparation on explicit Reference with a
recorded random seed and residue variants. Protocol
`soft-repulsion-flexible-hbonds-v1` applies 200 steps with flexible bonds,
bounded quadratic overlap repulsion and heavy-atom positional restraints,
then 200 steps under the unrestrained flexible physical force field, then
the requested final minimization with HBonds constraints. Temporary forces
are absent from the reported initial and final physical energies.
CUDA double precision is explicit by default; unavailable CUDA fails without
falling back. CPU and Reference must be requested explicitly.
The system is minimized in vacuum: no explicit water or implicit-solvent
model is added. Loading water parameter definitions does not add solvent.

## What it is for
Cleaning up a predicted or generated structure so that a physics-based score
means something. De novo designs and diffusion outputs frequently contain
atom clashes that dominate any energy term computed on them directly.

## When to use this instead of the alternatives
- This is preparation, not scoring. It tells you the structure's internal
  energy improved; it says nothing about whether two chains bind.
- `run_prodigy` estimates binding affinity from structural contacts;
  `run_ipsae` reports predictor confidence using the prediction and its PAE.
- Minimisation moves atoms. If you need the original coordinates preserved
  exactly, score the input instead of the output.

## What you must supply
A PDB or CIF/mmCIF file, optionally gzip-compressed. Multi-chain inputs
are relaxed as one system. CIF inputs are read directly without a
coordinate conversion; the minimized output is a PDB file. Every canonical
amino-acid residue in every chain must contain its complete nonterminal
heavy-atom set (backbone and side chains, including partial side chains).
Hydrogens and terminal OXT may be absent. Incomplete canonical residues
are rejected before engine dispatch with chain, residue and missing-atom
diagnostics. Unknown or modified residues remain subject to the engine's
template and force-field checks. The caller must ensure the structure's
sequence matches the exact intended design and target sequences; this tool
does not verify sequence identity or reconstruct missing internal atoms.

## What you get back
`initial_potential_energy_kj_mol`, `final_potential_energy_kj_mol`,
`energy_change_kj_mol`, `iterations`, `added_terminal_atoms`, `force_field`,
`solvent_model` (`none`), `energy_units` (`kJ/mol`), and under
`outputs` the path to
`minimized_pdb`. Additional fields include `minimization_protocol`, actual
`platform`, `precision`, `geometry_passed`, and `minimization_diagnostics`.
`iterations` counts observed reporter callbacks across stages and constraint
passes, not the requested cap. Full precision states and Systems are saved
with SHA256 identities. Geometry, force residuals, constraint errors and
heavy-atom displacement are recorded. A failed geometry check must not be
treated as a usable energy measurement. Passing the gross checks establishes
neither convergence nor a correct binding pose or affinity.

## Parameters

| Parameter | Type | Required | Default | Constraints | Description |
|---|---|---|---|---|---|
| `input_pdb` | string | yes | `—` | pattern: `\.(pdb\|cif\|mmcif)(\.gz)?$` | Structure to minimise (PDB or CIF/mmCIF, optionally .gz). Every canonical residue in every chain requires all nonterminal backbone and side-chain heavy atoms; a partial side chain is incomplete. Hydrogens and terminal OXT may be absent. Missing internal atoms are not reconstructed. The caller must ensure the structure contains the exact intended design and target sequences. Any structure source is acceptable if it meets these requirements; no particular upstream tool is required. |
| `max_iterations` | integer | no | `500` | minimum: `1`<br>maximum: `10000` | Iteration cap per OpenMM constraint pass in the final physical stage. Two fixed 200-step preparation stages precede it. Constraint restarts may exceed this cap in total. A cap or low energy does not prove convergence. |
| `forcefield` | string | no | `amber14` | enum: `['amber14', 'charmm36']` | Force field to minimise under. |
| `platform` | string | no | `CUDA` | enum: `['CUDA', 'Reference', 'CPU']` | Explicit computation platform; failure to initialize is an error, with no fallback. Reference supports CPU-only validation. |
| `precision` | string | no | `double` | enum: `['double', 'mixed']` | CUDA precision; other explicitly selected platforms use their native precision, reported in diagnostics. |
