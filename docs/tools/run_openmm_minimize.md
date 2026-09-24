# run_openmm_minimize

**Category:** preparation  
**Engine:** `openmm_minimize`  
**Environment:** `md`  
**GPU required:** no

> This file is generated from `src/protein_design_mcp/manifests/run_openmm_minimize.yaml`. Edit the manifest, then run `python scripts/generate_tool_docs.py`.

## Summary

Relax a structure with OpenMM molecular mechanics, removing the clashes and strained geometry that generative models routinely produce. Run this before any physics-based scoring; scoring an unrelaxed model measures its clashes more than its interface. Returns the energy before and after, and writes the relaxed structure.

## What this is
Gradient-based energy minimisation under an Amber or CHARMM force field,
using OpenMM. Hydrogens are added before minimising.

## What it is for
Cleaning up a predicted or generated structure so that a physics-based score
means something. De novo designs and diffusion outputs frequently contain
atom clashes that dominate any energy term computed on them directly.

## When to use this instead of the alternatives
- This is preparation, not scoring. It tells you the structure's internal
  energy improved; it says nothing about whether two chains bind.
- For an interface score after relaxing, use `run_prodigy` for an absolute
  free energy or `run_ipsae` for predictor confidence.
- Minimisation moves atoms. If you need the original coordinates preserved
  exactly, score the input instead of the output.

## What you must supply
A PDB file. Multi-chain inputs are relaxed as one system.

## What you get back
`initial_potential_energy_kj_mol`, `final_potential_energy_kj_mol`,
`energy_change_kj_mol`, `iterations`, and under `outputs` the path to
`minimized_pdb`.

## Parameters

| Parameter | Type | Required | Default | Constraints | Description |
|---|---|---|---|---|---|
| `input_pdb` | string | yes | `—` | pattern: `\.pdb$` | Structure to minimise. WHERE THIS COMES FROM -- Any structure to relax -- your own, or one a folding or generation tool returned. |
| `max_iterations` | integer | no | `500` | minimum: `1`<br>maximum: `10000` | Most L-BFGS steps to take before stopping, whether or not the energy has converged. Higher costs proportionally more CPU and buys less the further it goes; 0 means run until convergence, which on a badly clashing structure can be much longer than you expect. A few hundred is enough to relieve the clashes a predicted structure carries. |
| `forcefield` | string | no | `amber14` | enum: `['amber14', 'charmm36']` | Force field to minimise under. |
