# run_rebuild_backbone

**Category:** preparation  
**Engine:** `pulchra`  
**Environment:** `mpnn`  
**GPU required:** no

> This file is generated from `src/protein_design_mcp/manifests/run_rebuild_backbone.yaml`. Edit the manifest, then run `python scripts/generate_tool_docs.py`.

## Summary

Reconstruct a full protein backbone (N, CA, C, O) from a CA-only trace with PULCHRA, and rename placeholder residues so a sequence designer can read the result. Run this between a CA-trace generator and run_mpnn: run_genie3_binder and run_genie3_scaffold emit CA-only backbones that run_mpnn cannot consume at all. Every other generator here already emits a full backbone and does not need this step.

## What this is
PULCHRA (the published CA-trace reconstruction method) places the missing
backbone atoms implied by a chain of CA positions, then this tool renames
placeholder residues (`UNK`) to `GLY`.

## What it is for
Exactly one handoff: making a CA-trace generator's output designable.

`run_genie3_binder` and `run_genie3_scaffold` represent every residue they
GENERATE as a single CA atom -- Genie 3's "all-atom" refers to what it
conditions on, not what it produces, and no parameter changes this. Handed
such a file, `run_mpnn` fails with
`AttributeError: 'NoneType' object has no attribute 'select'`, because two
separate things are wrong with it:

- ProteinMPNN needs N, CA, C and O to build each residue's frame; and
- LigandMPNN parses with ProDy, whose `protein` selection is matched by
  residue NAME, and `UNK` is not in that set.

This tool fixes both. Verified end to end on a real Genie 3 binder output:
before, run_mpnn died; after, it designed the 80-residue chain and held the
target chain fixed.

## When to use this instead of the alternatives
`run_openmm_minimize` is the other preparation tool, and the two are not
interchangeable — they fix different defects and run at different points.

- Use **run_rebuild_backbone** when atoms are MISSING: a CA-only trace has
  no N, C or O to relax, and `run_openmm_minimize` cannot invent them. This
  runs first, on a generator's raw output, and its consumer is `run_mpnn`.
- Use **run_openmm_minimize** when atoms are PRESENT but strained: clashes
  and distorted geometry in an otherwise complete structure, before
  physics-based scoring.
- Needing both is ordinary: rebuild the backbone, design a sequence onto
  it, fold the design, then minimise the folded complex before scoring it.
  Minimising a reconstructed backbone before a sequence exists relaxes
  approximated coordinates that `run_mpnn` is about to reinterpret anyway.

## When you do NOT need this
Every other backbone generator here -- run_rfdiffusion3_binder,
run_rfdiffusion3_scaffold, run_rfdiffusion2, run_rfdiffusion_binder,
run_protpardelle, run_proteina_complexa_generate, run_boltzgen_design --
already writes a complete backbone. Passing one of those through this tool
is refused: reconstructing atoms that were measured replaces them with
approximations for no gain.

## What you must supply
A structure with at least one chain that has CA atoms but no complete
backbone. A file in which EVERY chain is already complete is refused.

A generator's own output usually holds the design AND the target in one
file; pass the whole file. Chains that already have a full backbone are
left as they are, so the target is not disturbed.

## What you get back
`structure_pdb`, the rebuilt structure, plus `chains`, `residues_rebuilt`
and `backbone_complete`. Check `backbone_complete`: if it is false the next
tool will fail on the same file, and the cause is still visible here.

## What it does not do
Side chains are not rebuilt, so the result is NOT a molecular-mechanics
input: `run_openmm_minimize` needs every residue's complete heavy-atom set
and refuses a backbone (a live round reached OpenMM's own
`ValueError: HIS residue (118) has the wrong set of atoms` this way).
Minimise the FOLDED complex a co-folding tool produces downstream, not this.

The chain is about to be redesigned, so
invented side chains would be discarded by the next tool, and ProteinMPNN
derives a virtual CB from N/CA/C regardless. The reconstructed backbone is
an approximation of an unobserved geometry, not a measurement -- score the
design that comes out of it, never this structure itself.

## Parameters

| Parameter | Type | Required | Default | Constraints | Description |
|---|---|---|---|---|---|
| `structure` | string | yes | `—` | pattern: `\.(pdb\|cif)(\.gz)?$` | The CA-trace structure to rebuild, as a generator wrote it -- typically run_genie3_binder's `output/target/pdbs/*.pdb` or run_genie3_scaffold's `output/pdbs/*.pdb`. Pass the file whole: it normally carries the target alongside the design, and chains that already have a backbone are left untouched. |
