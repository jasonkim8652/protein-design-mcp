# run_mpnn

**Category:** sequence_design  
**Engine:** `mpnn`  
**Environment:** `mpnn`  
**GPU required:** no

> This file is generated from `src/protein_design_mcp/manifests/run_mpnn.yaml`. Edit the manifest, then run `python scripts/generate_tool_docs.py`.

## Summary

Design amino acid sequences for a fixed protein backbone (ProteinMPNN and its variants). This is the step between generating a backbone and predicting what that sequence actually folds into. Select the variant with model_type: protein for the standard model, soluble to avoid designing buried hydrophobics, ligand when a small molecule, nucleotide or metal is present.

## What this is
Inverse folding: given backbone coordinates, predict amino acid sequences
likely to fold into them. One codebase serves three trained variants, chosen
with `model_type`.

## What it is for
Every generative backbone model produces coordinates without a sequence.
This turns them into something you can express, and is the most mature and
most validated step in the whole design pipeline.

## When to use this instead of the alternatives
- `model_type: protein` is the default and correct for a bare protein
  backbone.
- `model_type: soluble` is trained to avoid the exposed hydrophobic patches
  the standard model places when it has no membrane context. Prefer it for
  anything you intend to express in solution.
- `model_type: ligand` conditions on non-protein atoms. Use it whenever a
  small molecule, nucleotide or metal sits in the structure; the standard
  model ignores them and will design a sequence that clashes.
- After designing, fold the sequence back and check it matches the backbone
  you designed for. A sequence this tool likes is not automatically one that
  folds.

## What you must supply
A backbone PDB. Side chains are ignored.

## What you get back
`designs`, each with `id`, `sequence` and `overall_confidence`;
`num_designs`; and under `outputs` the path(s) to the FASTA file(s).

## Important caveat
The first record the engine emits is the INPUT sequence, not a design. It is
dropped here. Confidence is the model's own likelihood, not a prediction of
experimental success.

## Parameters

| Parameter | Type | Required | Default | Constraints | Description |
|---|---|---|---|---|---|
| `chains_to_design` | string | no | `None` | pattern: `^[A-Za-z0-9]( [A-Za-z0-9])*$` | Which chain(s) to design, space-separated (e.g. "B", or "A B"). LEAVING THIS UNSET DESIGNS EVERY CHAIN IN THE FILE, including a target you meant to keep -- a 504-residue target plus an 80-residue binder came back as one 585-residue sequence that is neither. Every binder generator here returns a two-chain complex, so for binder design you almost always want to name the design chain. Which chain that is differs per generator: run_rfdiffusion3_binder and run_genie3_binder put it in chain A, run_rfdiffusion2, run_protpardelle and run_proteina_complexa_generate in chain B. Read the generator's own output description rather than assuming. |
| `fixed_residues` | string | no | `None` | — | Individual residues to hold fixed, space-separated `{chain}{number}` tags (e.g. "A12 A13 B2"). Finer-grained than chains_to_design and combinable with it: use this to keep an interface motif while redesigning the rest of the same chain. |
| `redesigned_residues` | string | no | `None` | — | The inverse of fixed_residues -- design ONLY these residues and hold everything else fixed, same `{chain}{number}` tag format. Use whichever of the two is shorter to write; naming both is accepted by the engine but makes the intent hard to read. |
| `backbone_pdb` | string | yes | `—` | pattern: `\.(pdb\|cif)$` | Backbone structure to design a sequence for. |
| `model_type` | string | no | `protein` | enum: `['protein', 'soluble', 'ligand']` | Which trained variant to use. |
| `num_sequences` | integer | no | `8` | minimum: `1`<br>maximum: `128` | How many sequences to sample. |
| `sampling_temp` | number | no | `0.1` | minimum: `0.0001`<br>maximum: `1.0` | Sampling temperature. Lower is more conservative. |
| `seed` | integer | no | `37` | minimum: `0` | Random seed. Fixed by default so runs are reproducible. |
