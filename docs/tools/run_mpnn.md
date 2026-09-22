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
| `backbone_pdb` | string | yes | `—` | pattern: `\.(pdb\|cif)$` | Backbone structure to design a sequence for. |
| `model_type` | string | no | `protein` | enum: `['protein', 'soluble', 'ligand']` | Which trained variant to use. |
| `num_sequences` | integer | no | `8` | minimum: `1`<br>maximum: `128` | How many sequences to sample. |
| `sampling_temp` | number | no | `0.1` | minimum: `0.0001`<br>maximum: `1.0` | Sampling temperature. Lower is more conservative. |
| `seed` | integer | no | `37` | minimum: `0` | Random seed. Fixed by default so runs are reproducible. |
