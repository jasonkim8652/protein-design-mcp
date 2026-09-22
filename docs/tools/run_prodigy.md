# run_prodigy

**Category:** scoring  
**Engine:** `prodigy`  
**Environment:** `scoring`  
**GPU required:** no

> This file is generated from `src/protein_design_mcp/manifests/run_prodigy.yaml`. Edit the manifest, then run `python scripts/generate_tool_docs.py`.

## Summary

Estimate the binding free energy of an existing protein-protein complex from its interfacial contacts (PRODIGY). Runs on CPU in milliseconds and needs no model weights, so it is the cheapest first look at an interface. It scores a complex you already have; it does not predict structure and does not design. Fold a candidate with run_chai1 or run_esmfold2 first.

## What this is
PRODIGY predicts the binding affinity of a protein-protein complex with a
linear regression over the number and type of interfacial residue contacts.
It returns a binding free energy in kcal/mol and a dissociation constant.

## What it is for
A fast, interpretable, absolute-scale sanity floor on an interface. Because
it is CPU-only and takes milliseconds, it is cheap enough to run on every
candidate before spending GPU time on anything else.

## When to use this instead of the alternatives
- `run_rosetta_interface` (not yet implemented) gives a physics-based
  decomposition (dG_separated, buried surface area, shape complementarity,
  hydrogen bond counts) and will be the better choice when you need to know
  *why* an interface scores as it does. It is slower and depends on a
  license-gated PyRosetta install. Until then, use PRODIGY.
- `run_ipsae` (not yet implemented) and ipTM fields from co-folding tools
  measure model *confidence* in the interface, not its energy. Those
  discriminate binders from non-binders better than PRODIGY does and will
  be preferred for ranking designs once available. Until then, use PRODIGY.
- Use PRODIGY when you want an absolute number on a physical scale rather
  than a model-internal confidence score.

## Important caveat
PRODIGY is calibrated on natural protein complexes from the affinity
benchmark. It systematically mis-ranks de novo designed binders. Treat its
output as a sanity floor, never as the ranking criterion for a design
campaign.

## What you must supply
A PDB or mmCIF file containing both partners, and the chain identifier of
each partner.

## What you get back
`binding_affinity_kcal_per_mol`, `dissociation_constant_M`,
`intermolecular_contacts`, and a `caveat` string restating the calibration
limitation.

## Parameters

| Parameter | Type | Required | Default | Constraints | Description |
|---|---|---|---|---|---|
| `complex_pdb` | string | yes | `—` | pattern: `\.(pdb\|cif\|ent)$` | Path to a structure file containing both partners. |
| `chain_a` | string | yes | `—` | pattern: `^[A-Za-z0-9]$` | Chain identifier of the first partner: a single character only. PRODIGY requires single-character chain IDs; multi-character mmCIF chain identifiers are not supported. |
| `chain_b` | string | yes | `—` | pattern: `^[A-Za-z0-9]$` | Chain identifier of the second partner: a single character only. PRODIGY requires single-character chain IDs; multi-character mmCIF chain identifiers are not supported. |
| `temperature` | number | no | `25.0` | minimum: `0.0`<br>maximum: `100.0` | Temperature in Celsius used for the Kd conversion. |
