# run_epitope_scan

**Category:** target_analysis  
**Engine:** `protein_design_mcp`  
**Environment:** `server`  
**GPU required:** no

> This file is generated from `src/protein_design_mcp/manifests/run_epitope_scan.yaml`. Edit the manifest, then run `python scripts/generate_tool_docs.py`.

## Summary

Find candidate epitope residues on a target that has NO known binder yet, ranked by solvent exposure, with conservation exposed as separate per-residue evidence rather than folded into one score. Reads no network resource: the structure and (optionally) the alignment are both caller-supplied, exactly like every other tool here that could otherwise leak a novel sequence to a remote service. Feeds directly into the four tools that require hotspots and cannot produce them themselves: run_rfdiffusion_binder, run_genie3_binder, run_protpardelle, run_rfdiffusion3_binder.

## What this is
This server's own already-tested analysis (`utils.sasa.calculate_sasa` for
exposure, and `utils.conservation`'s per-position scoring primitive for
conservation) applied to an unbound target, so a caller with no existing
complex still has a path to hotspots. No structure or sequence data is
fetched by this tool -- see "No network access" below.

For every residue of `chain`, this tool reports `sasa_a2` (raw SASA),
`is_exposed` (whether it clears `exposure_threshold_a2`),
`conservation_score` (0-1, or `null` when no alignment was supplied or the
alignment carries no homologs), and `is_conserved` (whether it clears
`conserved_threshold`). Both flags are convenience thresholds on top of
the raw numbers, which are always returned too.

## Why this replaces `suggest_hotspots`
The tool this replaces combined several signals with hardcoded weights and
returned one ranked answer, with no way for a caller to see why a residue
ranked where it did or to disagree with the weighting. This tool never
combines exposure and conservation into one score. The convenience
`hotspot_tags` list is ordered by EXPOSURE ALONE: burial is a hard
structural precondition for being bindable at all (a buried residue
literally cannot be a binder hotspot), so filtering/ordering on it is not
a value judgment. Conservation is NOT part of that order, because whether
a CONSERVED or a VARIABLE epitope is what you want depends on your own
goal -- a conserved epitope for cross-reactivity, a variable one for
specificity -- and baking either preference into a default ranking would
be exactly the mistake `suggest_hotspots` made. Re-sort `residues`
yourself by `conservation_score` (ascending or descending, your choice)
once you have decided which you want.

## No network access
`target_pdb` is a structure you already have; this tool never fetches one
from RCSB or AlphaFold DB (unlike `utils.fetch_structure`, which exists in
this repo but is deliberately not wired to this tool). `msa` is a caller-
supplied alignment file; this tool never queries UniProt or builds an
alignment itself (unlike `utils.uniprot`, also deliberately not wired
here). A target sequence handed to this tool never leaves the machine --
the same reasoning that removed `"auto"` MSA generation from every
structure-prediction tool in this server: several of those engines default
to a REMOTE MSA server, and the sequences here are usually novel designs.

## Where the alignment comes from
`msa` follows the exact same contract as every other tool's `msa`
parameter in this server: `null` scores exposure alone (deliberately, no
conservation), a path uses that alignment, and the parameter is required
so the choice is always stated -- there is no `"auto"`. A path must be a
plain a3m WHOSE QUERY ROW IS EXACTLY `chain`'s own sequence --
`run_mmseqs_search`'s `unpaired_a3m` for that exact sequence is valid
here; `run_colabfold_search`'s a3m searches a different sequence universe
and is also valid, since both share the plain-a3m shape this tool reads
(see `run_mmseqs_search`'s own doc for why the two are not
interchangeable with EACH OTHER, which does not apply here -- this tool
only reads per-column identity, not which database a hit came from).

## Getting from this output to each of the four hotspot-consuming tools
`hotspot_tags` is a list of `"{chain_id}{residue_number}"` strings (e.g.
`["A41", "A52", "A58"]`), the exposed residues among `chain`, ordered by
`sasa_a2` descending and capped at `top_n_hotspots`. This is the exact
same token shape `run_interface_residues` returns, so it plugs into the
same four consumers the same way:

```
scan = call_tool("run_epitope_scan", {...})
tags = scan["hotspot_tags"]                # e.g. ["A41", "A52", "A58"]

call_tool("run_rfdiffusion_binder", {"hotspot_res": tags, ...})
call_tool("run_genie3_binder",      {"hotspot_residues": tags, ...})
call_tool("run_protpardelle",       {"hotspots": tags, ...})
call_tool("run_rfdiffusion3_binder", {"select_hotspots": ",".join(tags), ...})
# -> "A41,A52,A58"
```

If you want a conservation-informed shortlist instead of the default
exposure-only one, build your own list from `residues` (e.g. `[r["hotspot_tag"]
for r in scan["residues"] if r["is_exposed"] and r["is_conserved"]]`) and
pass THAT instead of `hotspot_tags` -- the per-residue evidence is there
precisely so you are not limited to this tool's own default ordering.

## Accuracy caveats
- `sasa_a2` shares `run_interface_residues`' documented caveat: this
  server's SASA implementation resolves every atom's van der Waals radius
  as carbon's, regardless of true element -- a systematic approximation,
  not fixed by this tool. Treat `sasa_a2` as indicative, not calibrated.
- `conservation_score` is a simple per-column most-common-residue
  frequency (Shannon-style, not a substitution-matrix-aware score), and is
  only as good as the alignment you supply. Check `num_aligned_sequences`
  before trusting a score computed from very few homologs -- a
  conservation_score from 2-3 sequences is close to meaningless. When
  `msa` is supplied but carries zero homolog rows (e.g. a genuine
  zero-hit `run_mmseqs_search` result, which still writes a query-only
  a3m -- see that tool's own doc), `conservation_score` is `null` for
  every residue rather than a fabricated number, and
  `num_aligned_sequences` reports 0 so this is never silently confused
  with `msa: null`.

## When to use this instead of the alternatives
- `run_interface_residues` is the other `target_analysis` tool, and
  answers a different question: it MEASURES an interface you already have
  (a complex), not a prediction on an unbound target. Use THIS tool
  (`run_epitope_scan`) only when no complex/known binder exists yet; once
  you have generated and folded a candidate, `run_interface_residues` on
  that result is the measured ground truth, not a re-run of this tool.
- Neither tool combines its evidence into one opaque score (the old
  `suggest_hotspots` composite did, and was removed for exactly that
  reason) -- both return the underlying numbers per residue so you can
  re-rank on your own criteria.

## What you must supply
`target_pdb`, `chain`, and `msa` (a path, or explicitly `null`).

## What you get back
`residues` (full per-residue table), `hotspot_tags` (the convenience list
described above), `n_exposed_residues`, `n_conserved_residues`,
`msa_provided`, `num_aligned_sequences`, and the two thresholds echoed
back (`exposure_threshold_a2`, `conserved_threshold`).

## Parameters

| Parameter | Type | Required | Default | Constraints | Description |
|---|---|---|---|---|---|
| `target_pdb` | string | yes | `—` | pattern: `\.(pdb\|ent)$` | Structure file for the unbound target. Only plain PDB (or legacy .ent) is read -- this tool's parser (Biopython's PDBParser, via this server's own utils.pdb) does not read mmCIF. WHERE THIS COMES FROM -- The unbound target -- your own structure file. Use `run_interface_residues` instead when you already have a complex, since that measures an interface rather than predicting where one could be. |
| `chain` | string | yes | `—` | pattern: `^[A-Za-z0-9]$` | Single-character chain ID to scan for candidate epitope residues. SASA is computed on the WHOLE structure in target_pdb exactly as given (so an oligomeric target's own subunit contacts are respected, not silently isolated) -- only the reported per-residue table is filtered to this chain. |
| `msa` | — | yes | `—` | pattern: `\.a3m$` | null scores exposure alone, deliberately, with no conservation term (every residue's conservation_score is null). A path to a plain a3m alignment whose QUERY ROW (first record) is EXACTLY chain's own sequence -- run_mmseqs_search's unpaired_a3m for that exact sequence is valid here. There is no "auto": this tool never builds or fetches an alignment itself. |
| `exposure_threshold_a2` | number | no | `30.0` | minimum: `0.0`<br>maximum: `300.0` | Per-residue SASA in Angstroms^2 above which is_exposed is set. 30.0 matches this server's own utils.sasa.EXPOSED_THRESHOLD convention (read from source), reused here as the default. Only the boolean flag and the hotspot_tags shortlist are affected -- sasa_a2 is always returned unfiltered, so you can re-threshold yourself from `residues`. |
| `conserved_threshold` | number | no | `0.8` | minimum: `0.0`<br>maximum: `1.0` | Per-position conservation score above which is_conserved is set. 0.8 matches the threshold utils.conservation's own (unexposed, hardcoded) "highly_conserved" cutoff uses (read from source), reused here as an adjustable default. Only meaningful when msa is supplied; with msa: null every conservation_score is null and is_conserved is always false. |
| `top_n_hotspots` | integer | no | `10` | minimum: `1`<br>maximum: `50` | How many of the highest-exposure residues to include in the convenience hotspot_tags shortlist. This never limits `residues`, which always reports EVERY residue of `chain` -- only the shortlist is capped. 10 is a middle ground: run_rfdiffusion_binder's own doc recommends 3-6 hotspots, run_genie3_binder allows up to 50. Build your own list from `residues` if 10 does not fit the tool you are feeding. |
