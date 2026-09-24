# run_interface_residues

**Category:** target_analysis  
**Engine:** `protein_design_mcp`  
**Environment:** `server`  
**GPU required:** no

> This file is generated from `src/protein_design_mcp/manifests/run_interface_residues.yaml`. Edit the manifest, then run `python scripts/generate_tool_docs.py`.

## Summary

Measure, from a structure you already have, which target residues actually contact an existing binder/partner chain and how much surface each buries on binding. This is the MEASURED counterpart to run_epitope_scan's prediction: you already have the complex, so there is no ranking or evidence weighting to argue about -- only geometry. Feeds directly into the four tools that require hotspots and cannot produce them themselves: run_rfdiffusion_binder, run_genie3_binder, run_protpardelle, run_rfdiffusion3_binder.

## What this is
Two already-tested pieces of this server's own analysis code
(`protein_design_mcp.utils.pdb.get_interface_residues` for contacts,
`protein_design_mcp.utils.sasa.calculate_sasa` for solvent-accessible
surface area), wired to CLI arguments and run for you. No new geometry is
implemented by this tool -- it exposes analysis that already existed and
was already tested in this repository.

For each residue of `target_chain`, this tool reports:
- `is_contact`: whether it lies within `contact_cutoff` (CA-CA distance,
  see below) of ANY of `binder_chains`.
- `buried_sasa_a2`: solvent-accessible surface area lost between the
  isolated `target_chain` and the `target_chain` + `binder_chains`
  subset -- the standard measurable definition of "how much of this
  residue is at the interface."
- `sasa_unbound_a2` / `sasa_bound_a2`: the two raw numbers `buried_sasa_a2`
  is the difference of, so you can see the underlying evidence rather than
  just the derived one.

The same table is also computed for `binder_chains` (returned under
`partner_residues`), since the identical geometry is available for free
once the bound/unbound SASA runs have been done -- useful if you want the
partner's own paratope/epitope, not just the target's.

## Getting from this output to each of the four hotspot-consuming tools
`hotspot_tags` is a list of `"{chain_id}{residue_number}"` strings (e.g.
`["A30", "A33", "A34"]`) for every contact residue of `target_chain`,
ordered by `buried_sasa_a2` descending (most-buried first) then by
residue number. **This single shape is valid, unmodified, as three of the
four consumers' parameter directly**:

```
interface = call_tool("run_interface_residues", {...})
tags = interface["hotspot_tags"]          # e.g. ["A30", "A33", "A34"]

call_tool("run_rfdiffusion_binder", {"hotspot_res": tags, ...})
call_tool("run_genie3_binder",      {"hotspot_residues": tags, ...})
call_tool("run_protpardelle",       {"hotspots": tags, ...})
```

The fourth, `run_rfdiffusion3_binder`, takes the SAME tokens but as one
comma-joined string instead of a JSON array:

```
call_tool("run_rfdiffusion3_binder", {"select_hotspots": ",".join(tags), ...})
# -> "A30,A33,A34"
```

No reformatting, residue-number offsetting, or chain-relabeling is ever
needed between this tool and any of the four -- they were all read from
source specifically to confirm they use this exact
`"<ChainID><ResidueNumber>"` token grammar (see each tool's own manifest
for the citation). `target_chain` here should be the SAME chain ID your
`target_pdb` for the consuming tool uses, since the tags are only
meaningful against that numbering.

## Why `contact_cutoff` is CA-CA, not heavy-atom
`get_interface_residues` (the function this tool calls) measures CA-CA
distance, a coarse, established proxy for contact -- not a heavy-atom or
side-chain-aware definition. A residue whose side chain reaches into the
interface while its CA stays farther back can be missed at a tight
cutoff. 8.0 A (the function's own default) is a reasonable middle ground;
widen it if you suspect this is under-calling contacts for a target with
long side chains at the interface (e.g. Arg, Lys, Trp).

## Accuracy caveat on `buried_sasa_a2` and its own components
This server's SASA implementation (`utils.sasa.calculate_sasa`) resolves
every atom's van der Waals radius as carbon's (1.70 A), regardless of the
atom's real element -- read from source, not modified here (editing a
shared utility is out of this tool's own scope; see this wave's report).
This is a SYSTEMATIC approximation applied identically to the bound and
unbound calculations, so the two numbers stay on a comparable scale and
`is_contact` (from the separate, unaffected CA-distance calculation) is
unaffected -- but the absolute `buried_sasa_a2` magnitude should be read
as indicative, not a calibrated physical value. Treat it the way
`run_prodigy` asks you to treat its binding-affinity number: a sanity
signal and a within-call ranking tool, not an absolute ground truth.

## When to use this instead of the alternatives
- `run_epitope_scan` is the other `target_analysis` tool, and answers a
  different question: it PREDICTS candidate epitope residues on an
  UNBOUND target (no known binder yet), ranked by exposure with
  conservation as separate evidence. Use THIS tool (`run_interface_residues`)
  whenever you already have a complex -- a crystal structure, a
  co-folded prediction, a previous design round's output -- and want the
  MEASURED interface rather than a prediction. Use `run_epitope_scan` only
  when no complex exists yet.
- Neither tool combines its evidence into one opaque score (the old
  `suggest_hotspots` composite did, and was removed for exactly that
  reason) -- both return the underlying numbers per residue so you can
  re-rank on your own criteria.

## What you must supply
`complex_pdb` (target + binder already in one file), `target_chain`, and
`binder_chains`.

## What you get back
`residues` (full per-residue table for `target_chain`), `partner_residues`
(the same, for `binder_chains`), `hotspot_tags` (the convenience list
described above), `n_interface_residues`, `total_buried_sasa_a2`, and
`contact_cutoff_angstrom` (echoing what was used).

## Parameters

| Parameter | Type | Required | Default | Constraints | Description |
|---|---|---|---|---|---|
| `complex_pdb` | string | yes | `—` | pattern: `\.(pdb\|ent)$` | Structure file with the target and its already-bound partner chain(s) in one file. Only plain PDB (or legacy .ent) is read -- this tool's parser (Biopython's PDBParser, via this server's own utils.pdb) does not read mmCIF. Convert first if you only have an mmCIF. WHERE THIS COMES FROM -- An existing complex -- your own structure, or one a folding tool returned (`run_boltz`, `run_chai1`, `run_alphafold3`). Use `run_epitope_scan` instead when you have only the unbound target. |
| `target_chain` | string | yes | `—` | pattern: `^[A-Za-z0-9]$` | Single-character chain ID of the residues to report and rank as candidate hotspots -- the chain you intend to design AGAINST in a future binder-generation call. Multi-character chain IDs are not supported (matches run_prodigy's chain convention). |
| `binder_chains` | array | yes | `—` | minItems: `1` | Chain ID(s) of the already-bound partner defining this interface -- e.g. an existing binder, an antibody's heavy+light pair, or a crystallization partner. A target_chain residue counts as a contact if it is within contact_cutoff of ANY of these chains. List more than one only when they form a single biological unit you want treated as one partner (e.g. an antibody Fab); otherwise call this tool once per partner chain. |
| `contact_cutoff` | number | no | `8.0` | minimum: `3.0`<br>maximum: `20.0` | CA-CA distance in Angstroms defining a contact, passed straight to utils.pdb.get_interface_residues. 8.0 is that function's own default -- a coarse, CA-only proxy for contact (see the doc's caveat), not a heavy-atom definition. Lower (e.g. 5.0) for a stricter interface; raising it only ever ADDS residues, never removes any (confirmed by that function's own existing tests). |
