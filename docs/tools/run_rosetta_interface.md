# run_rosetta_interface

**Category:** scoring  
**Engine:** `pyrosetta`  
**Environment:** `/home/jk661/.conda/envs/pyrosetta`  
**GPU required:** no

> This file is generated from `src/protein_design_mcp/manifests/run_rosetta_interface.yaml`. Edit the manifest, then run `python scripts/generate_tool_docs.py`.

## Summary

Physics-based protein-protein interface analysis with PyRosetta's InterfaceAnalyzerMover: binding energy (dG), buried surface area (dSASA), Lawrence-Coleman shape complementarity, and interface hydrogen-bond counts. NOT VERIFIED LIVE on this host -- read the doc's "Verification status" section before relying on this tool; use run_prodigy for a physics-based interface number that IS confirmed working here.

## What this is
PyRosetta's `InterfaceAnalyzerMover` run over an existing protein-protein
complex structure: it separates the named partners, optionally repacks
each side, and reports the Rosetta all-atom energy difference between
bound and separated states (`dG`), the buried solvent-accessible surface
area (`dSASA`), the Lawrence & Coleman shape-complementarity statistic
(`sc_value`, 0-1, higher is a tighter geometric fit), and interface
hydrogen-bond counts.

## Verification status -- read this before using this tool
This tool could not be exercised end to end on this host. The wheel this
environment installs from, `/opt/pyrosetta_wheels/pyrosetta-2017-cp312-cp312-linux_x86_64.whl`
(1.8GB), contains the pure-Python `pyrosetta` wrapper layer and the
~3.3GB Rosetta parameter database, but CONFIRMED LIVE (2026-09-22,
inspecting the wheel's own file listing and the installed package) it
contains **zero `.so` compiled extension files anywhere** -- the actual
`pyrosetta.rosetta` C++ binary bindings that every single PyRosetta
operation (including this one) depends on are simply absent from this
asset. `import pyrosetta` fails immediately and deterministically with
`ModuleNotFoundError: No module named 'pyrosetta.rosetta'`, before any
code this tool wrote runs at all. This is not an environment-configuration
problem this wave could fix (a different python version, extra pip
packages, or extra mounts would not help) -- the wheel itself is missing
its core binary. See the wave report for what an operator needs to do:
obtain a complete PyRosetta wheel (or the standard `.whl` + separately
distributed compiled `.so`) that actually bundles the compiled `rosetta`
extension for cp312/linux_x86_64.

Everything below (constructor arguments, output field names, and the
Lawrence-Coleman `sc_value` in particular) WAS independently confirmed
live on this same host, end to end, against a genuinely working PyRosetta
install found in a DIFFERENT, pre-existing environment
(`/home/jk661/.conda/envs/BindCraft`, python 3.10, used strictly
READ-ONLY to verify the API surface -- never modified, and this tool does
not depend on it) -- over `tests/fixtures/test_pdbs/1BRS.pdb` (the
barnase-barstar complex), chains `A_D`:
`InterfaceAnalyzerMover(DockingPartners.docking_partners_from_string("A_D"), False, scorefxn, True, False, True)`
followed by `.set_compute_interface_sc(True)` and `.apply(pose)` produced
`dG=210.3`, `dSASA=1574.0`, `sc_value=0.72` (a physically sane
shape-complementarity value for a real, tight protein-protein interface),
`interface_hbonds=13`, `delta_unsat_hbonds=10`, `packstat=0.55` -- so this
tool's wrapper script is built against a verified-correct API, even
though it cannot currently run in ITS OWN (differently-versioned, broken
wheel) environment. `run_prodigy` remains the confirmed-working choice
for a physics-based interface number on this host right now.

## `interface` -- Rosetta's own docking-partner notation, not reinvented
`"A_B"` for a simple two-chain interface, or `"AB_HL"` for a multi-chain
group on either side (e.g. an antibody's heavy+light chains as one side).
This is `DockingPartners.docking_partners_from_string`'s own format,
passed straight through rather than translated into a different
convention -- an underscore separates the two sides, and each side is one
or more concatenated single-letter chain IDs.

## Why `interface` is required and never inferred
Exactly like `run_prodigy`'s `chain_a`/`chain_b`: which chains are one
"side" of the interface and which are the other is the caller's decision
and changes the physical question being asked. A target-only pose with no
`interface` supplied would have nothing to separate at all.

## Repacking parameters -- these change the number, not just the speed
`pack_separated` (default true, this tool's own default -- matches the
legacy `pyrosetta_runner.py` PyRosetta wrapper this project already
carried) repacks each partner's side chains AFTER separation, which is
what makes `dG` reflect each partner's own relaxed, unbound state rather
than the bound conformation frozen in place -- turning it off is faster
but systematically biases `dG` more favourable (bound-state side chains
are not optimal for the separated state). `pack_input` (default false)
additionally repacks the input bound complex before scoring it -- off by
default because it can change the very geometry `sc_value`/`dSASA`
describe. `pack_rounds` (default 1, Rosetta's OWN documented default per
`set_pack_rounds`'s docstring: "Default is 1, but that certainly may not
be enough...") controls how many packing rounds run when either repacking
flag is on.

## What you must supply
`complex_pdb`: a structure file with both partners. `interface`: which
chains are on which side (see above) -- no default, must be stated.

## What you get back
`dG` (Rosetta energy units, bound minus separated -- more negative is a
more favourable interface), `dSASA` (buried surface area, Å²),
`shape_complementarity` (Lawrence-Coleman `sc_value`, 0-1), a plain
interface hydrogen-bond count (`interface_hbonds`) and the count of
UNSATISFIED interface hydrogen bonds after burial (`delta_unsat_hbonds`,
a common red flag when high), `packstat` (Rosetta's packing-quality
statistic, when `compute_packstat` was left on), and `num_interface_residues`.

## When to use this instead of the alternatives
- `run_prodigy` is a linear-regression estimate from contact counts alone
  -- much faster, needs no repacking, but calibrated on natural complexes
  and known to mis-rank de novo designs. This tool gives a full
  all-atom-energy decomposition instead, at much higher CPU cost, and is
  NOT confirmed working on this host right now (see above) -- use
  `run_prodigy` unless you specifically need this tool's fields once a
  working wheel is installed.
- `run_ipsae` and a co-folding tool's own `iptm` measure predictor
  CONFIDENCE, not physical interface quality -- a different question from
  either of the above.

## Parameters

| Parameter | Type | Required | Default | Constraints | Description |
|---|---|---|---|---|---|
| `complex_pdb` | string | yes | `—` | pattern: `\.(pdb\|cif\|ent)$` | Path to a structure file containing both interface partners. |
| `interface` | string | yes | `—` | pattern: `^[A-Za-z0-9]+_[A-Za-z0-9]+$` | Which chains form each side of the interface, in Rosetta's own docking-partner notation: one or more concatenated single-letter chain IDs, an underscore, then the other side's chain IDs -- e.g. "A_B" for a simple two-chain interface, "AB_HL" for a multi-chain group (e.g. an antibody's two chains) versus another. Never inferred -- which chains are which side is the caller's decision. |
| `score_function` | string | no | `ref2015` | pattern: `^[a-zA-Z0-9_]+$` | Name of the Rosetta all-atom score function to weight the energy calculation with (passed to pyrosetta.create_score_function). ref2015 is Rosetta's current standard full-atom score function and this tool's default; use a different name only if you specifically need a non-standard weight set installed in this Rosetta database. |
| `pack_separated` | boolean | no | `True` | — | Repack each partner's side chains AFTER separating them, before scoring the separated state. This tool's own default (matching this project's existing legacy PyRosetta wrapper) -- without it, dG is biased favourable because the separated partners keep their BOUND side-chain conformations instead of relaxing to their own optimum. |
| `pack_input` | boolean | no | `False` | — | Additionally repack the input BOUND complex before analysis, on top of pack_separated. Off by default -- it can change the bound-state geometry that dSASA/shape_complementarity describe, which is usually not what you want when scoring a structure you already trust. |
| `pack_rounds` | integer | no | `1` | minimum: `1`<br>maximum: `20` | How many packing rounds run when pack_separated and/or pack_input is on; ignored if both are false. 1 is Rosetta's own documented default (its docstring notes 1 round "certainly may not be enough" for a thorough repack) -- raise this for a more thorough (and slower) repack on a case where the default's single round leaves an obviously unconverged side-chain arrangement. |
| `compute_packstat` | boolean | no | `True` | — | Whether to compute Rosetta's packstat statistic (a measure of how well side chains at the interface are packed). Stochastic and adds meaningful runtime; this tool's default is true since packstat is one of this tool's returned fields -- set false to skip it and speed up the call when you only need dG/dSASA/shape_complementarity. |
| `compute_interface_sc` | boolean | no | `True` | — | Whether to compute the Lawrence-Coleman shape-complementarity statistic (this tool's shape_complementarity output). This tool's default is true since it is one of this tool's headline fields; set false only to skip its cost when you specifically do not need it. |
