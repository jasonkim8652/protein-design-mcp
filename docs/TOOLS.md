# protein-design-mcp — tools

**41 tools** (39 `run_*` + 2 meta), classified by **function** — not by which engine a
tool came from.

---

## 1. `binder_generation` — generate a binder against a target (7)

| Tool | Engine |
|---|---|
| `run_proteina_complexa_generate` | Proteina-Complexa 160M |
| `run_boltzgen_design` | BoltzGen 0.3.2 |
| `run_rfdiffusion_binder` | RFdiffusion 1.1.0 |
| `run_rfdiffusion2` | RFdiffusion2 |
| `run_rfdiffusion3_binder` | RFdiffusion3 |
| `run_genie3_binder` | Genie 3 |
| `run_protpardelle` | Protpardelle-1c 1.3.2 |

## 2. `monomer_generation` — generate a monomer or scaffold (6)

| Tool | Engine |
|---|---|
| `run_rfdiffusion3_scaffold` | RFdiffusion3 |
| `run_genie3_scaffold` | Genie 3 |
| `run_genie2` | Genie2 |
| `run_frameflow` | FrameFlow |
| `run_multiflow` | MultiFlow |
| `run_la_proteina` | La-Proteina |

## 3. `sequence_design` — design a sequence for a fixed backbone (2)

| Tool | Engine |
|---|---|
| `run_mpnn` | ProteinMPNN / LigandMPNN / SolubleMPNN |
| `run_boltzgen_inverse_fold` | BoltzGen's own inverse-folding head |

## 4. `structure_prediction` — predict a structure (11)

| Tool | Engine | MSA |
|---|---|---|
| `run_esmfold2` | ESMFold2 | takes none |
| `run_chai1` | Chai-1 0.6.1 | optional |
| `run_boltz` | Boltz-2 2.2.1 | optional |
| `run_protenix` | Protenix v1 | optional |
| `run_openfold3` | OpenFold3 | optional |
| `run_promera` | Promera | optional |
| `run_rf3` | RoseTTAFold3 | optional |
| `run_alphafold3` | AlphaFold 3 | optional (paired + unpaired) |
| `run_alphafold2_multimer` | AF2-Multimer / ColabFold | optional |
| `run_boltzgen_fold` | BoltzGen (refolding mode) | none |
| `run_boltzgen_fold` with `with_target: false` | BoltzGen (designfolding mode) | none |

No tool here requires an alignment, and none builds its own — see below.

## 5. `msa` — build an alignment (2)

| Tool | Engine |
|---|---|
| `run_mmseqs_search` | MMseqs2 vs the local 1.3 TB databases |
| `run_colabfold_search` | `colabfold_search` vs UniRef30 / envDB |

## 6. `scoring` — score an existing structure (4)

| Tool | Engine |
|---|---|
| `run_ipsae` | ipSAE — reads the PAE of run_alphafold2_multimer, run_boltz or run_alphafold3 (not Protenix/Chai-1: different key names) |
| `run_prodigy` | PRODIGY |
| `run_rosetta_interface` | PyRosetta InterfaceAnalyzer |
| `run_esm_score` | ESM2-650M / ESM-C |

## 7. `run_analysis` — operate on a finished run's outputs (4)

| Tool | Engine | Runs a model |
|---|---|---|
| `run_proteina_complexa_filter` | Proteina-Complexa | no |
| `run_boltzgen_analyze` | BoltzGen | no — CPU metrics + aggregation |
| `run_boltzgen_filter` | BoltzGen | no |
| `run_proteina_complexa_analyze` | Proteina-Complexa | no (foldseek / mmseqs) |

## 8. `target_analysis` — find where to bind (2)

| Tool | Input | Produces |
|---|---|---|
| `run_interface_residues` | an existing complex | per-residue contacts and buried surface area, in the hotspot formats the binder tools accept |
| `run_epitope_scan` | an unbound target | candidate surface residues ranked by exposure and conservation, with the evidence exposed |

Four binder-generation tools require hotspots — `run_rfdiffusion_binder` (`hotspot_res`),
`run_genie3_binder` (`hotspot_residues`), `run_protpardelle` (`hotspots`),
`run_rfdiffusion3_binder` (`select_hotspots`) — and until this category existed **nothing
produced them**. The same hole as the MSA one: a required input no tool could supply.

`run_epitope_scan` is the successor to the old composite `suggest_hotspots`, and differs
from it in the way that matters: it **exposes the evidence and lets the caller choose**
rather than collapsing hardcoded weights into a single answer.

## 9. `preparation` — modify a structure before scoring (1)

| Tool | Engine |
|---|---|
| `run_openmm_minimize` | OpenMM 8.6 |

## 10. `meta` (2)

| Tool |
|---|
| `describe_tool` |
| `get_job_status` |

---

## MSA is always supplied, never generated inside a folding tool

A folding tool that builds its own alignment is doing two steps — search, then fold —
and hiding one of them. The alignment comes from category 5 and nowhere else.

| `msa` value | meaning |
|---|---|
| `null` | run MSA-free |
| `<path>` | use this alignment |
| absent | rejected — the choice is stated, never inherited from a default |

There is no `"auto"`. Four reasons it was dropped:

1. **It hides a step this server registers separately** — the same rule that removed
   `run_proteina_complexa_evaluate`.
2. **It makes model comparison impossible.** If AlphaFold 3 searches its own databases
   while Chai-1 searches ColabFold's, the difference between two predicted structures
   confounds the model with the alignment. One a3m into all nine is the only clean
   comparison.
3. **It is how a sequence silently leaves the machine.** Several of these engines
   default to a *remote* MSA server, and the sequences here are usually novel designs.
4. **It hides cost and duplicates work.** A search over 1.3 TB is slow; under `"auto"`
   the caller can neither see nor control it, and nine tools rebuild the same alignment
   nine times instead of reusing one.

**Sequencing — the MSA tools land first.** AlphaFold 3's native pipeline produces a
paired MSA, an unpaired MSA and templates, not one plain a3m. `run_mmseqs_search` must
be able to emit each consumer's required shape *before* `"auto"` is removed anywhere;
otherwise a working path is replaced by a broken one. Each MSA tool states which
consumers its output is valid for, and each `msa` parameter states which producer it
expects — MMseqs2 against AlphaFold 3's databases and `colabfold_search` against
UniRef30/envDB search different sequence universes, and a consumer handed the wrong one
usually returns worse results rather than failing.

AF3 also carries a coupling rule the schema must enforce: `unpairedMsa` and `pairedMsa`
are either both set or both null.

## Parameter policy

Every knob the engine exposes is exposed here. A parameter is hidden only when it
selects a step this server registers separately, or when it would let a caller escape
the workdir.

- **`chains` is never inferred.** Whether a prediction runs with the target present or
  on the binder alone is the caller's decision and one of the most consequential it
  makes — the same binder predicted alone and in complex are different experiments.
- **Every description says what the parameter does, what changes when it moves, and a
  sensible range.** A type is not a description.
- **Defaults are documented as choices with a reason**, not stated as facts.

## Excluded by design

BindCraft, FreeBindCraft, ColabDesign, BoltzDesign1, mosaic, Odin-Multi, EasyNano,
dl_binder_design, RFantibody — gradient-based hallucination through a folding model.
`complexa design`, `boltzgen run`, Promera's `Design` task — orchestrators over steps
exposed here individually. `run_proteina_complexa_evaluate` — bundled refolding,
interface analysis and force-field metrics that are all exposed separately, and hid a
`binder is last chain` convention. Boltz-2's affinity head — protein-ligand only.
Protenix v2, SeedProteo, Chai-2, AlphaProteo, Pearl, Proteina (original), Chroma — no
public or permissive weights. Remote MSA servers — they transmit the caller's sequence
off the machine.
