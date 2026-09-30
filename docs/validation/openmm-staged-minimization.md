# Staged OpenMM minimization validation

The `run_openmm_minimize` engine uses the fixed protocol
`soft-repulsion-flexible-hbonds-v1`. Complex, binder and target must each run the
same protocol independently. This changes the minimization procedure and must be
locked into a new assay version; do not silently mix these energies with older
runs. It does not change the selected final Amber/CHARMM force field or add
solvent.

## Procedure and rationale

1. PDBFixer adds missing terminal heavy atoms only, using explicit Reference and
   seed 20260930. Missing side chains/internal heavy atoms are rejected; missing
   residues are not reconstructed.
2. OpenMM `Modeller.addHydrogens(forcefield=None)` uses its simplified hydrogen
   placement model on Reference, with Python/NumPy seed 20260930. This avoids
   placing hydrogens under the singular full force-field interactions of a
   severely clashing heavy-atom structure. Selected residue variants and atom
   ordering are recorded. This placement model is not the final energy model.
3. A 200-step preparation stage retains flexible force-field bonds/angles/torsions,
   replacing nonbonded terms with `0.5*k*max(0,sigma_ij-r)^2`, where
   `k=10000 kJ mol^-1 nm^-2` and `sigma_ij` is the mean of the atom LJ sigmas.
   Force-field excluded pairs remain excluded. Heavy atoms have harmonic
   restraints to prepared coordinates with `k=1000 kJ mol^-1 nm^-2`.
   CHARMM sigmas come from the diagonal tabulated LJ coefficients; its singular
   custom LJ and 1-4 LJ terms are removed for this stage. Repulsion is bounded
   at small separation, while stronger bonded terms discourage bond distortion.
   The positional restraint is weaker than overlap repulsion and limits movement
   during preparation. These are fixed stabilization choices, not fitted binding
   parameters or a calibrated physical scoring model.
4. A 200-step unrestrained stage uses the complete selected physical force field
   with flexible bonds. Water rigidity is disabled in both flexible stages.
5. The final stage uses the original physical force field with HBonds constraints
   and the requested cap (default 500 per constraint pass). Initial and final
   reported energies both use this final System, without temporary restraints or
   soft forces. Every stage uses OpenMM's 10 kJ mol^-1 nm^-1 tolerance. CUDA double
   with deterministic forces is the default; CUDA mixed, CPU or Reference require
   explicit selection. There is no automatic platform fallback or energy-ranked
   selection among retries.

Constraint enforcement can restart L-BFGS. `iterations` is the observed sum of
reporter callbacks, and can exceed the requested caps; it is not evidence of
convergence. Diagnostics include each stage's callback count, observed passes,
last callback, physical force RMS, constraint error, coordinates, geometry and
heavy-atom displacement without alignment. Raw physical force RMS includes
components along constrained bonds and is not a projected convergence test.

The engine saves prepared topology/coordinates, each stage's serialized System
and full precision NumPy coordinates, SHA256 identities, input and engine hashes,
OpenMM version, selected platform properties and preparation variants. Coordinates
are in nm in `.npy`; PDB output coordinates are rounded. Re-evaluate saved `.npy`
with its matching System when reproducing energies.

## Controlled regression evidence

Validated on OpenMM 8.6.1 with the packaged CUDA 12.9 runtime, image
`sha256:d15bc5ffa76060305a4cba316e10eac9312a15bd533707422af24a61f6d224ce`,
explicit CUDA double, without the earlier diagnostic NVRTC library replacement.
Engine SHA256:
`9aa69c8420c4f14fcd222e284bd9b089295268d7b398df4b68a5878800ce9cf7`.

| Raw AF2 component | Final energy (kJ/mol) | Heavy displacement RMS / max (Å) |
| --- | ---: | ---: |
| Previously passing r001_a001_d001 binder | -3806.39 | 2.427 / 8.152 |
| Previously failing r001_a001_d002 binder | -5981.59 | 2.884 / 7.140 |
| r002_a002_d001 complex | -54900.54 | 1.727 / 7.677 |
| r002_a002_d001 binder | -11309.87 | 1.182 / 4.769 |
| Previously failing r002_a002_d001 target | -43537.72 | 1.778 / 7.846 |
| Previously failing r003_a001_d001 binder | -2963.81 | 1.801 / 7.742 |

All six final structures had zero heavy-atom contacts below 1 Å and no checked
intra-residue N–CA, CA–C or C–O gross bond distortion. The pathological target's
previous 3.3 Å N–CA distortion was absent. Each run recorded 2400 callbacks total,
including four final constraint passes of up to 500 iterations. Final maximum
relative constraint errors were about 5e-6. Replaying all six final serialized
Systems and full precision states reproduced their reported energies within
0.001 kJ/mol and confirmed the absence of temporary forces. Tests also verified
coordinate/System hashes, engine identity and final geometry.

These six regressions demonstrate recovery of selected known failures and one
previously passing control, not universal recovery or convergence. The table
shows substantial coordinate motion, including the control. Neither passing
these gross checks nor obtaining a lower energy establishes a correct pose or
binding affinity. Even changing hydrogen/terminal preparation can alter the
local minimum and independent energy difference substantially. No historical
scores or original prompts were rewritten.

Machine-local regression inputs, logs and replay states are preserved under
`/home/jk661/projects/biodesignbench/tmp/operations/openmm-robust/final-validated/`.
The reproducible replay test is `tests/test_engine_openmm.py`; set
`OPENMM_REGRESSION_ROOT` to that directory in the md environment with CUDA exposed.
Without it, the expensive artifact regression is explicitly skipped.

## Failure contract

Grossly invalid final geometry returns `geometry_passed=false`. Explicit
nonfinite numerical results and OpenMM NaN-coordinate errors return
`numerical_failure=true` with a null final energy and diagnostic artifacts. They
must become an unmeasurable assay, never a favorable score. Missing CUDA,
incompatible PTX, missing dependencies and force-field/template errors remain
execution failures. They are not silently treated as molecular failures.

Only the diagnostics artifact is mandatory on a numerical failure. Optional
output declarations permit absent minimized coordinates and states; successful
or geometry-failed completed runs still require their actual collected minimized
PDB in the adapter. Prepared coordinates cannot substitute for a missing final
structure. Artifact collection preserves diagnostics before scratch cleanup.

Targeted validation: 137 adapter, collection, manifest and documentation tests
passed; eight md engine tests passed with the six-case replay enabled. The full
server suite also ran in the host tooling Python environment: 1470 passed,
15 skipped and 15 unrelated environment/configuration failures. The failed tests
were:

- `tests/test_alphafold2.py::TestAlphaFold2Config::test_default_config`
- `tests/test_esmfold.py::TestESMFoldRunner::test_predict_structure_validates_sequence`
- `tests/test_pdb_utils.py::TestValidatePdb::test_validate_nonexistent_file`
- `tests/test_pdb_utils.py::TestParsePdb::test_parse_nonexistent_file_raises`
- `tests/test_proteinmpnn.py::TestProteinMPNNRunner::test_design_sequences_returns_list`
- `tests/test_proteinmpnn.py::TestProteinMPNNRunner::test_design_sequences_validates_backbone`
- `tests/test_proteinmpnn.py::TestProteinMPNNRunner::test_design_sequences_creates_output_dir`
- `tests/test_proteinmpnn.py::TestProteinMPNNRunner::test_design_for_interface`
- `tests/test_proteinmpnn.py::TestEdgeCases::test_single_sequence`
- `tests/test_proteinmpnn.py::TestEdgeCases::test_many_sequences`
- `tests/test_rfdiffusion.py::TestRFdiffusionRunner::test_generate_backbones_returns_list`
- `tests/test_rfdiffusion.py::TestRFdiffusionRunner::test_generate_backbones_validates_target`
- `tests/test_rfdiffusion.py::TestRFdiffusionRunner::test_generate_backbones_creates_output_dir`
- `tests/test_rfdiffusion.py::TestEdgeCases::test_many_designs`
- `tests/test_sasa.py::TestCalculateSASA::test_invalid_pdb_raises_error`

The first expects an old MSA mode default; ESMFold lacks torch; remaining failures
are unavailable/unwritable `/nonexistent`, `/opt/ProteinMPNN` and `/models` test
paths. Exact logs and the command are under the same machine-local regression
root's parent (`full-suite.log`).

The same 15 failing node IDs were replayed against an untouched archive of
parent revision `bdb5329` with its own source directory on `PYTHONPATH`; all 15
failed there as well. This confirms these failures predate this change. The
comparison log is `baseline-failures.log` beside the full-suite log.
