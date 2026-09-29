# Integrated 2.4.0 validation

This report separates packaging checks, runtime loading, tool discovery, and
actual inference. A discoverable tool is not proof of a successful model run.

## Packaging checks

- Nineteen additional conda prefixes were copied and relocated; installed files
  and binary sizes were checked. Core environments come from `Dockerfile.envs`.
- AF3's separate Python 3.12 environment and source were staged independently.
- The public asset allowlist expanded to 55,228 files. No staged file was missing;
  45,436 model/data file sizes matched their sources.
- The completed payload archive is approximately 203 GiB. Shared identical
  runtime files were deduplicated inside the staging tree, saving 54.71 GiB.
- AF3 parameters, PyRosetta distributions, credentials and user databases are
  excluded from the payload. Component licenses and public asset notices are
  retained in the image.

## Automated checks

Focused packaging, manifest, registry, container-launch and generated-document
checks passed (147 tests). Portable engine paths and the affected Complexa and
MultiFlow adapters passed a separate 63-test selection.

The earlier full host suite reported 1,404 passed, six skipped and 25 failed.
Ten deployment-contract failures were subsequently fixed and their focused
checks passed. Fifteen legacy host-suite failures remain: old pipeline output
paths/host dependencies, inaccessible nonexistent-path fixtures, and an old
AF2 default expectation. These are not represented as a passing full suite.
The final host-suite rerun after the Rosetta/MPNN fixes reported **1,425 passed,
six skipped and the same 15 legacy failures** (201.9 seconds). The 42-test
Rosetta/MPNN selection passed, including missing and overlapping partner chains.

## Image and live execution

Final runtime image: `sha256:46a7dcb772d5c4a1f16f856b4cead22c80e4dc6f9e2eb281c891780657c01053`.
Installed package and OCI version: **2.4.0**. Source revision:
`3c127d3` (full revision retained in the OCI image label).
Uncompressed Docker image size: approximately 218 GiB.

The runtime matrix and offline assay checks below used the preceding image
`35038a524d97`, with the same runtime/weight payload. Follow-up live campaign
checks exposed and fixed Rosetta's acceptance of absent partner chains and
MPNN CPU oversubscription. The final image rejects a monomer requesting `A_B`,
successfully scores the actual two-chain complex, and generates eight MPNN
sequences through MCP in 7.19 seconds with four engine-local CPU workers.

- A fresh, network-disabled container running as UID/GID 65532 loaded all
  **26 runtime groups**. No host environment, engine checkout or public-weight
  cache was mounted.
- **18 PyTorch CUDA** matrix tests and **three JAX GPU** operation tests passed
  on the allocated NVIDIA L40S. These are runtime checks, not model inference.
- Known-path exclusion checks found no bundled AF3 parameters or PyRosetta
  distribution. PyRosetta imports successfully when its external package mount
  is supplied; its resolved module location is under `/data/licenses/pyrosetta`.
- MCP discovery is **READY with all 41 tools** when the four mandatory external
  asset categories are mounted. The optional ColabFold local database is absent.
- With only the workspace mounted and networking disabled, **37 tools** are
  discovered and AF2-Multimer/OpenMM preflight is **READY**.

Two independent evaluator smoke runs succeeded with networking disabled and
only the workspace mounted. Both used bundled AF2-Multimer weights, all five
multimer-v3 models, and separately minimized the predicted complex, binder and
target with OpenMM. Neither run used an LLM.

| Run | Target residues | OpenMM iteration limit | Delta E (kJ/mol) | Elapsed |
| --- | ---: | ---: | ---: | ---: |
| `integrated-240-offline-smoke-v2` | 30 | 50 | 457.1581 | 67.6 s |
| `integrated-240-full-target` | 504 | 500 | 504.6455 | 196.8 s |

The full-target run used the 18-residue synthetic smoke binder
`EIAALEKEIAALEKEIAALE`. Component energies were -39,623.0187 (complex),
-110.7777 (binder), and -40,016.8865 (target), in kJ/mol. The reported force
field was `amber14-all.xml`, with no solvent. Delta E is a computational
potential-energy proxy, not measured binding affinity or a binding free energy.
These checks establish execution of the assay; they do not validate binder
quality or successful inference by every one of the 41 discoverable tools.

The ProteinMEM campaign uses this assay after committing each model-generated
candidate. Pipeline code invokes the evaluator independently of the LLM, fixes
the evaluation protocol, and passes the resulting measurements to subsequent
rounds. Live multi-round results are recorded in the client validation report.

## BoltzGen structure handoff correction

A live model-selected workflow exposed an incomplete tool interface: the design
step produced coordinates, but standalone inverse folding still demanded a
caller-authored fixed-structure YAML. Passing the generation spec correctly
failed because its binder was specified as a length range.

`run_boltzgen_inverse_fold` now accepts `structure` plus explicit `design_chains`.
The tool validates the chain IDs, builds its YAML internally, and exports the
spec for subsequent fold/analyze/filter calls. Advanced YAML input remains
available as a mutually exclusive route. ProteinMEM does not author these files.
A real generated 554-residue complex completed inverse folding through MCP in
27.85 seconds; the output included redesigned coordinates, metadata and the
correct fixed-structure spec. The BoltzGen selection passed 118 tests, with
37 tests passing after adding wrapper export/relative-path coverage. A stale
handoff test that required caller-authored YAML was updated; all nine handoff
matrix tests then passed. The full host run before that test update reported
1,431 passed, six skipped and 16 failed (the same 15 legacy failures plus that
stale documentation assertion); this is not a green full-suite claim.

The final image also passed fresh MCP discovery as UID/GID 65532 with networking
disabled, an unrelated `/tmp` workspace, and no developer home mount: 37 tools,
including AF2-Multimer and OpenMM, were available. This checks discovery rather
than asserting successful inference for every tool.

## Chai and campaign archival follow-up

A full campaign exposed a Chai manifest pointing at `/opt/models/chai1`, while
its packaged assets live in the Chai environment's native `downloads` directory.
The manifest now points at those bundled files and checks all required assets
at discovery. The exact failing 72-residue binder plus 504-residue target
completed successfully in 227.15 seconds; ESM scoring followed in 14.31 seconds.

The final image enables optional retained scratch with
`PROTEIN_MCP_KEEP_WORKDIR=1`. Full stdout/stderr is tee-copied from subprocess
pipes, preserving descendant lifetime and timeout cleanup. Success, engine
failure, timeout, and parser failures expose retained paths for client archival.
MCP request metadata may shorten the engine deadline, allowing cleanup and
partial-file metadata to reach the client before its transport timeout.

On this final image, a real epitope scan and a deliberately timed-out ESM call
both returned retained files that ProteinMEM copied and hashed in its campaign
archive. Discovery returned all 41 tools with external assets, and 37 in a
fresh network-disabled container as UID/GID 65532 with only an unrelated writable
workspace mounted. No developer-home mount was needed for the latter check.

The 74-test retention, transport timeout, portable-path and wiring selection
passed. The final full host suite reported **1,442 passed, six skipped and 15
legacy failures** in 203.77 seconds. The failures remain in old AlphaFold2,
ESMFold, ProteinMPNN and RFdiffusion pipeline expectations and inaccessible
nonexistent-path fixtures in PDB/SASA tests; the full host suite is not green.
Image publication is tracked separately from these local-image validations.
