# Portable AF2/OpenMM validation — 2.3.6

The image includes an isolated ColabFold 1.6.3 / AlphaFold-ColabFold 2.3.20 /
JAX CUDA 0.6.2 environment and all five AF2-Multimer v3 parameter sets under
`/opt/weights/colabfold`. Unused parameter sets from the upstream archive are
removed in the download layer. The image is approximately 12 GB unpacked.

## Live proof

On 2026-09-29, the evaluator from proteinmem-mcp's isolated-blackbox-evaluator
branch (1d2013235c4cb29d5121529e8080ada9dbcad68a) completed with SUCCESS.
The container had networking disabled, a single GPU exposed, and only an
input/output workspace mount. No host engine environment, weights cache, or
home directory was mounted. Inputs were the first 30 residues of the 5WB7
target and the synthetic binder EIAALEKEIAALEKEIAALE. AF2 used the normal
five-model defaults; each OpenMM minimization allowed 50 iterations.

| Quantity | kJ/mol |
| --- | ---: |
| Complex potential energy | -1636.8873 |
| Binder potential energy | 704.0054 |
| Target potential energy | -1970.7458 |
| DeltaE (complex - binder - target) | -370.1469 |

This verifies infrastructure and artifact handoffs, not full-target binder
quality or a three-round optimization campaign. DeltaE is a potential-energy
proxy, not a binding free energy. Exact energies can vary between runs.

AF2 initially succeeded while OpenMM failed because its predicted chains
lacked terminal OXT atoms. The retained AF2 PDB fixture reproduces that failure.
`tests/test_openmm_engine.py` passed inside the built image's md environment,
verifying successful minimization, addition of two terminal atoms, finite
energies, and preservation of the original input. No loop or side-chain
reconstruction is performed.

Config generation also succeeded as an arbitrary UID/GID (12345:12345) inside
a network-disabled container with no host mounts. It produced a mount-free
launch command and reported unavailable optional engines instead of crashing.

## Tests and baseline limitations

Focused adapter, loader, registry, container-command and document checks pass.
The full legacy server suite has 15 existing failures, reproduced from the
unchanged parent commit c78761b in a separate source snapshot with the same
Python environment:

- test_alphafold2.py: TestAlphaFold2Config.test_default_config (old MSA default).
- test_esmfold.py: TestESMFoldRunner.test_predict_structure_validates_sequence
  (legacy runner imports unavailable torch).
- test_pdb_utils.py: TestValidatePdb.test_validate_nonexistent_file and
  TestParsePdb.test_parse_nonexistent_file_raises (the sentinel /nonexistent
  path is not traversable on this host).
- test_proteinmpnn.py: TestProteinMPNNRunner.test_design_sequences_returns_list,
  test_design_sequences_validates_backbone, test_design_sequences_creates_output_dir,
  test_design_for_interface; TestEdgeCases.test_single_sequence and
  test_many_sequences (legacy /opt or sentinel path permissions).
- test_rfdiffusion.py: TestRFdiffusionRunner.test_generate_backbones_returns_list,
  test_generate_backbones_validates_target, test_generate_backbones_creates_output_dir;
  TestEdgeCases.test_many_designs (legacy /opt or sentinel path permissions).
- test_sasa.py: TestCalculateSASA.test_invalid_pdb_raises_error
  (sentinel path permission).

The suite therefore must not be described as entirely green. These legacy v1
runners are not used by the manifest-driven AF2/OpenMM smoke test.
