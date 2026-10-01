# External-backbone folding and recoverable path errors

The validation-score campaign exposed two input-boundary failures. Earlier
releases repaired native BoltzGen inverse-fold/analysis handoffs and engine IPC;
they did not provide a folding input mode for an external designed backbone.

`run_boltzgen_fold` now accepts an observed PDB/mmCIF backbone (including gzip),
explicit design-chain IDs, and a designed sequence for each designated chain.
The tool uses BoltzGen's native parser, featurizer and writer on CPU to construct
the paired native artifacts, then runs the same BoltzGen folding engine. It
exports those artifacts and a portable design specification for downstream
analysis. Supplied backbone coordinates and target identity are preserved;
sidechains inconsistent with the supplied sequence are removed, not invented.
CA-only inputs require backbone reconstruction first. The native CIF/NPZ mode
continues to work and accepts selected complete pairs.

Malformed or missing staged source paths now return a bounded
`argument_validation` error before inference. The client can correct the
arguments rather than failing the whole candidate. Permission, storage and I/O
failures remain infrastructure errors. Raw failed inputs and engine artifacts
remain available in campaign records.

The client separately fixes a quadratic credential-redaction search that made
large sequence/artifact records expensive to archive. Secret-redaction coverage
is preserved. This is an observability fix, not a scientific protocol change.

Campaign prompts, generator choice, candidate allocation and independent
AF2/local-MSA/OpenMM scoring remain unchanged. When updating the design backend
of an existing campaign, keep its evaluator image and evaluation protocol pinned.
