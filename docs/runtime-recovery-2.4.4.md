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

## Verified publication

Published immutable image: `jasonkim8652/protein-design-mcp:2.4.4@sha256:bb01fe6f748f9729aba45f9f1e6038266777ae65e789f06702a29d29bcee966e`.

Server source revision: `2d899b1`. Remote manifest config digest
matches the validated local image, and pulling the immutable reference succeeded.

Registry layer payload: **181.41 GiB**
(194,785,314,182 bytes). Docker reports **265.76 GiB**
uncompressed (285,360,819,939 bytes). Shared base layers are reused
when the previous integrated image is already present; allow additional storage
for extraction and run artifacts.

Validation includes full server and client regression suites, native CPU
roundtrips of actual RFdiffusion3 and rebuilt Genie3 artifacts, nonstandard
chain IDs, and live folding followed by analysis for external RFdiffusion3,
external Genie3, and a selected native BoltzGen design pair. Each live path
analyzed one design; all 41 tool definitions were discoverable with the
configured external assets. Malformed path input returned a bounded, recoverable
argument error. These smoke checks do not claim campaign completion.
