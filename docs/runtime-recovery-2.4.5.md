# Recoverable structure input validation

The campaign selected an earlier RFdiffusion3 structure for OpenMM after choosing
a different ProteinMPNN sequence. That input also retained partial target side
chains. The previous preflight detected residues with no side chain at all, but
allowed partially missing side chains into the engine, where PDBFixer refused
them as a generic engine error. The candidate then stopped instead of receiving
an opportunity to correct its input.

The server validates the required nonterminal heavy atoms of canonical protein
residues before dispatch. Missing terminal OXT and hydrogens remain supported by
the unchanged preparation protocol. Incomplete or malformed inputs return
`argument_validation` with bounded residue and missing-atom details. Permission
and storage failures retain their infrastructure classification. Modified or
unknown residues still require the native engine's template support; this check
does not invent their chemistry or reconstruct missing coordinates.

The campaign client also verifies that sequence-preserving structure consumers
receive a structure containing the selected candidate sequence. Renamed chains
and supported PDB/mmCIF/gzip formats remain valid. An incompatible input is
returned to the executor for bounded argument correction; no generator,
predictor, or replacement file is chosen automatically.

The original design prompts, saved strategies, independent evaluator and score
feedback are unchanged. Numerical measurement failures remain distinct from
input validation, and UNMEASURABLE observations retain null scores. The OpenMM
engine script and force-field/minimization protocol are unchanged. Existing
campaigns retain their original pinned independent evaluator configuration.

## Verified publication

Immutable image: `jasonkim8652/protein-design-mcp:2.4.5@sha256:b6b81defafb145881c5f21eb3e5937d93fb9f1878afa7d9b5b6a261327219e91`. Source revision: `f3d41f1`.
Remote manifest identity matches the validated local image; pulling the
immutable digest succeeded.

Registry layer payload: **181.41 GiB**
(194,790,201,414 bytes). Docker reports **265.78 GiB**
uncompressed (285,378,900,166 bytes). Existing integrated base
layers are reusable. Allow additional space for extraction and run artifacts.

The executor retains factual argument-rejection history across retries and
checkpoint resume so a later correction can account for every prior refusal.
It does not choose a replacement file or change the role prompt.

Validation covered the full server/client regression suites, all canonical
residue atom sets, supported structure formats, bounded input retries and
cached-step invalidation. Live MCP checks reject the original partially
missing target and an incomplete refold before engine execution. An isolated
replay used the original saved generation/prediction prefix and the original
executor prompt with gpt-5.6-terra to correct the failed OpenMM input, then
execute OpenMM and Rosetta. This replay does not run the independent assay
and is not recorded as a campaign observation. The campaign remains stopped.
