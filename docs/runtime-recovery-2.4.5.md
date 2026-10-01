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
