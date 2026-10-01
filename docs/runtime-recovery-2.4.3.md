# BoltzGen analysis handoff validation

The analysis tool requires full-complex refold structures and metrics matching
the generated design IDs and complete chain sequences. A binder-only refold
can have the same file basename while representing a different molecule.
Previously the engine skipped every mismatched sample and failed during metric
aggregation. The adapter now rejects incompatible inputs before launching the
analysis process, with a structured argument_validation error the client can
return to the model for correction. It checks IDs, complete CIF identities and
NPZ token identities with pickle disabled.

Optional design-only refolds remain separate inputs. No structures, sequences,
metrics or caller-selected folding modes are silently changed. Exact atom-token
expansion for noncanonical polymers and the optional designed subset mask remain
engine responsibilities; the adapter validates their available paired identities.

The client also allows a bounded model-requested return to an earlier successful
step before a tool rejects its input. It records that request separately from an
actual tool rejection, invalidates dependent results, and preserves checkpoints
and restart budgets. Original role prompts and independent assay code are unchanged.

For campaigns with existing observations, preserve the original evaluator image,
launch configuration, database and scoring protocol when updating the design
backend. Existing complete observations can then be reused on resume.
