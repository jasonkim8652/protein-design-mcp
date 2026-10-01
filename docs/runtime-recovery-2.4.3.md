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

## Verified publication

Published and pulled by immutable digest on 2026-10-01T06:21:15.957340+00:00:

```text
jasonkim8652/protein-design-mcp:2.4.3@sha256:0b0a40d409374e0d858292a7c09cd7aa386a9354b8bf3b9be4fda30d1ff3fad0
```

Image source: `a85179b013582f5ee7e6644eb1ab2d38ef34c546`. The remote manifest configuration
digest matches the locally validated image. The image exposes all 41 tools.
Compressed layers total 194,780,431,813 bytes
(181.40 GiB); local uncompressed size is
285,342,751,434 bytes (265.75 GiB).
Existing shared layers can reduce the download. Allow extra disk space for
Docker extraction, retained artifacts and user-supplied databases.

Validation: 1,554 server tests passed, 30 skipped; 808 client tests passed.
A live container rejected the observed monomer-as-complex input in 0.26 seconds
with `argument_validation`; a saved valid complex input analyzed all 16 designs
in 80.07 seconds. No independent assay was rerun for this validation.
