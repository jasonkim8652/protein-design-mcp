# Runtime recovery in 2.4.1

The patch retains the integrated environments and model assets from the immutable
2.4.0 image. Build it with `Dockerfile.runtime-patch` and a committed
`SOURCE_REVISION`; databases and restricted assets remain external mounts.

GPU BoltzGen tools use `num_workers=0`. Positive values are rejected before
execution with an actionable validation error. This avoids the multiprocessing
DataLoader queue path that stalled a live design call; model, sampling and design
parameters remain caller-selected. A controlled single-process run completed both
designs with the original checkpoints and sampling settings.

Engine subprocess output is forwarded while the engine runs. Retained stdout and
stderr files preserve full logs; response capture is bounded, so large responses
carry an omission marker and retain their beginning and end. Consumers requiring
full output should read the retained files or declared output artifacts.

Explicit alignment input refusals expose `error_kind: argument_validation`,
including an MSA query that differs from the requested target chain. Such inputs
remain invalid: the backend does not trim terminal residues or silently substitute
sequences. Engine timeouts are classified separately after process cleanup.

These changes do not alter the independent AF2/OpenMM assay or its score formula.
For a campaign with existing measurements, retain its pinned evaluator image and
configuration; a design-backend update does not authorize mixing assay protocols.
