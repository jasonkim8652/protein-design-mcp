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

## Verified publication and validation

Published on 2026-10-01 (UTC):

```text
jasonkim8652/protein-design-mcp:2.4.1@sha256:87e0db49d2170eff971d79e7d8a71951daddada6934cf4b1a2d3b2ecab4c03d5
```

The remote manifest config matches local image
`sha256:e905aadecb1a1dec0f26c7188adb53e241a9f079a1fda4cbea447d3d30bec19a`;
the source revision is `80c604351803acf4f44a1ba3fcac76386337287e`.
A pull using the immutable reference completed successfully. The image has
102 layers: 285,307,021,325 bytes uncompressed (about 266 GiB), with
194,770,828,068 compressed layer bytes (about 181 GiB). Docker storage
and run outputs require additional space; shared base layers can be reused.

- Server suite: 1,534 passed, 30 skipped.
- Client suite: 797 passed, including checkpoint replay, upstream argument
  repair, candidate failure isolation and shutdown regressions.
- New-image MCP discovery: all 41 tools with external assets mounted.
- Actual BoltzGen design: two structures and two NPZ files in 169 seconds,
  including engine startup and artifact collection, with zero DataLoader workers.
- Actual Docker/MCP cancellation: waiting caller released and container removed
  in 0.166 seconds.

A controlled one-worker run with 64 MiB shared memory failed in queue transport;
a one-worker control with the original 16 GiB shared memory completed. The
original live stall's precise trigger is therefore unresolved. Zero workers
bypass that multiprocessing path; this is not evidence that every one-worker
run fails. This validation does not claim that a complete three-round campaign
has finished or that all scientific engines have been re-executed.
