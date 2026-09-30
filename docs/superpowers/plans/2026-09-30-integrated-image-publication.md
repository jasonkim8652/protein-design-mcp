# Integrated Image Publication Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Publish the integrated 2.4.0 image with current engine fixes and practical registry layers.

**Architecture:** Keep the audited uncompressed payload unchanged. Generate a Dockerfile that extracts contiguous TAR member ranges into separate layers, targeting 4 GiB per layer; a single larger file remains intact. Preserve member order so cross-layer hard links resolve. Install current source after the payload, then validate and publish.

**Tech Stack:** Python tarfile, GNU dd/tar, Docker BuildKit, Docker Hub.

- [ ] Add tests comparing normal TAR extraction against range extraction, including hard links across layers, symlinks, long names, file modes and an oversized file.
- [ ] Implement the range planner and Dockerfile generation; fail closed if the template marker is missing or input is compressed.
- [ ] Update build instructions to generate the layered Dockerfile from the audited payload.
- [ ] Run focused packaging tests, review changes, and commit current release source.
- [ ] Build with the exact source revision, validate installed fixes, tool discovery and independent AF2/OpenMM smoke execution.
- [ ] Push 2.4.0, verify its registry digest, and update the client pin and distribution documentation only after verification.

The completed campaign and original prompts are unchanged. Evaluator scoring changes are a separate decision.
