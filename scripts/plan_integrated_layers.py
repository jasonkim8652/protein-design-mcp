#!/usr/bin/env python3
"""Generate registry-sized Docker layers without recopying the curated TAR.

Only use an audited, uncompressed archive produced by assemble_integrated_payload.
Ranges start and end at complete TAR members, preserving long-name headers and
hard-link order. A file larger than the target stays intact in one layer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tarfile


MARKER = (
    "RUN --mount=type=bind,from=payload,source=rootfs.tar,target=/tmp/rootfs.tar \\\n"
    "    tar -xf /tmp/rootfs.tar -C /"
)


def plan_ranges(archive: Path, target_bytes: int) -> list[dict[str, int]]:
    if target_bytes < 512:
        raise ValueError("Layer target must be at least one TAR block (512 bytes)")
    ranges = []
    start = end = 0
    with tarfile.open(archive, "r:") as tar:
        for member in tar:
            if member.pax_headers or member.issparse():
                raise ValueError("Use the non-sparse GNU archive produced by the assembler")
            end = member.offset_data + ((member.size + 511) // 512) * 512
            if end - start > target_bytes and member.offset > start:
                ranges.append({"offset": start, "length": member.offset - start})
                start = member.offset
        if end > start:
            ranges.append({"offset": start, "length": end - start})
    if not ranges:
        raise ValueError("Payload archive is empty")
    # tarfile can stop silently at a corrupt non-first header. Do not interpret
    # that as a complete payload and publish only its valid prefix.
    with archive.open("rb") as stream:
        stream.seek(end)
        trailer = stream.read(1024)
        if len(trailer) != 1024 or any(trailer):
            raise ValueError("Invalid TAR trailer: missing end blocks or unparsed members")
        while padding := stream.read(1024 * 1024):
            if any(padding):
                raise ValueError("Invalid TAR trailer: nonzero data after end blocks")
    return ranges


def render_dockerfile(template: str, ranges: list[dict[str, int]]) -> str:
    if template.count(MARKER) != 1:
        raise ValueError("Dockerfile must contain exactly one payload extraction marker")
    instructions = []
    for index, item in enumerate(ranges, 1):
        instructions.append(
            f"# Curated payload layer {index}/{len(ranges)}; complete TAR members.\n"
            "RUN --mount=type=bind,from=payload,source=rootfs.tar,target=/tmp/rootfs.tar \\\n"
            "    bash -o pipefail -c 'dd if=/tmp/rootfs.tar bs=4M "
            f"iflag=skip_bytes,count_bytes skip={item['offset']} count={item['length']} "
            "status=none | tar -xf - -C /'"
        )
    return template.replace(MARKER, "\n".join(instructions))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--template", type=Path, default=Path("Dockerfile.integrated"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-gib", type=float, default=4.0)
    args = parser.parse_args()
    if args.output.resolve() in {args.archive.resolve(), args.template.resolve()}:
        parser.error("Output must be separate from the archive and template")
    ranges = plan_ranges(args.archive, int(args.target_gib * 1024**3))
    rendered = render_dockerfile(args.template.read_text(), ranges)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered)
    report = {"archive_bytes": args.archive.stat().st_size, "layers": ranges,
              "maximum_payload_layer_bytes": max(item["length"] for item in ranges)}
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(report, indent=2) + "\n")
    print(json.dumps({"dockerfile": str(args.output), "layers": len(ranges),
                      "maximum_payload_layer_bytes": report["maximum_payload_layer_bytes"]}), flush=True)


if __name__ == "__main__":
    main()
