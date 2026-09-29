#!/usr/bin/env python3
"""Stage explicitly allowlisted public assets for an integrated image.

The external JSON's ``staging_entries`` contain kind (file, tree, git, or
hf_snapshot), source, and absolute in-image destination. HF entries also name
a revision and optional allowed_blob_roots. No network, hardlinks, or source
mutations are used. The default is a dry run; --execute copies after validating
the complete plan. Databases and user-obtained licensed materials stay external.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile


EXCLUDED_COMPONENTS = {
    ".git", ".env", ".netrc", ".npmrc", ".pypirc", ".ssh", ".aws",
    "token", "stored_tokens", "credentials", "credentials.json",
    ".venv", ".uv-cache", "__pycache__", "pyrosetta", "rosetta",
}
EXCLUDED_NAMES = (
    "*.partial", "*.incomplete", "*.pem", "*.key", "pyrosetta*.whl",
    "pyrosetta-*.dist-info", "af3.bin", "af3.bin.zst", "protenix-v2.pt",
)


def excluded(path: Path | PurePosixPath, patterns: list[str]) -> bool:
    return any(
        part in EXCLUDED_COMPONENTS
        or any(fnmatch.fnmatch(part, pattern) for pattern in EXCLUDED_NAMES)
        for part in path.parts
    ) or any(path.match(pattern) or any(fnmatch.fnmatch(p, pattern) for p in path.parts)
             for pattern in patterns)


def destination(rootfs: Path, value: str) -> Path:
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts or path == PurePosixPath("/"):
        raise ValueError(f"Invalid image destination: {value}")
    if excluded(path, []) or path.parts[:2] == ("/", "data"):
        raise ValueError(f"Restricted/external image destination: {value}")
    target = rootfs.joinpath(*path.parts[1:])
    current = target
    while current != rootfs.parent:
        if current.is_symlink():
            raise ValueError(f"Destination contains a symlink: {current}")
        current = current.parent
    return target


def plan_files(manifest: dict, rootfs: Path) -> list[tuple[Path, Path]]:
    """Expand allowlists and validate every file before creating anything."""
    planned: dict[Path, Path] = {}
    entries = manifest.get("staging_entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Manifest needs a nonempty staging_entries list")
    for entry in entries:
        source = Path(entry["source"]).absolute()
        target = destination(rootfs, entry["destination"])
        kind = entry["kind"]
        patterns = entry.get("exclude", [])
        allowed = [source.resolve() if source.is_dir() else source.parent.resolve()]
        pairs: list[tuple[Path, Path]] = []
        if excluded(source, []):
            raise ValueError(f"Restricted source: {source}")
        if kind == "file":
            pairs.append((source, target))
        elif kind in {"tree", "git", "hf_snapshot"}:
            if not source.is_dir():
                raise ValueError(f"Missing source directory: {source}")
            scan_root = source
            if kind == "hf_snapshot":
                revision = entry["revision"]
                if not revision or "/" in revision or revision in {".", ".."}:
                    raise ValueError("Invalid HF revision")
                scan_root = source / "snapshots" / revision
                if not scan_root.is_dir():
                    raise ValueError(f"Missing HF snapshot: {scan_root}")
                allowed.extend(Path(p).resolve() for p in entry.get("allowed_blob_roots", []))
                ref = source / "refs/main"
                if not ref.is_file() or ref.read_text().strip() != revision:
                    raise ValueError(f"HF refs/main does not match revision: {source}")
                pairs.append((ref, target / "refs/main"))
            if kind == "git":
                result = subprocess.run(
                    ["git", "-C", str(source), "ls-files", "-z", "--recurse-submodules"],
                    check=True, capture_output=True,
                )
                candidates = [source / os.fsdecode(name) for name in result.stdout.split(b"\0") if name]
            else:
                candidates = []
                for base, dirs, files in os.walk(scan_root, followlinks=False):
                    dirs[:] = [d for d in dirs if not excluded((Path(base) / d).relative_to(source), patterns)]
                    for directory in dirs:
                        if (Path(base) / directory).is_symlink():
                            raise ValueError(f"Directory symlink requires an explicit entry: {Path(base) / directory}")
                    candidates.extend(Path(base) / name for name in files)
            for candidate in candidates:
                relative = candidate.relative_to(source)
                if not excluded(relative, patterns):
                    if kind == "git" and candidate.is_symlink() and candidate.is_dir():
                        linked_root = candidate.resolve()
                        if not linked_root.is_relative_to(source.resolve()):
                            raise ValueError(f"Directory symlink escapes source: {candidate}")
                        # Materialize only already-tracked target resources; a
                        # tracked directory link must not admit untracked files.
                        for resource in candidates:
                            resolved_resource = resource.resolve()
                            if (resource.is_file()
                                    and resolved_resource.is_relative_to(linked_root)
                                    and not excluded(resource.relative_to(source), patterns)):
                                pairs.append((resource, target / relative / resolved_resource.relative_to(linked_root)))
                    else:
                        pairs.append((candidate, target / relative))
        else:
            raise ValueError(f"Unknown asset kind: {kind}")
        for original, output in pairs:
            resolved = original.resolve(strict=True)
            if not resolved.is_file():
                raise ValueError(f"Not a regular file: {original}")
            if excluded(resolved, []):
                raise ValueError(f"Restricted resolved source: {original}")
            if not any(resolved.is_relative_to(root) for root in allowed):
                raise ValueError(f"Symlink escapes allowed source roots: {original}")
            # Validate nested destinations too, including pre-existing symlinks.
            destination(rootfs, "/" + output.relative_to(rootfs).as_posix())
            if output in planned and planned[output] != resolved:
                raise ValueError(f"Conflicting sources for destination: {output}")
            planned[output] = resolved
    return [(source, target) for target, source in sorted(planned.items())]


def execute_plan(plan: list[tuple[Path, Path]], rootfs: Path) -> None:
    for source, target in plan:
        destination(rootfs, "/" + target.relative_to(rootfs).as_posix())
        target.parent.mkdir(parents=True, exist_ok=True)
        # copy2 creates an independent file. Atomic replacement avoids partial
        # checkpoint files being mistaken for finished assets after interruption.
        fd, temporary = tempfile.mkstemp(prefix=".asset-", dir=target.parent)
        os.close(fd)
        try:
            shutil.copy2(source, temporary)
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


def rewrite_staged_text(plan: list[tuple[Path, Path]], rootfs: Path, mapping: dict[str, str]) -> int:
    """Rewrite source/config text only; never deserialize or edit model files."""
    if not all(isinstance(k, str) and k and isinstance(v, str) for k, v in mapping.items()):
        raise ValueError("Path map must contain nonempty string keys and string values")
    if not mapping:
        return 0
    matcher = re.compile("|".join(re.escape(k) for k in sorted(mapping, key=len, reverse=True)))
    suffixes = {".py", ".yaml", ".yml", ".json", ".toml", ".cfg", ".ini",
                ".txt", ".md", ".sh", ".pth", ".egg-link"}
    changed = 0
    for _, target in plan:
        if target.suffix not in suffixes and target.name not in {"ckpt_root", "Makefile"}:
            continue
        destination(rootfs, "/" + target.relative_to(rootfs).as_posix())
        if not target.is_file():
            raise ValueError(f"Staged file missing before rewrite: {target}")
        if target.stat().st_size > 5 * 1024 * 1024:
            continue
        raw = target.read_bytes()
        if b"\0" in raw:
            continue
        try:
            original = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        revised = matcher.sub(lambda match: mapping[match.group()], original)
        if revised != original:
            target.write_bytes(revised.encode("utf-8"))
            changed += 1
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--rootfs", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="Actually copy; otherwise only validate and report")
    parser.add_argument("--path-map", type=Path, help="JSON mapping to rewrite in staged source/config text")
    parser.add_argument("--rewrite-only", action="store_true", help="Rewrite an already staged tree without copying weights")
    parser.add_argument("--plan-json", type=Path, help="Write the validated file plan for inspection")
    args = parser.parse_args()
    if args.rewrite_only and (args.execute or not args.path_map):
        parser.error("--rewrite-only requires --path-map and cannot be combined with --execute")
    try:
        rootfs = args.rootfs.absolute()
        if rootfs == Path("/"):
            raise ValueError("Refusing to stage into filesystem root")
        plan = plan_files(json.loads(args.manifest.read_text()), rootfs)
        summary = {"files": len(plan), "bytes": sum(source.stat().st_size for source, _ in plan),
                   "rootfs": str(rootfs), "executed": args.execute}
        if args.plan_json:
            args.plan_json.write_text(json.dumps([
                {"source": str(source), "destination": str(target), "bytes": source.stat().st_size}
                for source, target in plan], indent=2) + "\n")
        if args.execute:
            execute_plan(plan, rootfs)
        if args.path_map and (args.execute or args.rewrite_only):
            summary["rewritten_text_files"] = rewrite_staged_text(
                plan, rootfs, json.loads(args.path_map.read_text()))
        print(json.dumps(summary, indent=2))
        return 0
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        print(f"Asset staging failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
