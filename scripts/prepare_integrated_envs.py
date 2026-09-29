#!/usr/bin/env python3
"""Stage installed conda runtimes without retaining private host dependencies.

Supply machine-specific inputs separately; do not commit them. Example mapping:
{"environments": [{"source": "/HOST/env", "destination": "/opt/conda/envs/model"}],
 "path_map": {"/HOST/checkout": "/opt/engines/checkout"}}

Alternatively --inventory accepts the read-only environment inventory (a list
with prefix/conda fields). It skips colabfold, already provided by the base
image, and non-conda environments such as AlphaFold3. Source checkouts and
weights must be staged separately. --dry-run validates package inventories
without copying; final path checks run after packaging, before publication.

Requires conda-pack 0.9.x for staging. Uses its prefix relocation, but always
reads installed files, not cached originals that can discard local fixes.
NoArchive normally hardlinks; our copy-only backend explicitly disables that.
No source prefix or package cache is modified. An existing destination is an
error. Work is published atomically only after its configuration passes audit.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile


HOST_PATH = re.compile(r"/(?:home|Users)/[^/\s'\"]+|/file_server(?:/|$)")
SENSITIVE_NAMES = {".env", ".netrc", ".pypirc", ".condarc", "pip.conf", "pip.ini",
                   "credentials", "token", "tokens"}


def normalized_name(value):
    return re.sub(r"[-_.]+", "-", value).lower()


def excluded(relative, dropped):
    parts = PurePosixPath(relative).parts
    if relative == "conda-meta/history":
        return True  # Can contain authenticated installation commands.
    if "__pycache__" in parts or relative.endswith((".pyc", ".pyo")):
        return True
    if any(p in SENSITIVE_NAMES or p in {".git", ".ssh"} for p in parts):
        return True
    if parts[-1] == "direct_url.json":
        return True  # Installation provenance, never required to import a package.
    for part in parts:
        name = normalized_name(part)
        if any(name == d or name.startswith(d + "-") or
               ("editable" in name and d in name) for d in dropped):
            return True
    return False


def iter_files(root):
    """Yield actual files and symlinks, never recurse through directory links."""
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in list(dirs):
            path = Path(directory) / name
            if path.is_symlink():
                yield path
                dirs.remove(name)
        for name in files:
            yield Path(directory) / name


def complete_pip_replacements(source):
    """Return packages whose installed wheel RECORD accounts for present files."""
    found = set()
    for info in source.glob("lib/python*/site-packages/*.dist-info"):
        if not (info / "RECORD").is_file() or not (info / "METADATA").is_file():
            continue
        name = next((line[6:].strip() for line in (info / "METADATA").read_text(
            errors="replace").splitlines() if line.startswith("Name: ")), None)
        if not name:
            continue
        with (info / "RECORD").open(newline="") as handle:
            paths = [row[0] for row in csv.reader(handle) if row]
        required = [p for p in paths if not p.endswith((".pyc", ".pyo"))]
        valid = bool(required)
        for relative in required:
            path = info.parent / relative
            if not path.resolve().is_relative_to(source.resolve()) or not path.exists():
                valid = False
                break
        if valid:
            found.add(normalized_name(name))
    return found


def package_inventory(source, dropped):
    """Use installed conda metadata, even if the package cache was cleaned.

    Missing runtime files fail unless an installed, complete pip RECORD proves
    that package was replaced. In that case remove its stale conda record from
    the staged payload only. Missing bytecode is harmless and regenerated later.
    """
    modes, stale, repairs = {}, set(), []
    replacements = None
    for path in sorted((source / "conda-meta").glob("*.json")):
        record = json.loads(path.read_text())
        name = normalized_name(record["name"])
        if name in dropped:
            stale.add(path.relative_to(source).as_posix())
            continue
        paths = record.get("files", [])
        missing = [p for p in paths if not excluded(p, dropped)
                   and not os.path.lexists(source / p)]
        if missing:
            if replacements is None:
                replacements = complete_pip_replacements(source)
            wheel_name = {"numpy-base": "numpy"}.get(name, name)
            if wheel_name not in replacements:
                raise ValueError(f"missing managed files in {name}: {len(missing)}; "
                                 "no complete pip replacement")
            stale.add(path.relative_to(source).as_posix())
            repairs.append({"package": name, "action": "omit stale conda record", "missing": len(missing)})
            continue
        entries = record.get("paths_data", {}).get("paths", [])
        # Older records may keep relocation data only in the package cache.
        if not entries:
            cache = Path((record.get("link") or {}).get("source") or "/nonexistent")
            cached = cache / "info/paths.json"
            if cached.is_file():
                entries = json.loads(cached.read_text()).get("paths", [])
            elif (cache / "info/has_prefix").is_file():
                import shlex
                for line in (cache / "info/has_prefix").read_text().splitlines():
                    fields = shlex.split(line)
                    if len(fields) == 3:
                        entries.append({"_path": fields[2], "file_mode": fields[1],
                                        "prefix_placeholder": fields[0]})
        for entry in entries:
            if entry.get("prefix_placeholder"):
                modes[entry["_path"]] = entry.get("file_mode", "text")
    return modes, stale, repairs


def replace_paths(text, mapping):
    for old, new in sorted(mapping.items(), key=lambda item: len(item[0]), reverse=True):
        text = text.replace(old, new)
    return text


def repair_build_paths(text):
    """Normalize known conda build-machine leftovers, never arbitrary home paths."""
    build = (r"/home/(?:conda|task_[A-Za-z0-9_]+)/(?:feedstock_root/build_artifacts|croot|conda-bld)"
             r"/[^/\s'\"]+")
    text = re.sub(build + r"/_build_env/bin/(?:x86_64-conda-linux-gnu-)?", "/usr/bin/", text)
    text = re.sub(build + r"/_build_env(?:/x86_64-conda-linux-gnu)?", "/usr", text)
    return re.sub(build + r"/work", "/usr/local/src/conda-build", text)


def runtime_configuration(relative):
    path = PurePosixPath(relative)
    return (path.parts[0] == "bin" or "activate.d" in path.parts or
            "deactivate.d" in path.parts or path.suffix in {".pth", ".egg-link"} or
            path.name.startswith(("__editable__", "_editable_impl")))


def preflight_configuration(source, paths, mapping):
    for path in paths:
        relative = path.relative_to(source).as_posix()
        if path.is_symlink():
            if HOST_PATH.search(replace_paths(os.readlink(path), mapping)):
                raise ValueError(f"unmapped host path in symlink: {relative}")
        elif runtime_configuration(relative):
            with path.open("rb") as handle:
                raw = handle.read(16 * 1024 * 1024 + 1)
            if b"\0" in raw:
                continue
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                continue
            if len(raw) > 16 * 1024 * 1024:
                raise ValueError(f"oversized executable text needs review: {relative}")
            if HOST_PATH.search(repair_build_paths(replace_paths(text, mapping))):
                raise ValueError(f"unmapped host path in executable configuration: {relative}")


def repair_and_audit(root, mapping, dropped):
    """Repair editable loaders and launch configuration; fail on unresolved paths.

    Small UTF-8 runtime/source/config files are rewritten. Binary relocation is
    exclusively conda-pack's job; it must never be done by text replacement.
    License texts are preserved unchanged. Unknown private paths in executable
    startup configuration fail with filenames only, never file contents.
    """
    rewritten = 0
    old_paths = tuple(mapping)
    for path in iter_files(root):
        relative = path.relative_to(root).as_posix()
        if excluded(relative, dropped):
            raise ValueError(f"excluded payload remains: {relative}")
        if path.is_symlink():
            target = replace_paths(os.readlink(path), mapping)
            if HOST_PATH.search(target):
                raise ValueError(f"unmapped host path in symlink: {relative}")
            if target != os.readlink(path):
                path.unlink()
                path.symlink_to(target)
            continue
        if any("license" in p.lower() or "copying" in p.lower() for p in path.parts):
            continue
        # Large tensors and compiled libraries are not executable text config.
        if path.stat().st_size > 16 * 1024 * 1024:
            if runtime_configuration(relative):
                with path.open("rb") as handle:
                    if handle.read(2) == b"#!":
                        raise ValueError(f"oversized executable text needs review: {relative}")
            continue
        raw = path.read_bytes()
        if b"\0" in raw:
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        text = repair_build_paths(replace_paths(text, mapping))
        if runtime_configuration(relative) and HOST_PATH.search(text):
            raise ValueError(f"unmapped host path in executable configuration: {relative}")
        if any(old in text for old in old_paths):
            raise ValueError(f"unmapped host path remains: {relative}")
        if text.encode() != raw:
            path.write_text(text)
            rewritten += 1
    return rewritten


def audit_completeness(source, output, paths):
    """Check every selected payload entry, including binary sizes, before publish."""
    missing, damaged = [], []
    for path in paths:
        relative = path.relative_to(source)
        target = output / relative
        if not os.path.lexists(target):
            missing.append(relative.as_posix())
            continue
        if path.is_symlink():
            if not target.is_symlink():
                damaged.append(relative.as_posix())
            continue
        if not target.is_file() or target.is_symlink():
            damaged.append(relative.as_posix())
            continue
        expected_size = path.stat().st_size
        actual_size = target.stat().st_size
        if expected_size and not actual_size:
            damaged.append(relative.as_posix())
            continue
        # Prefix rewriting changes text length. Binary replacement preserves
        # length using NUL padding; unmanaged binaries are copied unchanged.
        with path.open("rb") as handle:
            binary = b"\0" in handle.read(4096)
        if binary and expected_size != actual_size:
            damaged.append(relative.as_posix())
    if missing or damaged:
        raise ValueError(f"incomplete staged environment: {len(missing)} missing, "
                         f"{len(damaged)} damaged entries; "
                         f"examples: {(missing + damaged)[:5]}")
    return len(paths)


def stage_environment(spec, rootfs, path_map, dry_run=False):
    source = Path(spec["source"]).absolute()
    destination = spec["destination"]
    dest = PurePosixPath(destination)
    if (dest.parent != PurePosixPath("/opt/conda/envs") or
            dest.name in {"", ".", ".."} or ".." in dest.parts):
        raise ValueError("destination must be /opt/conda/envs/<name>")
    if not (source / "conda-meta").is_dir():
        raise ValueError("source must be a conda prefix")
    rootfs = Path(rootfs).absolute()
    output = rootfs / destination.lstrip("/")
    if not output.resolve().is_relative_to(rootfs.resolve()):
        raise ValueError("destination escapes rootfs through a symlink")
    if output.exists() or output.is_symlink():
        raise ValueError(f"destination already exists: {destination}")
    if rootfs.resolve().is_relative_to(source.resolve()):
        raise ValueError("destination rootfs must not be inside source")
    dropped = {normalized_name(n) for n in spec.get("drop_packages", [])}
    if source.name == "BindCraft" or dest.name == "BindCraft":
        dropped.update({"pyrosetta", "rosetta"})
    if source.name == "esm_env" or dest.name == "esm_env":
        dropped.add("pioneer")
    modes, stale, repairs = package_inventory(source, dropped)
    paths = [p for p in iter_files(source)
             if not excluded(p.relative_to(source).as_posix(), dropped)
             and p.relative_to(source).as_posix() not in stale]
    mapping = dict(path_map)
    mapping[str(source)] = destination
    preflight_configuration(source, paths, mapping)
    result = {"name": dest.name, "destination": destination, "files": len(paths),
              "repairs": repairs, "dropped_packages": sorted(dropped), "dry_run": dry_run}
    if dry_run:
        return result
    import conda_pack
    from conda_pack import formats
    from conda_pack.core import File

    class CopyArchive(formats.NoArchive):
        def __enter__(self):
            self.copy_func = lambda src, dst: shutil.copy2(src, dst, follow_symlinks=False)
            return self

        def __exit__(self, exc_type, exc, traceback):
            # Upstream NoArchive returns self, swallowing OSError and even
            # KeyboardInterrupt. Never publish an interrupted archive as valid.
            return False

    files = []
    for path in paths:
        relative = path.relative_to(source).as_posix()
        mode = modes.get(relative, "unknown")
        if relative.startswith("conda-meta/"):
            mode = None  # conda-pack sanitizes installation/cache metadata.
        files.append(File(str(path), relative, is_conda=relative in modes,
                          file_mode=mode,
                          prefix_placeholder=str(source) if relative in modes else None))
    output.parent.mkdir(parents=True, exist_ok=True)
    # Temporary prefix is on the same filesystem for atomic publication. It is
    # never the relocation destination: paths embedded in files use destination.
    temporary = Path(tempfile.mkdtemp(prefix=f".{dest.name}-", dir=output.parent))
    original_archive = formats.NoArchive
    try:
        formats.NoArchive = CopyArchive
        conda_pack.CondaEnv(str(source), files).pack(
            output=str(temporary), format="no-archive", dest_prefix=destination,
            force=True, verbose=False)
        # conda-pack moves its empty archive tempfile into no-archive output.
        # Remove only that generated scratch file, never a source payload file.
        for scratch in temporary.glob("tmp*"):
            if (scratch.is_file() and not scratch.is_symlink() and
                    scratch.stat().st_size == 0 and not (source / scratch.name).exists()):
                scratch.unlink()
        result["rewritten_text_files"] = repair_and_audit(temporary, mapping, dropped)
        result["verified_files"] = audit_completeness(source, temporary, paths)
        temporary.chmod(0o755)
        temporary.rename(output)
    finally:
        formats.NoArchive = original_archive
        if temporary.exists():
            shutil.rmtree(temporary)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--rootfs", type=Path, required=True)
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    mapping = json.loads(args.mapping.read_text())
    environments = mapping.get("environments", [])
    if args.inventory:
        if environments:
            parser.error("choose mapping environments or --inventory, not both")
        for item in json.loads(args.inventory.read_text()):
            prefix = Path(item["prefix"])
            if item.get("conda") and prefix.name != "colabfold":
                environments.append({"source": str(prefix),
                                     "destination": f"/opt/conda/envs/{prefix.name}"})
    if args.only:
        environments = [e for e in environments if PurePosixPath(e["destination"]).name in args.only]
    if not environments:
        parser.error("no environments selected")
    if len({e["destination"] for e in environments}) != len(environments):
        parser.error("duplicate environment destination")
    for spec in environments:
        result = stage_environment(spec, args.rootfs, mapping.get("path_map", {}), args.dry_run)
        print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
