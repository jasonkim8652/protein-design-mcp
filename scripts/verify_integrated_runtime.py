#!/usr/bin/env python3
"""Probe installed engine runtimes inside an image without inference/downloads.

Run with the server Python. Example:
  /opt/conda/envs/server/bin/python /app/scripts/verify_integrated_runtime.py \
      --output /tmp/runtime-imports.json
Use --engine TOOL, REPO, or ENV to repeat selected checks. Shared prefixes are
probed once. Missing external PyRosetta is reported as expected, not as a
successful PyRosetta import. This checks runtime imports, not model inference.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time


ALIASES = {"mpnn": "ligandmpnn", "openmm_minimize": "openmm", "prodigy": "prodigy_prot"}
BINARY_REPOS = {"mmseqs": "mmseqs", "pulchra": "pulchra"}
MARKER = "INTEGRATED_RUNTIME_JSON="
PROBE = r'''
import importlib,importlib.util,json,os,shutil,sys,traceback
request=json.loads(sys.argv[1]); records=[]
for item in request["imports"]:
    name=item["module"]
    try:
        if item.get("external") and importlib.util.find_spec(name) is None:
            records.append(dict(module=name,ok=True,status="expected_external_absent")); continue
        mod=importlib.import_module(name)
        location=getattr(mod,"__file__",None)
        if item.get("external") and location and not os.path.realpath(location).startswith("/data/"):
            records.append(dict(module=name,ok=False,status="restricted_package_bundled",file=location)); continue
        records.append(dict(module=name,ok=True,status="imported",file=location,version=getattr(mod,"__version__",None)))
    except BaseException as exc:
        records.append(dict(module=name,ok=False,status="import_failed",error=repr(exc),traceback=traceback.format_exc()))
for name in request["binaries"]:
    path=shutil.which(name)
    records.append(dict(binary=name,ok=path is not None,path=path))
torch_result={"installed":False}
try:
    if importlib.util.find_spec("torch") is not None:
        import torch
        torch_result=dict(installed=True,ok=True,version=torch.__version__,file=torch.__file__,
            compiled_cuda=torch.version.cuda,cuda_available=torch.cuda.is_available(),device_count=torch.cuda.device_count())
        if request.get("gpu_smoke"):
            if torch_result["cuda_available"]:
                tensor=torch.ones((8,8),device="cuda")
                value=(tensor@tensor).sum().item()
                torch.cuda.synchronize()
                torch_result["gpu_smoke"]={"ok":value==512.0,"result":value,"device":torch.cuda.get_device_name(0)}
                torch_result["ok"]=value==512.0
            else:
                torch_result["gpu_smoke"]={"status":"skipped_cuda_unavailable"}
except BaseException as exc:
    torch_result=dict(installed=True,ok=False,error=repr(exc),traceback=traceback.format_exc())
jax_result={"checked":False}
if request.get("gpu_smoke") and request.get("jax_runtime"):
    try:
        import jax
        import jax.numpy as jnp
        gpu_devices=[device for device in jax.devices() if device.platform=="gpu"]
        jax_result=dict(checked=True,ok=True,version=jax.__version__,gpu_available=bool(gpu_devices))
        if gpu_devices:
            tensor=jax.device_put(jnp.ones((2,),dtype=jnp.float32),gpu_devices[0])
            value=float(jnp.sum((tensor+tensor).block_until_ready()))
            jax_result["gpu_smoke"]={"ok":value==4.0,"result":value,"device":str(gpu_devices[0])}
            jax_result["ok"]=value==4.0
        else:
            jax_result["gpu_smoke"]={"status":"skipped_gpu_unavailable"}
    except BaseException as exc:
        jax_result=dict(checked=True,ok=False,error=repr(exc),traceback=traceback.format_exc())
ok=all(item["ok"] for item in records) and torch_result.get("ok",True) and jax_result.get("ok",True)
print("INTEGRATED_RUNTIME_JSON="+json.dumps(dict(ok=ok,executable=sys.executable,prefix=sys.prefix,checks=records,torch=torch_result,jax=jax_result)))
sys.exit(0 if ok else 1)
'''


def installed_version() -> str | None:
    try:
        return importlib.metadata.version("protein-design-mcp")
    except importlib.metadata.PackageNotFoundError:
        return None


def load_groups(manifest_dir: Path, filters: list[str]) -> list[dict]:
    import yaml

    groups = {}
    for path in sorted(manifest_dir.glob("*.yaml")):
        manifest = yaml.safe_load(path.read_text())
        engine = manifest.get("engine") or {}
        if not engine or manifest.get("composite"):
            continue
        repo = engine.get("repo", "")
        prefix = engine.get("prefix")
        env_name = engine.get("env")
        selectors = {manifest.get("name", path.stem), repo, env_name, Path(prefix).name if prefix else None}
        if filters and not any(value in selectors for value in filters):
            continue
        key = prefix or "env:" + str(env_name)
        group = groups.setdefault(key, {"runtime": key, "prefix": prefix, "environment": env_name,
                                       "tools": [], "imports": {}, "binaries": set(), "env_vars": {}, "external_paths": set()})
        group["tools"].append(manifest.get("name", path.stem))
        if repo in BINARY_REPOS:
            group["binaries"].add(BINARY_REPOS[repo])
        else:
            module = ALIASES.get(repo, repo)
            group["imports"][module] = {"module": module, "external": repo == "pyrosetta"}
        for name, value in (engine.get("env_vars") or {}).items():
            if name in group["env_vars"] and group["env_vars"][name] != str(value):
                if name == "PYTHONPATH":
                    group["env_vars"][name] += ":" + str(value)
                else:
                    raise ValueError(f"Conflicting {name} values in shared runtime {key}")
            else:
                group["env_vars"][name] = str(value)
        for value in engine.get("mounts", []):
            if value.startswith("/data/"):
                group["external_paths"].add(value)
        for value in (manifest.get("requires") or {}).get("files", []):
            if value.startswith("/data/"):
                group["external_paths"].add(value)
    return list(groups.values())


def probe(group: dict, timeout: float, gpu_smoke: bool = False) -> dict:
    request = {"imports": list(group["imports"].values()), "binaries": sorted(group["binaries"]),
               "gpu_smoke": gpu_smoke, "jax_runtime": bool({"alphafold3", "colabfold"} & group["imports"].keys())}
    prefix = group["prefix"]
    if prefix:
        command = [str(Path(prefix) / "bin/python")]
    else:
        micromamba = shutil.which("micromamba")
        command = [micromamba or "micromamba", "run", "-n", group["environment"], "python"]
    result = {"runtime": group["runtime"], "tools": group["tools"], "command": command,
              "external_paths": [{"path": p, "exists": Path(p).exists()} for p in sorted(group["external_paths"])],
              "env_vars": group["env_vars"]}
    env = os.environ.copy()
    env.update(group["env_vars"])
    # A probe cannot fetch models even if an imported package requests them.
    env.update(PYTHONNOUSERSITE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               WANDB_MODE="disabled", MPLCONFIGDIR="/tmp/pdmcp-probe-matplotlib")
    if prefix:
        env["PATH"] = str(Path(prefix) / "bin") + os.pathsep + env.get("PATH", "")
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="pdmcp-import-") as workdir:
        env["HOME"] = workdir
        env["XDG_CACHE_HOME"] = str(Path(workdir) / ".cache")
        try:
            proc = subprocess.run(command + ["-c", PROBE, json.dumps(request)], env=env, cwd=workdir,
                                  capture_output=True, text=True, timeout=timeout)
            result.update(returncode=proc.returncode, stdout=proc.stdout[-16000:], stderr=proc.stderr[-16000:])
            payloads = [line[len(MARKER):] for line in proc.stdout.splitlines() if line.startswith(MARKER)]
            result["probe"] = json.loads(payloads[-1]) if payloads else None
            result["ok"] = proc.returncode == 0 and bool(result["probe"] and result["probe"].get("ok"))
        except subprocess.TimeoutExpired as exc:
            result.update(ok=False, timed_out=True, error=f"Exceeded {timeout:g} seconds",
                          stdout=str(exc.stdout or "")[-16000:], stderr=str(exc.stderr or "")[-16000:])
        except OSError as exc:
            result.update(ok=False, error=str(exc))
    result["elapsed_seconds"] = round(time.monotonic() - started, 3)
    return result


def exclusion_details() -> dict:
    bundled = []
    external_links = []
    root = Path("/opt/conda/envs")
    if root.exists():
        for pattern in ["*/lib/python*/site-packages/pyrosetta", "*/lib/python*/site-packages/rosetta"]:
            for path in root.glob(pattern):
                resolved = path.resolve()
                if path.is_symlink() and resolved.is_relative_to("/data"):
                    external_links.append({"path": str(path), "target": str(resolved), "target_exists": resolved.exists()})
                else:
                    bundled.append(str(path))
    for root in [Path("/opt/models"), Path("/opt/alphafold3_data/weights")]:
        if root.exists():
            for name in ["af3.bin", "af3.bin.zst", "protenix-v2.pt"]:
                bundled.extend(str(p) for p in root.rglob(name))
    return {"ok": not bundled, "bundled_restricted_materials": sorted(set(bundled)), "external_package_links": external_links,
            "external_locations": [{"path": p, "exists": Path(p).exists()} for p in
                                   ["/data/models/alphafold3", "/data/licenses/pyrosetta", "/data/databases"]],
            "note": "External files may be absent. This is a known-path exclusion check, not a full image-layer audit."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, default=Path("/app/src/protein_design_mcp/manifests"))
    parser.add_argument("--engine", action="append", default=[], help="Tool name, repo name, or environment name; repeatable")
    parser.add_argument("--timeout", type=float, default=90, help="Seconds per environment (default: 90)")
    parser.add_argument("--gpu-smoke", action="store_true", help="Run tiny CUDA matmul/JAX GPU addition when available; no model inference")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {"scope": "Installed engine imports, CUDA availability, and optional tiny GPU operations; no model inference", "gpu_smoke_requested": args.gpu_smoke, "server_package_version": installed_version(),
              "verifier_python": sys.executable, "checks": []}
    try:
        if args.timeout <= 0:
            raise ValueError("Timeout must be positive")
        groups = load_groups(args.manifest_dir, args.engine)
        if not groups:
            raise ValueError("No engine manifests matched")
        for group in groups:
            result = probe(group, args.timeout, args.gpu_smoke)
            report["checks"].append(result)
            print(json.dumps({"runtime": result["runtime"], "ok": result["ok"]}), file=sys.stderr, flush=True)
        report["external_exclusions"] = exclusion_details()
        report["ok"] = all(r["ok"] for r in report["checks"]) and report["external_exclusions"]["ok"] and report["server_package_version"] is not None
    except (OSError, ValueError, KeyError) as exc:
        report.update(ok=False, error=str(exc))
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded)
    print(encoded, end="")
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
