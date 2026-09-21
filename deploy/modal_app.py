"""
Modal deployment for Protein Design MCP Server.

Deploy to your own Modal account for serverless GPU access.
No GPU on your machine? No problem — Modal provisions A10G GPUs on demand.

Setup (one-time):
    pip install modal
    modal setup

Deploy:
    cd protein-design-mcp
    modal deploy deploy/modal_app.py

Test:
    modal run deploy/modal_app.py

Then connect the local MCP proxy:
    MODAL_URL=https://<your-workspace>--protein-design-tools.modal.run \
        python -m protein_design_mcp.modal_proxy

Cost: ~$1.10/hr (A10G), billed per-second. Containers auto-stop after 5 min idle.
"""

import os
from pathlib import Path

import modal

app = modal.App("protein-design-mcp")

# Persistent volume for model weights — survives container restarts
weights_volume = modal.Volume.from_name(
    "protein-design-weights", create_if_missing=True
)

# ---------------------------------------------------------------------------
# GPU Image (mirrors Dockerfile — all 11 tools available)
# ---------------------------------------------------------------------------
gpu_image = (
    modal.Image.from_registry(
        "nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install("git", "wget", "build-essential")
    # PyTorch (CUDA 11.8). `extra_index_url` (not `index_url`) so pip
    # still falls back to the default PyPI for build-time deps
    # (setuptools, wheel) -- the pytorch wheel index doesn't mirror them.
    # numpy<2 is pinned here AND in every subsequent pip_install below
    # because torch 2.0.1 was compiled against numpy<2; a later layer
    # pulling numpy 2.x triggers "Failed to initialize NumPy: _ARRAY_API
    # not found" at runtime.
    .pip_install(
        "torch==2.0.1+cu118",
        "torchvision==0.15.2+cu118",
        "numpy<2",
        extra_index_url="https://download.pytorch.org/whl/cu118",
    )
    # --- RFdiffusion ---
    .run_commands(
        "git clone --depth 1 https://github.com/RosettaCommons/RFdiffusion.git /opt/RFdiffusion",
        "cd /opt/RFdiffusion/env/SE3Transformer && pip install --no-cache-dir -e .",
        "cd /opt/RFdiffusion && pip install --no-cache-dir -e .",
    )
    .pip_install("hydra-core", "omegaconf", "e3nn==0.5.1", "numpy<2")
    .pip_install(
        "dgl==1.1.3",
        find_links="https://data.dgl.ai/wheels/cu118/repo.html",
    )
    # --- ProteinMPNN ---
    .run_commands(
        "git clone --depth 1 https://github.com/dauparas/ProteinMPNN.git /opt/ProteinMPNN",
    )
    # --- OpenFold v1.0.1 (ESMFold dependency) ---
    .run_commands(
        "git clone --depth 1 --branch v1.0.1 "
        "https://github.com/aqlaboratory/openfold.git /opt/openfold",
    )
    .pip_install("biopython", "dm-tree", "ml-collections", "modelcif", "einops", "numpy<2")
    # ColabFold / AlphaFold2 install removed on 2026-04-15: the source
    # never imports `colabfold` or `alphafold` directly (structure
    # prediction uses ESMFold via protein_design_mcp.pipelines.esmfold),
    # and colabfold[alphafold] 1.6.1 hard-pins numpy>=2.0.2 which
    # conflicts with torch 2.0.1's numpy<2 requirement.
    # --- OpenMM ---
    .pip_install("openmm", "numpy<2")
    # --- ESM + MCP server deps ---
    .pip_install(
        "fair-esm>=2.0.0",
        "aiohttp",
        "tqdm",
        "aiofiles",
        "pyyaml",
        "mcp>=0.1.0",
        # Required by Modal's @fastapi_endpoint decorator. Must be in
        # the image, not just the local driver env.
        "fastapi[standard]",
        "numpy<2",
    )
    # Environment variables must be set BEFORE add_local_python_source:
    # Modal forbids any build step after add_local_*.
    .env(
        {
            "RFDIFFUSION_PATH": "/opt/RFdiffusion",
            "PROTEINMPNN_PATH": "/opt/ProteinMPNN",
            "PYTHONPATH": "/opt/openfold",
            "PYTORCH_JIT_USE_NNC_NOT_NVFUSER": "1",
            "DS_BUILD_OPS": "0",
            "DEVICE": "cuda",
            "MODELS_DIR": "/models",
            "TORCH_HOME": "/models/esm",
            "COLABFOLD_BACKEND": "api",
        }
    )
    # --- Install protein-design-mcp package (must be last build step) ---
    # From local source (requires `pip install -e .` locally first):
    .add_local_python_source("protein_design_mcp")
    # When published to PyPI, replace the line above with:
    # .pip_install("protein-design-mcp")
)


# ---------------------------------------------------------------------------
# Helper: materialize inlined file contents into temp files
# ---------------------------------------------------------------------------
def _materialize_files(arguments: dict) -> dict:
    """Write inlined file contents (from proxy) to local temp files.

    The MCP proxy reads local PDB files and sends their content as
    ``_file_<argname>`` keys. This function writes them to temp files
    so the tool handlers can use regular file paths.
    """
    import tempfile

    result = dict(arguments)
    for key in list(result.keys()):
        if not key.startswith("_file_"):
            continue
        real_key = key[6:]  # "_file_target_pdb" -> "target_pdb"
        content = result.pop(key)
        # Write to a temp file
        suffix = ".pdb" if "pdb" in real_key else ".txt"
        fd, path = tempfile.mkstemp(suffix=suffix, prefix=f"{real_key}_")
        with os.fdopen(fd, "w") as f:
            f.write(content)
        result[real_key] = path
    return result


def _json_default(o):
    """Fallback encoder for json.dumps: convert numpy / non-standard
    objects to JSON-compatible types."""
    try:
        import numpy as np

        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.generic):
            return o.item()
    except Exception:
        pass
    if hasattr(o, "tolist"):
        try:
            return o.tolist()
        except Exception:
            pass
    return str(o)


def _json_safe(obj):
    """Force an object through a json.dumps/json.loads roundtrip with a
    fallback encoder, so anything that leaks out of a handler (numpy
    arrays, numpy scalars, tensors, dataclasses) is guaranteed to be
    plain Python by the time FastAPI's pydantic serializer sees it.

    We hit PydanticSerializationError: Unable to serialize unknown type:
    <class 'numpy.ndarray'> with a manual recursive walker because the
    walker only descended dict/list containers; switching to a json
    roundtrip sidesteps any custom object tree.
    """
    import json as _json

    try:
        return _json.loads(_json.dumps(obj, default=_json_default))
    except Exception:
        return {"error": "tool returned non-JSON object", "repr": repr(obj)[:500]}


# ---------------------------------------------------------------------------
# Web endpoint — single entry point for all tool calls
# ---------------------------------------------------------------------------
@app.function(
    image=gpu_image,
    gpu="A10G",
    volumes={"/models": weights_volume},
    timeout=1800,
    scaledown_window=300,
)
@modal.fastapi_endpoint(method="POST", label="protein-design-tools", docs=True)
async def call_tool(item: dict) -> dict:
    """Execute a protein design tool on GPU.

    Request body: ``{"name": "predict_structure", "arguments": {...}}``
    """
    name = item.get("name")
    arguments = item.get("arguments", {})

    if not name:
        return {"error": "Missing 'name' in request body"}

    # This deployment still dispatches the old hardcoded handle_* functions
    # (design_binder, analyze_interface, ...), which were deleted from
    # protein_design_mcp.server when the server moved to the manifest-driven
    # ToolRegistry/ServerApp (see src/protein_design_mcp/app.py and
    # manifests/*.yaml). Porting this Modal deployment to the new
    # manifest-driven surface is deliberately deferred to a later plan, so
    # fail early and explicitly here rather than hitting an ImportError deep
    # in a lazy import.
    return {
        "error": (
            f"deploy/modal_app.py is obsolete: it still dispatches the "
            f"pre-manifest tool named {name!r}, which no longer exists. "
            "The server now derives its tool surface from "
            "protein_design_mcp.app.ServerApp and manifests/*.yaml. This "
            "Modal deployment has not been ported to that surface yet; "
            "redeploy is tracked as a follow-up. Do not call this endpoint "
            "until it has been updated."
        ),
        "tool": name,
    }


# ---------------------------------------------------------------------------
# CLI test entrypoint: modal run deploy/modal_app.py
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def test():
    """Quick smoke test — score stability of a short peptide."""
    import json

    print("Testing protein-design-mcp on Modal...")
    result = call_tool.remote(
        {"name": "score_stability", "arguments": {"sequence": "MKTLYVGDAKEF" * 5}}
    )
    print(json.dumps(result, indent=2))
    print("\nDeployment is working! Your endpoint URL:")
    print("  https://<your-workspace>--protein-design-tools.modal.run")
